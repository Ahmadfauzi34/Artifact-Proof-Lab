from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path
import threading
import unittest

from gym.agent_socket_server import ReferenceAgentTCPServer
from gym.attested_external import run_attested_external_episode
from gym.authority_attestation import (
    AUTHORITY_ATTESTATION_FORMAT,
    REFERENCE_ASSURANCE,
    AuthorityAttestationError,
    AuthorityAttestationReceipt,
    AuthorityAttestationRequest,
    AuthorityClaims,
    ReferenceHMACAuthorityAttestor,
    ReferenceHMACAuthorityVerifier,
)
from gym.core import MemoryLearningSink, ReferenceGym
from gym.external_agent import ExternalAgentEndpoint
from gym.host import create_environment
from gym.io import load_task
from gym.ledger import sha256_json
from gym.private_pack import PrivateHoldoutPack


ROOT = Path(__file__).resolve().parents[1]
GYM_ROOT = ROOT / "gym"
REFERENCE_PACK = GYM_ROOT / "private_holdout_reference"
KEY_A = b"A" * 32
KEY_B = b"B" * 32


class _RunningReferenceServer:
    def __init__(self):
        self.server = ReferenceAgentTCPServer(("127.0.0.1", 0))
        self.thread = threading.Thread(
            target=self.server.serve_forever,
            daemon=True,
        )

    @property
    def address(self):
        host, port = self.server.server_address
        return str(host), int(port)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2.0)


def _task():
    return load_task(GYM_ROOT / "tasks" / "train" / "tg_a7.json")


def _request() -> AuthorityAttestationRequest:
    task = _task()
    return AuthorityAttestationRequest.create(
        peer_host="127.0.0.1",
        peer_port=8765,
        host_task_commitment=sha256_json(asdict(task)),
        agent_task_view_sha256=sha256_json(asdict(task.agent_view())),
        evaluation_context_sha256=sha256_json(
            {"scope": "g25-test-context"}
        ),
    )


def _claims() -> AuthorityClaims:
    return AuthorityClaims(
        filesystem_isolated=True,
        kernel_isolated=True,
        credential_isolated=True,
        network_isolated=False,
        machine_identity_verified=False,
        model_identity_verified=False,
    )


class GymG25Tests(unittest.TestCase):
    def test_request_uses_fresh_challenge_and_exact_bindings(self):
        first = _request()
        second = _request()
        self.assertNotEqual(first.challenge, second.challenge)
        self.assertEqual(len(first.challenge), 64)
        self.assertEqual(len(first.endpoint_locator_sha256), 64)
        self.assertEqual(len(first.host_task_commitment), 64)
        self.assertEqual(len(first.agent_task_view_sha256), 64)
        self.assertEqual(len(first.evaluation_context_sha256), 64)

    def test_valid_reference_hmac_receipt_verifies(self):
        request = _request()
        attestor = ReferenceHMACAuthorityAttestor(
            KEY_A,
            claims=_claims(),
        )
        verifier = ReferenceHMACAuthorityVerifier(KEY_A)

        receipt = attestor.attest(request)
        verification = verifier.verify(request, receipt)

        self.assertTrue(verification.valid, verification.reason)
        self.assertEqual(
            verification.assurance,
            REFERENCE_ASSURANCE,
        )
        self.assertTrue(
            verification.claims.filesystem_isolated
        )
        self.assertTrue(
            verification.claims.kernel_isolated
        )
        self.assertFalse(
            verification.claims.machine_identity_verified
        )

    def test_claim_tamper_breaks_authentication(self):
        request = _request()
        receipt = ReferenceHMACAuthorityAttestor(
            KEY_A,
            claims=_claims(),
        ).attest(request)
        tampered = replace(
            receipt,
            claims=replace(
                receipt.claims,
                filesystem_isolated=False,
            ),
        )
        verification = ReferenceHMACAuthorityVerifier(
            KEY_A
        ).verify(request, tampered)
        self.assertFalse(verification.valid)
        self.assertIn("MAC mismatch", verification.reason)

    def test_request_binding_mismatches_are_rejected(self):
        request = _request()
        receipt = ReferenceHMACAuthorityAttestor(
            KEY_A,
            claims=_claims(),
        ).attest(request)

        cases = [
            replace(request, challenge="1" * 64),
            replace(
                request,
                endpoint_locator_sha256="2" * 64,
            ),
            replace(
                request,
                host_task_commitment="3" * 64,
            ),
            replace(
                request,
                agent_task_view_sha256="4" * 64,
            ),
            replace(
                request,
                evaluation_context_sha256="5" * 64,
            ),
        ]
        expected = [
            "challenge mismatch",
            "endpoint_locator_sha256 mismatch",
            "host_task_commitment mismatch",
            "agent_task_view_sha256 mismatch",
            "evaluation_context_sha256 mismatch",
        ]
        for bad_request, message in zip(cases, expected):
            verification = ReferenceHMACAuthorityVerifier(
                KEY_A
            ).verify(bad_request, receipt)
            self.assertFalse(verification.valid)
            self.assertIn(message, verification.reason)

    def test_valid_receipt_is_one_time_and_replay_fails(self):
        request = _request()
        receipt = ReferenceHMACAuthorityAttestor(
            KEY_A,
            claims=_claims(),
        ).attest(request)
        verifier = ReferenceHMACAuthorityVerifier(KEY_A)

        first = verifier.verify(request, receipt)
        second = verifier.verify(request, receipt)

        self.assertTrue(first.valid)
        self.assertFalse(second.valid)
        self.assertIn("replayed", second.reason)

    def test_receipt_parser_rejects_extra_fields(self):
        request = _request()
        receipt = ReferenceHMACAuthorityAttestor(
            KEY_A,
            claims=_claims(),
        ).attest(request)
        raw = receipt.to_dict()
        raw["agent_claimed_truth"] = True

        with self.assertRaises(AuthorityAttestationError):
            AuthorityAttestationReceipt.from_mapping(raw)

    def test_invalid_attestation_stops_before_agent_start_and_learning(self):
        task = _task()
        learning = MemoryLearningSink()

        class TrackingEndpoint(ExternalAgentEndpoint):
            def __init__(self, host, port):
                super().__init__(host, port)
                self.start_calls = 0

            def start(self, task_view):
                self.start_calls += 1
                return super().start(task_view)

        with _RunningReferenceServer() as running:
            host, port = running.address
            endpoint = TrackingEndpoint(host, port)
            try:
                with self.assertRaises(
                    AuthorityAttestationError
                ) as caught:
                    run_attested_external_episode(
                        ReferenceGym(create_environment),
                        task,
                        endpoint,
                        evaluation_context_sha256=sha256_json(
                            {"scope": "invalid-attestation-test"}
                        ),
                        attestor=ReferenceHMACAuthorityAttestor(
                            KEY_A,
                            claims=_claims(),
                        ),
                        verifier=ReferenceHMACAuthorityVerifier(
                            KEY_B
                        ),
                        learning_sink=learning,
                    )
                self.assertIn(
                    "MAC mismatch",
                    str(caught.exception),
                )
                self.assertEqual(endpoint.start_calls, 0)
                self.assertEqual(learning.updates, [])
                with self.assertRaises(Exception):
                    endpoint.ping()
            finally:
                endpoint.disconnect()

    def test_valid_attestation_gates_train_learning(self):
        task = _task()
        learning = MemoryLearningSink()

        with _RunningReferenceServer() as running:
            host, port = running.address
            with ExternalAgentEndpoint(host, port) as endpoint:
                result = run_attested_external_episode(
                    ReferenceGym(create_environment),
                    task,
                    endpoint,
                    evaluation_context_sha256=sha256_json(
                        {"scope": "train-attested-test"}
                    ),
                    attestor=ReferenceHMACAuthorityAttestor(
                        KEY_A,
                        claims=_claims(),
                    ),
                    verifier=ReferenceHMACAuthorityVerifier(
                        KEY_A
                    ),
                    learning_sink=learning,
                )

        self.assertTrue(result.accepted)
        self.assertTrue(result.authority.valid)
        self.assertEqual(
            result.authority.assurance,
            REFERENCE_ASSURANCE,
        )
        self.assertTrue(result.episode.learning_updated)
        self.assertEqual(learning.updates, [task.task_id])
        self.assertEqual(
            result.episode.ledger_entries[0].payload[
                "host_task_commitment"
            ],
            sha256_json(asdict(task)),
        )
        self.assertEqual(
            result.episode.ledger_entries[0].payload[
                "agent_task_view_sha256"
            ],
            sha256_json(asdict(task.agent_view())),
        )

    def test_attested_private_holdout_never_updates_learning(self):
        pack = PrivateHoldoutPack.load(
            REFERENCE_PACK,
            expected_runtime_id="reference-v1",
        )
        task = pack.task("hg_f8")
        learning = MemoryLearningSink()

        with _RunningReferenceServer() as running:
            host, port = running.address
            with ExternalAgentEndpoint(host, port) as endpoint:
                result = run_attested_external_episode(
                    ReferenceGym(create_environment),
                    task,
                    endpoint,
                    evaluation_context_sha256=pack.commitment_sha256,
                    attestor=ReferenceHMACAuthorityAttestor(
                        KEY_A,
                        claims=_claims(),
                    ),
                    verifier=ReferenceHMACAuthorityVerifier(
                        KEY_A
                    ),
                    learning_sink=learning,
                )

        self.assertTrue(result.accepted)
        self.assertTrue(result.authority.valid)
        self.assertFalse(result.episode.learning_updated)
        self.assertEqual(learning.updates, [])

    def test_reference_assurance_is_explicitly_non_native(self):
        runner = (
            GYM_ROOT / "run_attested_external_reference.py"
        ).read_text()
        self.assertIn(
            '"native_competence_claim": False',
            runner,
        )
        self.assertIn(
            '"authority_independence_verified": False',
            runner,
        )
        self.assertIn(
            '"authority_cryptography": '
            '"shared_hmac_reference_only"',
            runner,
        )
        self.assertEqual(
            REFERENCE_ASSURANCE,
            "reference_hmac_conformance_only",
        )

    def test_authority_schema_is_strict_json_schema(self):
        schema = json.loads(
            (
                GYM_ROOT
                / "contracts"
                / "authority_attestation.schema.json"
            ).read_text()
        )
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(
            schema["properties"]["format"]["const"],
            AUTHORITY_ATTESTATION_FORMAT,
        )


if __name__ == "__main__":
    unittest.main()
