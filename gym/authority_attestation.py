from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import hmac
import secrets
from typing import Any, Mapping, Protocol

from .ledger import sha256_json


AUTHORITY_ATTESTATION_FORMAT = "proof-gym-authority-attestation-v1"
REFERENCE_ASSURANCE = "reference_hmac_conformance_only"


class AuthorityAttestationError(RuntimeError):
    """Authority evidence was malformed, invalid, replayed, or cross-bound wrong."""


@dataclass(frozen=True)
class AuthorityClaims:
    filesystem_isolated: bool
    kernel_isolated: bool
    credential_isolated: bool
    network_isolated: bool
    machine_identity_verified: bool
    model_identity_verified: bool

    def to_dict(self) -> dict[str, bool]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "AuthorityClaims":
        expected = {
            "filesystem_isolated",
            "kernel_isolated",
            "credential_isolated",
            "network_isolated",
            "machine_identity_verified",
            "model_identity_verified",
        }
        if set(value) != expected:
            raise AuthorityAttestationError(
                "authority claims fields mismatch"
            )
        for key in expected:
            if not isinstance(value.get(key), bool):
                raise AuthorityAttestationError(
                    f"authority claim {key} must be boolean"
                )
        return cls(**{key: bool(value[key]) for key in expected})

    @classmethod
    def none_verified(cls) -> "AuthorityClaims":
        return cls(
            filesystem_isolated=False,
            kernel_isolated=False,
            credential_isolated=False,
            network_isolated=False,
            machine_identity_verified=False,
            model_identity_verified=False,
        )


@dataclass(frozen=True)
class AuthorityAttestationRequest:
    challenge: str
    endpoint_locator_sha256: str
    host_task_commitment: str
    agent_task_view_sha256: str
    evaluation_context_sha256: str

    @classmethod
    def create(
        cls,
        *,
        peer_host: str,
        peer_port: int,
        host_task_commitment: str,
        agent_task_view_sha256: str,
        evaluation_context_sha256: str,
    ) -> "AuthorityAttestationRequest":
        if not isinstance(peer_host, str) or not peer_host:
            raise AuthorityAttestationError(
                "authority peer host must be non-empty"
            )
        if (
            isinstance(peer_port, bool)
            or not isinstance(peer_port, int)
            or not (1 <= peer_port <= 65535)
        ):
            raise AuthorityAttestationError(
                "authority peer port must be in 1..65535"
            )
        _require_sha256(host_task_commitment, "host_task_commitment")
        _require_sha256(agent_task_view_sha256, "agent_task_view_sha256")
        _require_sha256(
            evaluation_context_sha256,
            "evaluation_context_sha256",
        )
        locator_sha = sha256_json(
            {
                "transport": "proof-gym-agent-jsonl-v1-over-tcp",
                "peer_host": peer_host,
                "peer_port": peer_port,
            }
        )
        return cls(
            challenge=secrets.token_hex(32),
            endpoint_locator_sha256=locator_sha,
            host_task_commitment=host_task_commitment,
            agent_task_view_sha256=agent_task_view_sha256,
            evaluation_context_sha256=evaluation_context_sha256,
        )

    def __post_init__(self) -> None:
        _require_sha256(self.challenge, "challenge")
        _require_sha256(
            self.endpoint_locator_sha256,
            "endpoint_locator_sha256",
        )
        _require_sha256(
            self.host_task_commitment,
            "host_task_commitment",
        )
        _require_sha256(
            self.agent_task_view_sha256,
            "agent_task_view_sha256",
        )
        _require_sha256(
            self.evaluation_context_sha256,
            "evaluation_context_sha256",
        )

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class AuthorityAttestationReceipt:
    format: str
    issuer_id: str
    assurance: str
    challenge: str
    endpoint_locator_sha256: str
    host_task_commitment: str
    agent_task_view_sha256: str
    evaluation_context_sha256: str
    claims: AuthorityClaims
    evidence_sha256: str
    mac_sha256: str

    def signed_payload(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "issuer_id": self.issuer_id,
            "assurance": self.assurance,
            "challenge": self.challenge,
            "endpoint_locator_sha256": self.endpoint_locator_sha256,
            "host_task_commitment": self.host_task_commitment,
            "agent_task_view_sha256": self.agent_task_view_sha256,
            "evaluation_context_sha256": self.evaluation_context_sha256,
            "claims": self.claims.to_dict(),
            "evidence_sha256": self.evidence_sha256,
        }

    def to_dict(self) -> dict[str, Any]:
        value = self.signed_payload()
        value["mac_sha256"] = self.mac_sha256
        return value

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
    ) -> "AuthorityAttestationReceipt":
        expected = {
            "format",
            "issuer_id",
            "assurance",
            "challenge",
            "endpoint_locator_sha256",
            "host_task_commitment",
            "agent_task_view_sha256",
            "evaluation_context_sha256",
            "claims",
            "evidence_sha256",
            "mac_sha256",
        }
        if set(value) != expected:
            raise AuthorityAttestationError(
                "authority receipt fields mismatch"
            )
        claims_raw = value.get("claims")
        if not isinstance(claims_raw, dict):
            raise AuthorityAttestationError(
                "authority receipt claims must be an object"
            )
        receipt = cls(
            format=_required_string(value, "format"),
            issuer_id=_required_string(value, "issuer_id"),
            assurance=_required_string(value, "assurance"),
            challenge=_required_string(value, "challenge"),
            endpoint_locator_sha256=_required_string(
                value,
                "endpoint_locator_sha256",
            ),
            host_task_commitment=_required_string(
                value,
                "host_task_commitment",
            ),
            agent_task_view_sha256=_required_string(
                value,
                "agent_task_view_sha256",
            ),
            evaluation_context_sha256=_required_string(
                value,
                "evaluation_context_sha256",
            ),
            claims=AuthorityClaims.from_mapping(claims_raw),
            evidence_sha256=_required_string(
                value,
                "evidence_sha256",
            ),
            mac_sha256=_required_string(value, "mac_sha256"),
        )
        receipt.validate_shape()
        return receipt

    def validate_shape(self) -> None:
        if self.format != AUTHORITY_ATTESTATION_FORMAT:
            raise AuthorityAttestationError(
                "unsupported authority attestation format"
            )
        _require_sha256(self.challenge, "challenge")
        _require_sha256(
            self.endpoint_locator_sha256,
            "endpoint_locator_sha256",
        )
        _require_sha256(
            self.host_task_commitment,
            "host_task_commitment",
        )
        _require_sha256(
            self.agent_task_view_sha256,
            "agent_task_view_sha256",
        )
        _require_sha256(
            self.evaluation_context_sha256,
            "evaluation_context_sha256",
        )
        _require_sha256(
            self.evidence_sha256,
            "evidence_sha256",
        )
        _require_sha256(self.mac_sha256, "mac_sha256")


@dataclass(frozen=True)
class AuthorityVerification:
    valid: bool
    reason: str
    issuer_id: str | None
    assurance: str | None
    claims: AuthorityClaims
    evidence_sha256: str | None


class AuthorityAttestor(Protocol):
    def attest(
        self,
        request: AuthorityAttestationRequest,
    ) -> AuthorityAttestationReceipt: ...


class AuthorityVerifier(Protocol):
    def verify(
        self,
        request: AuthorityAttestationRequest,
        receipt: AuthorityAttestationReceipt,
    ) -> AuthorityVerification: ...


class ReferenceHMACAuthorityAttestor:
    """Reference-only receipt issuer for contract tests.

    HMAC is used to exercise authenticity/tamper/replay mechanics with stdlib
    only. It is not an independent hardware/cloud authority attestation scheme.
    """

    def __init__(
        self,
        key: bytes,
        *,
        issuer_id: str = "reference-authority-attestor",
        claims: AuthorityClaims | None = None,
        evidence_sha256: str | None = None,
    ) -> None:
        self._key = _require_key(key)
        if not isinstance(issuer_id, str) or not issuer_id:
            raise ValueError("issuer_id must be non-empty")
        self._issuer_id = issuer_id
        self._claims = claims or AuthorityClaims.none_verified()
        self._evidence_sha256 = (
            evidence_sha256
            or sha256_json(
                {
                    "scope": "reference_authority_fixture",
                    "issuer_id": issuer_id,
                }
            )
        )
        _require_sha256(
            self._evidence_sha256,
            "evidence_sha256",
        )

    def attest(
        self,
        request: AuthorityAttestationRequest,
    ) -> AuthorityAttestationReceipt:
        unsigned = AuthorityAttestationReceipt(
            format=AUTHORITY_ATTESTATION_FORMAT,
            issuer_id=self._issuer_id,
            assurance=REFERENCE_ASSURANCE,
            challenge=request.challenge,
            endpoint_locator_sha256=request.endpoint_locator_sha256,
            host_task_commitment=request.host_task_commitment,
            agent_task_view_sha256=request.agent_task_view_sha256,
            evaluation_context_sha256=request.evaluation_context_sha256,
            claims=self._claims,
            evidence_sha256=self._evidence_sha256,
            mac_sha256="0" * 64,
        )
        mac = _mac_payload(self._key, unsigned.signed_payload())
        return AuthorityAttestationReceipt(
            **{
                **unsigned.__dict__,
                "mac_sha256": mac,
            }
        )


class ReferenceHMACAuthorityVerifier:
    """Reference trusted-side verifier with one-time challenge consumption."""

    def __init__(
        self,
        key: bytes,
        *,
        expected_issuer_id: str = "reference-authority-attestor",
    ) -> None:
        self._key = _require_key(key)
        if not isinstance(expected_issuer_id, str) or not expected_issuer_id:
            raise ValueError("expected_issuer_id must be non-empty")
        self._expected_issuer_id = expected_issuer_id
        self._consumed_challenges: set[str] = set()

    def verify(
        self,
        request: AuthorityAttestationRequest,
        receipt: AuthorityAttestationReceipt,
    ) -> AuthorityVerification:
        try:
            receipt.validate_shape()
        except AuthorityAttestationError as exc:
            return _invalid(str(exc))

        if request.challenge in self._consumed_challenges:
            return _invalid("authority attestation challenge replayed")
        if receipt.issuer_id != self._expected_issuer_id:
            return _invalid("authority attestation issuer mismatch")
        if receipt.assurance != REFERENCE_ASSURANCE:
            return _invalid("authority attestation assurance mismatch")

        expected = request.to_dict()
        for field in (
            "challenge",
            "endpoint_locator_sha256",
            "host_task_commitment",
            "agent_task_view_sha256",
            "evaluation_context_sha256",
        ):
            if getattr(receipt, field) != expected[field]:
                return _invalid(
                    f"authority attestation {field} mismatch"
                )

        expected_mac = _mac_payload(
            self._key,
            receipt.signed_payload(),
        )
        if not hmac.compare_digest(
            receipt.mac_sha256,
            expected_mac,
        ):
            return _invalid("authority attestation MAC mismatch")

        self._consumed_challenges.add(request.challenge)
        return AuthorityVerification(
            valid=True,
            reason="authority attestation valid",
            issuer_id=receipt.issuer_id,
            assurance=receipt.assurance,
            claims=receipt.claims,
            evidence_sha256=receipt.evidence_sha256,
        )


def authority_receipt_sha256(
    receipt: AuthorityAttestationReceipt,
) -> str:
    return sha256_json(receipt.to_dict())


def _mac_payload(key: bytes, payload: Mapping[str, Any]) -> str:
    message = sha256_json(payload).encode("ascii")
    return hmac.new(
        key,
        message,
        hashlib.sha256,
    ).hexdigest()


def _invalid(reason: str) -> AuthorityVerification:
    return AuthorityVerification(
        valid=False,
        reason=reason,
        issuer_id=None,
        assurance=None,
        claims=AuthorityClaims.none_verified(),
        evidence_sha256=None,
    )


def _require_key(key: bytes) -> bytes:
    if not isinstance(key, bytes) or len(key) < 32:
        raise ValueError(
            "reference authority HMAC key must be at least 32 bytes"
        )
    return bytes(key)


def _require_sha256(value: str, field: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(ch not in "0123456789abcdef" for ch in value)
    ):
        raise AuthorityAttestationError(
            f"{field} must be 64 lowercase hex characters"
        )


def _required_string(
    value: Mapping[str, Any],
    field: str,
) -> str:
    item = value.get(field)
    if not isinstance(item, str) or not item:
        raise AuthorityAttestationError(
            f"{field} must be a non-empty string"
        )
    return item
