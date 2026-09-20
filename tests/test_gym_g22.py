from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

from gym.core import ReferenceGym
from gym.host import create_environment
from gym.private_pack import PrivateHoldoutPack, PrivatePackError
from gym.reference_policy import ReferencePolicy
from gym.subprocess_host import SubprocessHostGateway


ROOT = Path(__file__).resolve().parents[1]
GYM_ROOT = ROOT / "gym"
REFERENCE_PACK = GYM_ROOT / "private_holdout_reference"


def _canonical_json_bytes(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _copy_pack(tmp: str) -> Path:
    target = Path(tmp) / "pack"
    shutil.copytree(REFERENCE_PACK, target)
    return target


def _manifest(pack: Path) -> dict:
    return json.loads((pack / "PRIVATE_HOLDOUT_PACK.json").read_text())


def _write_manifest(pack: Path, manifest: dict) -> None:
    (pack / "PRIVATE_HOLDOUT_PACK.json").write_bytes(_canonical_json_bytes(manifest))


def _server_env():
    current = os.environ.get("PYTHONPATH", "")
    parts = [str(ROOT), str(ROOT / "src")]
    if current:
        parts.append(current)
    return {"PYTHONPATH": os.pathsep.join(parts)}


def _server_command(pack: Path):
    return [
        sys.executable,
        "-B",
        "-m",
        "gym.private_host_server",
        "--pack",
        str(pack),
    ]


class GymG22PrivatePackTests(unittest.TestCase):
    def test_reference_pack_loads_as_host_only_holdout(self):
        pack = PrivateHoldoutPack.load(
            REFERENCE_PACK,
            expected_runtime_id="reference-v1",
        )
        self.assertEqual(pack.pack_id, "reference-holdout-v1")
        self.assertFalse(pack.native_competence_claim)
        self.assertEqual(pack.task_ids, ("hg_f8",))

        task = pack.task("hg_f8")
        self.assertEqual(task.split, "holdout")
        self.assertTrue(task.metadata["private_holdout_verified"])
        self.assertEqual(task.metadata["private_pack_id"], pack.pack_id)
        self.assertIn("private_snapshot", task.environment)
        self.assertNotIn("metadata", task.agent_view().__dict__)
        self.assertFalse(hasattr(task.agent_view(), "environment"))
        self.assertFalse(hasattr(task.agent_view(), "split"))
        self.assertEqual(
            pack.task_commitment("hg_f8"),
            pack.tasks["hg_f8"].descriptor_sha256,
        )

    def test_pack_cannot_self_assert_native_competence(self):
        with tempfile.TemporaryDirectory() as tmp:
            pack_root = _copy_pack(tmp)
            manifest = _manifest(pack_root)
            manifest["native_competence_claim"] = True
            _write_manifest(pack_root, manifest)
            with self.assertRaises(PrivatePackError) as caught:
                PrivateHoldoutPack.load(pack_root, expected_runtime_id="reference-v1")
            self.assertIn("cannot self-assert native competence", str(caught.exception))

    def test_task_cannot_self_assert_native_competence(self):
        with tempfile.TemporaryDirectory() as tmp:
            pack_root = _copy_pack(tmp)
            task_path = pack_root / "tasks" / "hg_f8.json"
            task = json.loads(task_path.read_text())
            task["metadata"]["native_competence_claim"] = True
            task_bytes = _canonical_json_bytes(task)
            task_path.write_bytes(task_bytes)

            manifest = _manifest(pack_root)
            manifest["tasks"][0]["sha256"] = _sha256(task_bytes)
            _write_manifest(pack_root, manifest)

            with self.assertRaises(PrivatePackError) as caught:
                PrivateHoldoutPack.load(pack_root, expected_runtime_id="reference-v1")
            self.assertIn("cannot self-assert native competence", str(caught.exception))

    def test_task_competence_claim_must_be_literal_false(self):
        with tempfile.TemporaryDirectory() as tmp:
            pack_root = _copy_pack(tmp)
            task_path = pack_root / "tasks" / "hg_f8.json"
            task = json.loads(task_path.read_text())
            task["metadata"]["native_competence_claim"] = 0
            task_bytes = _canonical_json_bytes(task)
            task_path.write_bytes(task_bytes)

            manifest = _manifest(pack_root)
            manifest["tasks"][0]["sha256"] = _sha256(task_bytes)
            _write_manifest(pack_root, manifest)

            with self.assertRaises(PrivatePackError) as caught:
                PrivateHoldoutPack.load(pack_root, expected_runtime_id="reference-v1")
            self.assertIn("cannot self-assert native competence", str(caught.exception))

    def test_task_tamper_is_rejected_before_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            pack_root = _copy_pack(tmp)
            task_path = pack_root / "tasks" / "hg_f8.json"
            task_path.write_bytes(task_path.read_bytes() + b"\n")
            with self.assertRaises(PrivatePackError) as caught:
                PrivateHoldoutPack.load(pack_root, expected_runtime_id="reference-v1")
            self.assertIn("task hg_f8 sha256 mismatch", str(caught.exception))

    def test_snapshot_tamper_is_rejected_before_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            pack_root = _copy_pack(tmp)
            snapshot = pack_root / "snapshots" / "env-d13f6e.json"
            snapshot.write_bytes(snapshot.read_bytes() + b" ")
            with self.assertRaises(PrivatePackError) as caught:
                PrivateHoldoutPack.load(pack_root, expected_runtime_id="reference-v1")
            self.assertIn("snapshot env-d13f6e sha256 mismatch", str(caught.exception))

    def test_runtime_descriptor_tamper_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            pack_root = _copy_pack(tmp)
            runtime = pack_root / "runtime" / "reference-runtime.json"
            runtime.write_bytes(runtime.read_bytes() + b" ")
            with self.assertRaises(PrivatePackError) as caught:
                PrivateHoldoutPack.load(pack_root, expected_runtime_id="reference-v1")
            self.assertIn("runtime descriptor sha256 mismatch", str(caught.exception))

    def test_unlisted_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            pack_root = _copy_pack(tmp)
            (pack_root / "ambient-secret.txt").write_text("must not become ambient state")
            with self.assertRaises(PrivatePackError) as caught:
                PrivateHoldoutPack.load(pack_root, expected_runtime_id="reference-v1")
            self.assertIn("file coverage mismatch", str(caught.exception))
            self.assertIn("ambient-secret.txt", str(caught.exception))

    def test_path_traversal_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            pack_root = _copy_pack(tmp)
            manifest = _manifest(pack_root)
            manifest["runtime"]["path"] = "../runtime.json"
            _write_manifest(pack_root, manifest)
            with self.assertRaises(PrivatePackError) as caught:
                PrivateHoldoutPack.load(pack_root, expected_runtime_id="reference-v1")
            self.assertIn("unsafe private pack path", str(caught.exception))

    def test_duplicate_json_key_is_rejected_even_with_valid_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            pack_root = _copy_pack(tmp)
            task_path = pack_root / "tasks" / "hg_f8.json"
            task_text = task_path.read_text()
            task_text = task_text.replace(
                '"split":"holdout"',
                '"split":"holdout","split":"holdout"',
            )
            task_bytes = task_text.encode("utf-8")
            task_path.write_bytes(task_bytes)

            manifest = _manifest(pack_root)
            manifest["tasks"][0]["sha256"] = _sha256(task_bytes)
            _write_manifest(pack_root, manifest)

            with self.assertRaises(PrivatePackError) as caught:
                PrivateHoldoutPack.load(pack_root, expected_runtime_id="reference-v1")
            self.assertIn("duplicate JSON key", str(caught.exception))

    def test_overflow_nonfinite_json_number_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            pack_root = _copy_pack(tmp)
            snapshot_path = pack_root / "snapshots" / "env-d13f6e.json"
            snapshot_text = snapshot_path.read_text().replace(
                '"timeout":30',
                '"timeout":1e999',
            )
            snapshot_bytes = snapshot_text.encode("utf-8")
            snapshot_path.write_bytes(snapshot_bytes)

            manifest = _manifest(pack_root)
            manifest["tasks"][0]["snapshot"]["sha256"] = _sha256(snapshot_bytes)
            _write_manifest(pack_root, manifest)

            with self.assertRaises(PrivatePackError) as caught:
                PrivateHoldoutPack.load(pack_root, expected_runtime_id="reference-v1")
            self.assertIn("non-finite number", str(caught.exception))

    def test_non_private_split_is_rejected_even_with_valid_file_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            pack_root = _copy_pack(tmp)
            task_path = pack_root / "tasks" / "hg_f8.json"
            task = json.loads(task_path.read_text())
            task["split"] = "train"
            task_bytes = _canonical_json_bytes(task)
            task_path.write_bytes(task_bytes)

            manifest = _manifest(pack_root)
            manifest["tasks"][0]["sha256"] = _sha256(task_bytes)
            _write_manifest(pack_root, manifest)

            with self.assertRaises(PrivatePackError) as caught:
                PrivateHoldoutPack.load(pack_root, expected_runtime_id="reference-v1")
            self.assertIn("non-private split", str(caught.exception))

    def test_verified_snapshot_drives_hidden_oracle_and_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            pack_root = _copy_pack(tmp)
            snapshot_path = pack_root / "snapshots" / "env-d13f6e.json"
            snapshot = json.loads(snapshot_path.read_text())
            snapshot["state"]["project_config"] = {"timeout": 91}
            snapshot["state"]["cache_config"] = {"timeout": 7}
            snapshot["oracle"]["admissible_action"] = "SELECT_CACHE_CONFIG"
            snapshot_bytes = _canonical_json_bytes(snapshot)
            snapshot_path.write_bytes(snapshot_bytes)

            manifest = _manifest(pack_root)
            manifest["tasks"][0]["snapshot"]["sha256"] = _sha256(snapshot_bytes)
            _write_manifest(pack_root, manifest)

            pack = PrivateHoldoutPack.load(pack_root, expected_runtime_id="reference-v1")
            task = pack.task("hg_f8")
            env = create_environment(task)
            try:
                reset = env.reset()
                self.assertEqual(reset[0].value, {"timeout": 91})
                self.assertEqual(reset[1].value, {"timeout": 7})
                env.step("INSPECT_CONFIG_ORIGIN")
                outcome = env.step("SELECT_CACHE_CONFIG")
                self.assertTrue(outcome.done)
                self.assertTrue(env.semantic_verdict())
            finally:
                env.close()

    def test_subprocess_host_runs_verified_pack_end_to_end(self):
        pack = PrivateHoldoutPack.load(
            REFERENCE_PACK,
            expected_runtime_id="reference-v1",
        )
        task = pack.task("hg_f8")
        with SubprocessHostGateway(
            _server_command(REFERENCE_PACK),
            cwd=ROOT,
            env=_server_env(),
        ) as host:
            result = ReferenceGym(host_gateway=host).run(task, ReferencePolicy())
            self.assertTrue(result.accepted, result.admission.reason)
            self.assertFalse(result.learning_updated)
            self.assertEqual(
                task.metadata["private_snapshot_sha256"],
                pack.tasks["hg_f8"].snapshot_sha256,
            )

    def test_explicit_environment_close_removes_temp_root(self):
        pack = PrivateHoldoutPack.load(
            REFERENCE_PACK,
            expected_runtime_id="reference-v1",
        )
        env = create_environment(pack.task("hg_f8"))
        root = env.root
        self.assertTrue(root.exists())
        env.close()
        self.assertFalse(root.exists())


if __name__ == "__main__":
    unittest.main()
