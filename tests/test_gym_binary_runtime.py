from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from gym.binary_runtime_entry import _allowed_temp_script, _execute, _resolve


class BinaryRuntimeEntryTests(unittest.TestCase):
    def test_module_role_resolves(self):
        self.assertEqual(
            _resolve(["adaptive-train"]),
            ("module", "gym.run_adaptive_training_reference", []),
        )

    def test_allowed_temp_script_resolves(self):
        with tempfile.TemporaryDirectory(prefix="gym-runtime-test-") as tmp:
            script = Path(tmp) / "probe.py"
            script.write_text("pass\n")
            self.assertEqual(
                _resolve(["-B", str(script)]),
                ("script", str(script.resolve()), []),
            )

    def test_outside_script_is_rejected(self):
        root = Path(__file__).resolve().parent
        script = root / "_binary_runtime_outside_probe.py"
        script.write_text("pass\n")
        try:
            self.assertIsNone(_allowed_temp_script(str(script)))
            with self.assertRaises(SystemExit):
                _resolve(["-B", str(script)])
        finally:
            script.unlink(missing_ok=True)

    def test_temp_script_symlink_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="gym-runtime-test-") as tmp:
            root = Path(tmp)
            target = root / "target.py"
            link = root / "link.py"
            target.write_text("pass\n")
            try:
                link.symlink_to(target)
            except (OSError, NotImplementedError):
                self.skipTest("symlink unavailable")
            self.assertIsNone(_allowed_temp_script(str(link)))

    def test_script_mode_matches_sibling_import_semantics(self):
        with tempfile.TemporaryDirectory(prefix="gym-runtime-test-") as tmp:
            root = Path(tmp)
            (root / "helper.py").write_text("VALUE = 42\n")
            marker = root / "marker.txt"
            script = root / "probe.py"
            script.write_text(
                "from pathlib import Path\n"
                "from helper import VALUE\n"
                f"Path({str(marker)!r}).write_text(str(VALUE))\n"
            )
            mode, target, rest = _resolve(["-B", str(script)])
            self.assertEqual(_execute(mode, target, rest), 0)
            self.assertEqual(marker.read_text(), "42")


if __name__ == "__main__":
    unittest.main()
