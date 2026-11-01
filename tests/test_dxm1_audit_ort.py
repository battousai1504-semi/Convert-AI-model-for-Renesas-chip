"""Host-side regression: a CPU+NPU graph must run CPU tasks in both paths."""
import contextlib
import importlib
import io
import json
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import audit_dxm1_io as audit


class HybridAuditTests(unittest.TestCase):
    def test_affine_diagnostic_detects_offset_without_changing_arrays(self):
        rng = np.random.default_rng(1)
        source = rng.normal(size=(1, 128, 384)).astype(np.float32)
        actual = (2 * source + rng.normal(0, 10, size=(1, 1, 384))).astype(np.float32)
        a, b = source.copy(), actual.copy()
        result = audit.centered_diagnostic(source, actual)
        self.assertAlmostEqual(result["centered_cosine"], 1, places=6)
        self.assertAlmostEqual(result["centered_gain"], 2, places=6)
        self.assertLess(result["affine_fit_residual_relative_l2"], 1e-6)
        self.assertTrue(np.array_equal(source, a) and np.array_equal(actual, b))

    def test_constant_arrays_do_not_invent_centered_similarity(self):
        result = audit.centered_diagnostic(np.ones((1, 128, 384), np.float32),
                                            np.full((1, 128, 384), 3, np.float32))
        self.assertIsNone(result["centered_cosine"])

    def run_fixture(self, root, native_support=True, python_support=True):
        root = Path(root)
        run = root / "run"
        run.mkdir()
        source = run / "encoder_normalized_reference.onnx"
        source.write_bytes(b"reference fixture; ORT is stubbed")
        (run / "contract.json").write_text(json.dumps({"source_sha256": audit.sha256(source)}))
        model = root / "hybrid.dxnn"
        model.write_bytes(b"hybrid fixture; NPU is stubbed")
        image = root / "image.png"
        pixels = np.zeros((32, 128, 3), np.uint8)
        pixels[:, :, 0] = 17
        pixels[:, :, 1] = 101
        pixels[:, :, 2] = 230
        Image.fromarray(pixels).save(image)
        # Model semantics: NPU gives 1; a required CPU task adds 1.
        expected = np.full((1, 128, 384), 2, np.float32)
        seen = []

        class Option:
            def __init__(self):
                self.use_ort = False

        class Engine:
            def __init__(self, path, options):
                self.options = options
                if not python_support:
                    options.use_ort = False

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def get_input_tensors_info(self):
                return [{"shape": [1, 32, 128, 3], "dtype": np.uint8}]

            def get_output_tensors_info(self):
                return [{"shape": [1, 128, 384], "dtype": np.float32}]

            def get_input_size(self):
                return 12288

            def get_output_size(self):
                return 196608

            def run(self, feed):
                if not np.array_equal(feed[0][0], pixels):
                    raise AssertionError("Python input differs from native fixture input")
                return [expected.copy() if self.options.use_ort else expected - 1]

        class Reference:
            def get_inputs(self):
                return [types.SimpleNamespace(name="images")]

            def run(self, outputs, inputs):
                return [expected.copy()]

        def native(command, **kwargs):
            if command[1:] == ["--help"]:
                return subprocess.CompletedProcess(command, 0, "--use-ort" if native_support else "--single", "")
            seen.append(command)
            raw = Path(command[command.index("--input") + 1])
            self.assertEqual(raw.read_bytes(), pixels.tobytes())
            output = Path(command[command.index("--output") + 1])
            (expected if "--use-ort" in command else expected - 1).tofile(output)
            return subprocess.CompletedProcess(command, 0, "", "")

        fake_package = types.ModuleType("dx_engine")
        fake_package.InferenceEngine = Engine
        fake_option = types.ModuleType("dx_engine.inference_option")
        fake_option.InferenceOption = Option
        argv = ["audit", "--model", str(model), "--reference-run", str(run),
                "--input", str(image), "--output", str(root / "output")]
        with patch.dict(sys.modules, {"dx_engine": fake_package, "dx_engine.inference_option": fake_option}), \
                patch.object(sys, "argv", argv), patch.object(audit.shutil, "which", return_value="/fake/dxrun"), \
                patch.object(audit, "ort_session", return_value=Reference()), \
                patch.object(audit.subprocess, "run", side_effect=native), \
                patch("importlib.metadata.version", return_value="3.4.2"), contextlib.redirect_stdout(io.StringIO()):
            result = audit.main()
        return result, json.loads((root / "output/io_audit.json").read_text()), seen

    def test_required_cpu_task_executed_in_both_paths(self):
        with tempfile.TemporaryDirectory() as root:
            result, report, commands = self.run_fixture(root)
        self.assertEqual(result, 0)
        self.assertTrue(report["rows"][0]["python_native_close"])
        self.assertAlmostEqual(report["rows"][0]["onnx_python_cosine"], 1)
        self.assertAlmostEqual(report["rows"][0]["onnx_native_cosine"], 1)
        self.assertEqual(report["rows"][0]["python_native_max_abs"], 0)
        self.assertEqual(len(commands), 1)

    def test_native_without_ort_is_rejected(self):
        with tempfile.TemporaryDirectory() as root, self.assertRaisesRegex(RuntimeError, "--use-ort"):
            self.run_fixture(root, native_support=False)

    def test_python_disabling_ort_is_rejected(self):
        with tempfile.TemporaryDirectory() as root, self.assertRaisesRegex(RuntimeError, "disabled ORT"):
            self.run_fixture(root, python_support=False)


if __name__ == "__main__":
    unittest.main()
