"""Numerical contract tests; no NPU, dx_engine, or PyTorch needed."""
import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

SCRIPT = Path(__file__).resolve().parents[1] / "probe_parseq_dxm1.py"
spec = importlib.util.spec_from_file_location("probe_dxm1", SCRIPT)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class RgbContractTests(unittest.TestCase):
    def encoder(self):
        dims = [1, 3, 32, 128]
        graph = helper.make_graph([
            helper.make_node("Mul", ["images", "weights"], ["weighted"]),
            helper.make_node("Tanh", ["weighted"], ["memory"]),
        ], "contract_fixture", [helper.make_tensor_value_info("images", TensorProto.FLOAT, dims)],
            [helper.make_tensor_value_info("memory", TensorProto.FLOAT, dims)],
            [numpy_helper.from_array(np.array([0.5, 1.5, -0.75], np.float32).reshape(1, 3, 1, 1), "weights")])
        return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8)

    def test_wrapper_matches_original_for_full_pixel_range_and_limits(self):
        original = self.encoder()
        old_serialized = original.SerializeToString()
        with tempfile.TemporaryDirectory() as root:
            source, wrapped = Path(root) / "source.onnx", Path(root) / "wrapped.onnx"
            onnx.save(original, source)
            onnx.save(probe.wrap_graph(original), wrapped)
            src, raw_model = probe.ort_session(source), probe.ort_session(wrapped)
            gradient = (np.arange(32 * 128 * 3, dtype=np.int32) % 256).astype(np.uint8).reshape(32, 128, 3)
            for pixels in [np.zeros_like(gradient), np.full_like(gradient, 255), gradient]:
                raw = probe.raw_nchw(pixels)
                a = src.run(None, {"images": probe.normalize(raw)})[0]
                b = raw_model.run(None, {raw_model.get_inputs()[0].name: raw})[0]
                np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-7)
        self.assertEqual(original.SerializeToString(), old_serialized)

    def test_refuses_double_wrapping_and_wrong_input_shape(self):
        with self.assertRaises(ValueError):
            probe.wrap_graph(probe.wrap_graph(self.encoder()))
        model = self.encoder()
        model.graph.input[0].type.tensor_type.shape.dim[2].dim_value = 64
        with self.assertRaises(ValueError):
            probe.wrap_graph(model)

    def test_detects_quantized_feature_drift_and_invalid_values(self):
        a = np.array([[[1, -1, 2, -2]]], dtype=np.float32)
        exact = probe.feature_metrics(a, a.copy())
        self.assertAlmostEqual(exact["feature_cosine"], 1)
        self.assertEqual(exact["feature_relative_l2"], 0)
        altered = probe.feature_metrics(a, -a)
        self.assertAlmostEqual(altered["feature_cosine"], -1)
        self.assertAlmostEqual(altered["feature_relative_l2"], 2)
        with self.assertRaises(ValueError):
            probe.feature_metrics(a, np.full_like(a, np.nan))
        with self.assertRaises(ValueError):
            probe.feature_metrics(a, a.reshape(-1))


if __name__ == "__main__":
    unittest.main()
