"""Regression checks for symbolic batch metadata at the embedding cut point."""
import copy
import sys
import unittest
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from probe_dxm1_embedding import extract_embedding, verify_static_output
from probe_parseq_dxm1 import ort_session


def fixture():
    graph = helper.make_graph([
        helper.make_node("Reshape", ["images", "shape"], ["flattened"]),
        helper.make_node("Tile", ["flattened", "repeats"], ["tokens"]),
        helper.make_node("Add", ["tokens", "pos_embed"], ["positioned"], name="positional_add"),
        helper.make_node("Tanh", ["positioned"], ["memory"], name="blocks.0/nonlinear"),
    ], "embedding_fixture",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 32, 128])],
        [helper.make_tensor_value_info("memory", TensorProto.FLOAT, [1, 128, 384])],
        [numpy_helper.from_array(np.array([1, 128, 96], np.int64), "shape"),
         numpy_helper.from_array(np.array([1, 1, 4], np.int64), "repeats"),
         numpy_helper.from_array(np.ones([1, 128, 384], np.float32), "pos_embed")])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8)


class EmbeddingShapeTests(unittest.TestCase):
    def prefix(self):
        return extract_embedding(fixture())[0]

    def unknown_batch(self, prefix):
        for value in [prefix.graph.output[0], *prefix.graph.value_info]:
            if value.name == prefix.graph.output[0].name:
                dimension = value.type.tensor_type.shape.dim[0]
                dimension.Clear()
                dimension.dim_param = "batch_from_reshape"

    def test_unknown_batch_uses_runtime_before_specializing(self):
        prefix = self.prefix()
        self.unknown_batch(prefix)
        self.assertEqual(prefix.graph.output[0].type.tensor_type.shape.dim[0].dim_value, 0)
        self.assertFalse(prefix.graph.output[0].type.tensor_type.shape.dim[0].HasField("dim_value"))
        nodes = [n.SerializeToString() for n in prefix.graph.node]
        weights = [n.SerializeToString() for n in prefix.graph.initializer]
        info = verify_static_output(prefix)
        self.assertEqual(info["output_shape_metadata_before"], ["batch_from_reshape", 128, 384])
        self.assertEqual([c["shape"] for c in info["runtime_shape_checks"]], [[1, 128, 384]] * 2)
        self.assertEqual([d.dim_value for d in prefix.graph.output[0].type.tensor_type.shape.dim], [1, 128, 384])
        self.assertEqual(nodes, [n.SerializeToString() for n in prefix.graph.node])
        self.assertEqual(weights, [n.SerializeToString() for n in prefix.graph.initializer])

    def test_literal_zero_is_rejected(self):
        prefix = self.prefix()
        prefix.graph.output[0].type.tensor_type.shape.dim[0].dim_value = 0
        with self.assertRaisesRegex(ValueError, "Concrete"):
            verify_static_output(prefix)

    def test_real_batch_two_is_not_masked(self):
        prefix = self.prefix()
        self.unknown_batch(prefix)
        for value in prefix.graph.initializer:
            if value.name == "repeats":
                value.CopyFrom(numpy_helper.from_array(np.array([2, 1, 4], np.int64), "repeats"))
        # Remove stale intermediate metadata so the real shape comes from ops.
        del prefix.graph.value_info[:]
        with self.assertRaisesRegex(ValueError, "Real prefix output"):
            verify_static_output(prefix)

    def test_nonfinite_output_is_rejected(self):
        prefix = self.prefix()
        self.unknown_batch(prefix)
        for value in prefix.graph.initializer:
            if value.name == "pos_embed":
                value.CopyFrom(numpy_helper.from_array(np.full([1, 128, 384], np.nan, np.float32), "pos_embed"))
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            verify_static_output(prefix)

    def test_extracted_prefix_matches_real_intermediate(self):
        import onnxruntime as ort
        model = fixture()
        unchanged = model.SerializeToString()
        prefix, _ = extract_embedding(model)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        full = ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])
        cut = ort.InferenceSession(prefix.SerializeToString(), options, providers=["CPUExecutionProvider"])
        x = np.random.default_rng(13).uniform(-1, 1, (1, 3, 32, 128)).astype(np.float32)
        a, b = full.run(None, {"images": x})[0], cut.run(None, {"images": x})[0]
        np.testing.assert_allclose(a, np.tanh(b), rtol=1e-6, atol=1e-7)
        self.assertEqual(model.SerializeToString(), unchanged)


if __name__ == "__main__":
    unittest.main()
