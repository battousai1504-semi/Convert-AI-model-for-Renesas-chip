"""CPU numerical and state-restoration checks; no actual NPU dependency."""
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper, TensorProto

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from probe_dxm1_matmul_head import replace_head, pack_pixels
from probe_parseq_dxm1 import normalize, raw_nchw, sha256
from compare_dxm1_matmul_ocr import load_run, check_runtime, decode_with_memory


def encoder_fixture(attention=True):
    rng = np.random.default_rng(32)
    def tensor(name, a):
        return numpy_helper.from_array(np.asarray(a), name)
    nodes = [
        helper.make_node("Conv", ["images", "patch_embed.proj.weight", "patch_embed.proj.bias"],
                         ["conv"], kernel_shape=[4, 8], strides=[4, 8], pads=[0, 0, 0, 0]),
        helper.make_node("Reshape", ["conv", "flat_shape"], ["flat"]),
        helper.make_node("Transpose", ["flat"], ["patch_tokens"], perm=[0, 2, 1]),
        helper.make_node("Add", ["patch_tokens", "pos_embed"], ["positioned"]),
    ]
    initializers = [
        tensor("patch_embed.proj.weight", rng.normal(0, .05, (384, 3, 4, 8)).astype(np.float32)),
        tensor("patch_embed.proj.bias", rng.normal(0, .03, 384).astype(np.float32)),
        tensor("pos_embed", rng.normal(0, .02, (1, 128, 384)).astype(np.float32)),
        tensor("flat_shape", np.array([1, 384, 128], np.int64)),
    ]
    output = "positioned"
    if attention:
        for name in ["q", "k", "v"]:
            initializers.append(tensor("w" + name, rng.normal(0, .01, (384, 384)).astype(np.float32)))
            nodes.append(helper.make_node("MatMul", ["positioned", "w" + name], [name]))
        nodes.extend([
            helper.make_node("Transpose", ["k"], ["kt"], perm=[0, 2, 1]),
            helper.make_node("MatMul", ["q", "kt"], ["scores"]),
            helper.make_node("Softmax", ["scores"], ["probabilities"], axis=-1),
            helper.make_node("MatMul", ["probabilities", "v"], ["attended"]),
            helper.make_node("Add", ["positioned", "attended"], ["residual"]),
            helper.make_node("LayerNormalization", ["residual", "gamma", "beta"], ["memory"],
                             axis=-1, epsilon=1e-6),
        ])
        initializers.extend([tensor("gamma", np.ones(384, np.float32)),
                             tensor("beta", np.zeros(384, np.float32))])
        output = "memory"
    graph = helper.make_graph(nodes, "full_tail_fixture",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 32, 128])],
        [helper.make_tensor_value_info(output, TensorProto.FLOAT, [1, 128, 384])], initializers)
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8)


class FullEncoderHelperTests(unittest.TestCase):
    def test_replacement_preserves_attention_and_layernorm_numerically(self):
        original = encoder_fixture()
        serialized = original.SerializeToString()
        rewritten = replace_head(original)
        self.assertIn("Softmax", [node.op_type for node in rewritten.graph.node])
        self.assertIn("LayerNormalization", [node.op_type for node in rewritten.graph.node])
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        sessions = [ort.InferenceSession(m.SerializeToString(), options,
                    providers=["CPUExecutionProvider"]) for m in [original, rewritten]]
        images = [np.zeros((32,128,3),np.uint8), np.full((32,128,3),255,np.uint8),
                  np.random.default_rng(7).integers(0,256,(32,128,3),dtype=np.uint8)]
        for pixels in images:
            a = sessions[0].run(None, {sessions[0].get_inputs()[0].name: normalize(raw_nchw(pixels))})[0]
            b = sessions[1].run(None, {sessions[1].get_inputs()[0].name: pack_pixels(pixels)})[0]
            np.testing.assert_allclose(a, b, rtol=1e-3, atol=1e-4)
        self.assertEqual(original.SerializeToString(), serialized)

    def test_ocr_refuses_embedding_prefix_and_changed_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "original_reference.onnx"
            onnx.save(encoder_fixture(attention=False), source)
            matmul = root / "parseq_rawpatch_matmul.onnx"
            matmul.write_bytes(b"fixture")
            dxnn = root / "fixture.dxnn"
            dxnn.write_bytes(b"fixture")
            manifest = {"status":"compiled", "dxnn":dxnn.name, "dxnn_sha256":sha256(dxnn),
                        "original_sha256":sha256(source), "matmul_sha256":sha256(matmul),
                        "placement":{"projection_nodes":["projection"], "partitions":{"npu_0":"NPU"}}}
            (root / "matmul_contract.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "FULL encoder"):
                load_run(root)
            dxnn.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "hash"):
                load_run(root)

    def test_decoder_encoder_override_restored_on_success_and_failure(self):
        class Tensor:
            def __init__(self, array): self.array = array
            def to(self, **kwargs): return self
        fake_torch = types.ModuleType("torch")
        fake_torch.from_numpy = Tensor
        image = types.SimpleNamespace(shape=(1,3,32,128), device="cpu", dtype=np.float32)
        original = lambda x: "original"
        core = types.SimpleNamespace(encode=original, decode_ar=True, refine_iters=1, max_label_length=25)
        class Model:
            def __init__(self, fail): self.model, self.fail = core, fail
            def __call__(self, x):
                values = self.model.encode(x).array
                if self.fail: raise RuntimeError("decoder fixture failure")
                return float(values.mean())
        with patch.dict(sys.modules, {"torch": fake_torch}):
            for fail in [False, True]:
                if fail:
                    with self.assertRaisesRegex(RuntimeError, "fixture failure"):
                        decode_with_memory(Model(True), image, np.full((1,128,384),7,np.float32))
                else:
                    self.assertEqual(decode_with_memory(Model(False), image, np.full((1,128,384),7,np.float32)), 7)
                self.assertIs(core.encode, original)
                self.assertEqual((core.decode_ar, core.refine_iters, core.max_label_length),(True,1,25))

    def test_runtime_rejects_uint8_patch_interface(self):
        class Engine:
            def get_input_tensors_info(self): return [{"shape":[1,128,96], "dtype":np.uint8}]
            def get_output_tensors_info(self): return [{"shape":[1,128,384], "dtype":np.float32}]
            def get_input_size(self): return 128*96
            def get_output_size(self): return 128*384*4
        with self.assertRaisesRegex(ValueError, "input contract"):
            check_runtime(Engine())


if __name__ == "__main__":
    unittest.main()
