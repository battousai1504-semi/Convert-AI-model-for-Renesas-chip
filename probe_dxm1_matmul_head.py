#!/usr/bin/env python3
"""Exact raw-patch MatMul replacement for PARSeq patch Conv; diagnostic probe.

CPU packs RGB pixels into patches; NPU runs the learned projection and tail.
Normalization is folded algebraically into the projection weights and bias.
"""
import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np

from probe_parseq_dxm1 import (
    discover, rgb_pixels, raw_nchw, normalize, ort_session,
    sha256, save_json, feature_metrics,
)

PREFIX = "__onekiwi_mm_v1__"


def pack_pixels(pixels):
    if pixels.shape != (32, 128, 3) or pixels.dtype != np.uint8:
        raise ValueError("Expected RGB uint8 pixels [32,128,3]")
    # Conv weight order is [out_channel, in_channel, kernel_h, kernel_w].
    # Token order is row-major patch_y, patch_x. Preserve C,dy,dx within a patch.
    return np.ascontiguousarray(
        pixels.transpose(2, 0, 1).reshape(3, 8, 4, 16, 8)
        .transpose(1, 3, 0, 2, 4).reshape(1, 128, 96), dtype=np.float32)


def replace_head(original):
    import onnx
    from onnx import helper, numpy_helper, TensorProto
    from onnx.utils import Extractor
    if len(original.graph.input) != 1 or len(original.graph.output) != 1:
        raise ValueError("Expected single-input/single-output PARSeq encoder or embedding prefix")
    shape = [d.dim_value for d in original.graph.input[0].type.tensor_type.shape.dim]
    if shape != [1, 3, 32, 128]:
        raise ValueError(f"Use the ORIGINAL normalized-input encoder, got input {shape}")
    weights = {v.name: numpy_helper.to_array(v) for v in original.graph.initializer}
    wname, bname = "patch_embed.proj.weight", "patch_embed.proj.bias"
    w, b = weights[wname], weights[bname]
    if w.shape != (384, 3, 4, 8) or b.shape != (384,):
        raise ValueError("Unexpected PARSeq projection weights")
    convs = [n for n in original.graph.node if n.op_type == "Conv" and wname in n.input]
    if len(convs) != 1:
        raise ValueError("Cannot identify a unique patch Conv")
    attrs = {a.name: helper.get_attribute_value(a) for a in convs[0].attribute}
    if (attrs.get("kernel_shape") != [4, 8] or attrs.get("strides") != [4, 8]
            or attrs.get("pads", [0]*4) != [0]*4
            or attrs.get("dilations", [1, 1]) != [1, 1] or attrs.get("group", 1) != 1):
        raise ValueError(f"Unsupported patch Conv geometry: {attrs}")
    position_names = {n for n, v in weights.items()
                      if "pos_embed" in n and v.shape == (1, 128, 384)}
    adds = [n for n in original.graph.node if n.op_type == "Add"
            and any(name in position_names for name in n.input)]
    if len(adds) != 1:
        raise ValueError("Cannot identify unique positional Add")
    token_inputs = [name for name in adds[0].input if name not in position_names]
    if len(token_inputs) != 1:
        raise ValueError("Cannot identify the tensor before positional Add")
    token_name = token_inputs[0]
    tail = Extractor(original).extract_model([token_name], [original.graph.output[0].name])
    used = {name for n in tail.graph.node for name in [n.name, *n.input, *n.output]}
    used.update(v.name for v in tail.graph.initializer)
    if any(name.startswith(PREFIX) for name in used):
        raise ValueError("Reserved names already exist")
    raw_name, wnew, bnew = PREFIX + "raw_patches", PREFIX + "weight", PREFIX + "bias"
    matrix = w.astype(np.float64).reshape(384, 96)
    folded_weight = (matrix.T / 127.5).astype(np.float32)
    folded_bias = (b.astype(np.float64) - matrix.sum(axis=1)).astype(np.float32)
    tail.graph.initializer.extend([
        numpy_helper.from_array(folded_weight, wnew),
        numpy_helper.from_array(folded_bias, bnew),
    ])
    head = [
        helper.make_node("MatMul", [raw_name, wnew], [PREFIX + "projected"], name=PREFIX + "project"),
        helper.make_node("Add", [PREFIX + "projected", bnew], [token_name], name=PREFIX + "add_bias"),
    ]
    old_nodes = list(tail.graph.node)
    del tail.graph.node[:]
    tail.graph.node.extend(head + old_nodes)
    del tail.graph.input[:]
    tail.graph.input.extend([helper.make_tensor_value_info(raw_name, TensorProto.FLOAT, [1, 128, 96])])
    # This tensor's shape is established by the new explicit MatMul operands.
    for value in tail.graph.value_info:
        if value.name == token_name:
            value.CopyFrom(helper.make_tensor_value_info(token_name, TensorProto.FLOAT, [1, 128, 384]))
    onnx.checker.check_model(tail)
    return tail


def verify_projection_on_npu(report):
    html = Path(report).read_text(encoding="utf-8")
    graph = None
    for match in re.finditer(r"\bvar\s+graphData\s*=\s*(\{)", html):
        try:
            value, _ = json.JSONDecoder().raw_decode(html[match.start(1):])
        except ValueError:
            continue
        if isinstance(value, dict) and "nodes" in value and "subgraphs" in value:
            graph = value
            break
    if graph is None:
        raise ValueError("Cannot read partition graph from compilation summary; inspect the report")
    devices = {part["id"]: part.get("device") for part in graph["subgraphs"]}
    projections = [node for node in graph["nodes"]
                   if str(node.get("id", "")).startswith(PREFIX + "project")]
    if not projections or any(devices.get(node.get("subgraph_id")) != "NPU" for node in projections):
        raise ValueError("Learned projection is not confirmed on NPU; a CPU-only result is not an NPU test")
    return {"projection_nodes": [node["id"] for node in projections], "partitions": devices}


def compile_probe(args):
    import onnx
    import torch
    import dx_com
    from torch.utils.data import Dataset, DataLoader
    source = Path(args.source).resolve()
    paths = discover(args.input)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    original = onnx.load(str(source))
    reference_file = output / "original_reference.onnx"
    model_file = output / "parseq_rawpatch_matmul.onnx"
    onnx.save(original, reference_file)
    onnx.save(replace_head(original), model_file)
    ref, mm = ort_session(reference_file), ort_session(model_file)
    rows = []
    for image in paths:
        pixels = rgb_pixels(image)
        a = ref.run(None, {ref.get_inputs()[0].name: normalize(raw_nchw(pixels))})[0]
        z = mm.run(None, {mm.get_inputs()[0].name: pack_pixels(pixels)})[0]
        if (a.shape != z.shape or not np.isfinite(a).all() or not np.isfinite(z).all()
                or not np.allclose(a, z, rtol=1e-3, atol=1e-4)):
            raise ValueError(f"MatMul replacement is not equivalent on {image}")
        error = float(np.max(np.abs(a-z)))
        rows.append({"image": image.name, "sha256": sha256(image), "max_abs": error})
        print(f"CPU EQUIVALENCE OK {image.name}: max_abs={error:.8g}", flush=True)
    manifest = {"scope": "raw-patch MatMul replacement; CPU patch packing + compiled projection/tail (see placement)",
                "original_sha256": sha256(reference_file), "matmul_sha256": sha256(model_file),
                "input_contract": "float32 raw pixel patches [1,128,96], C/dy/dx order, values [0,255]",
                "normalization": "Folded into W'=W/127.5 and b'=b-sum(W)",
                "calibration": rows, "status": "cpu_equivalence_verified"}
    save_json(output / "matmul_contract.json", manifest)
    del ref, mm

    class Patches(Dataset):
        def __len__(self):
            return len(paths)

        def __getitem__(self, index):
            return torch.from_numpy(pack_pixels(rgb_pixels(paths[index]))[0])

    torch.set_num_threads(4)
    dx_com.compile(
        model=str(model_file), output_dir=str(output / "compiled"),
        dataloader=DataLoader(Patches(), batch_size=1, shuffle=False, num_workers=0),
        calibration_method="ema", calibration_num=len(paths), quantization_device="cpu",
        opt_level=0, gen_log=True, export_html=True,
    )
    files = sorted((output / "compiled").rglob("*.dxnn"))
    if len(files) != 1 or not files[0].stat().st_size:
        raise RuntimeError(f"Expected one nonempty DXNN, got {files}")
    placement = verify_projection_on_npu(output / "compiled/parseq_rawpatch_matmul_summary.html")
    print("VERIFIED NPU PLACEMENT:", json.dumps(placement), flush=True)
    manifest.update(status="compiled", dxnn=files[0].relative_to(output).as_posix(),
                    dxnn_sha256=sha256(files[0]), placement=placement)
    save_json(output / "matmul_contract.json", manifest)
    print("DXNN:", files[0], flush=True)
    return 0


def compare_probe(args):
    from dx_engine import InferenceEngine
    from dx_engine.inference_option import InferenceOption
    run = Path(args.run).resolve()
    manifest = json.loads((run / "matmul_contract.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "compiled":
        raise ValueError("Compile did not finish")
    model = run / manifest["dxnn"]
    reference = run / "original_reference.onnx"
    if sha256(model) != manifest["dxnn_sha256"] or sha256(reference) != manifest["original_sha256"]:
        raise ValueError("Model hash differs from the manifest")
    output = Path("results") / datetime.now().strftime("dxm1_matmul_compare_%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    ref = ort_session(reference)
    option = InferenceOption()
    option.use_ort = True
    rows = []
    with InferenceEngine(str(model), option) as engine:
        if not option.use_ort:
            raise RuntimeError("ORT support is required for any CPU tasks")
        inputs, outputs = engine.get_input_tensors_info(), engine.get_output_tensors_info()
        info = {"inputs": [{**x, "dtype": np.dtype(x["dtype"]).name} for x in inputs],
                "outputs": [{**x, "dtype": np.dtype(x["dtype"]).name} for x in outputs]}
        save_json(output / "runtime_io.json", info)
        print("RUNTIME IO:", json.dumps(info), flush=True)
        # No guessed quantization/uint8 reinterpretation: require the real float
        # interface. Only harmless leading singleton dimensions may differ.
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError("Expected one input/output")
        shape = list(inputs[0]["shape"])
        if (len(shape) < 3 or shape[-2:] != [128,96] or np.prod(shape[:-2]) != 1
                or np.dtype(inputs[0]["dtype"]) != np.dtype("float32")
                or engine.get_input_size() != 128*96*4):
            raise ValueError("Compiler did not expose the expected FLOAT32 patch input. "
                             "Stop and inspect runtime_io.json; do not guess input scale.")
        if np.dtype(outputs[0]["dtype"]) != np.dtype("float32"):
            raise ValueError("Expected float32 output")
        for image in discover(args.input):
            pixels = rgb_pixels(image)
            feed = np.empty(shape, np.float32)
            np.copyto(feed.reshape(1,128,96), pack_pixels(pixels))
            expected = ref.run(None, {ref.get_inputs()[0].name: normalize(raw_nchw(pixels))})[0]
            values = engine.run([feed])
            if len(values) != 1:
                raise ValueError("Unexpected output count")
            actual = np.array(values[0], copy=True)
            if (actual.shape != expected.shape and actual.shape[-2:] == expected.shape[-2:]
                    and actual.size == expected.size):
                actual = actual.reshape(expected.shape)
            metrics = feature_metrics(expected, actual)
            row = {"image": image.name, **metrics}
            rows.append(row)
            print(f"{image.name}: cosine={metrics['feature_cosine']:.6f}; "
                  f"relL2={metrics['feature_relative_l2']:.6f}; max_abs={metrics['feature_max_abs']:.6g}", flush=True)
    save_json(output / "feature_comparison.json", {"rows": rows, "dxnn_sha256": sha256(model),
              "note": "Diagnostic prefix/features only. No ground-truth OCR accuracy or full-model pass claim."})
    print("Saved:", output.resolve(), flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("compile")
    build.add_argument("--source", required=True)
    build.add_argument("--input", default="demo_images")
    build.add_argument("--output", required=True)
    build.set_defaults(handler=compile_probe)
    compare = commands.add_parser("compare")
    compare.add_argument("--run", required=True)
    compare.add_argument("--input", default="demo_images")
    compare.set_defaults(handler=compare_probe)
    args = parser.parse_args()
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
