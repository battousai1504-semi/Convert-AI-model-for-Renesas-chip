#!/usr/bin/env python3
"""Compile only PARSeq's patch embedding + positional addition for diagnosis.

Run inside the existing DX-COM CPU container, then audit with audit_dxm1_io.py.
This prefix is not a complete encoder and must not be used for OCR decoding.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

from probe_parseq_dxm1 import compile_model, sha256, save_json


def verify_static_output(prefix):
    """Specialize unknown output dimensions only after CPU execution verifies them.

    DimensionProto.dim_value defaults to 0 when a dim_param/unknown is set;
    that default is not evidence that the real tensor has zero elements.
    Concrete incompatible dimensions remain errors and are never overwritten.
    """
    import numpy as np
    import onnx
    import onnxruntime as ort
    expected_input, expected_output = [1, 3, 32, 128], [1, 128, 384]
    if len(prefix.graph.input) != 1 or len(prefix.graph.output) != 1:
        raise ValueError("Expected one prefix input and one prefix output")
    input_tensor = prefix.graph.input[0].type.tensor_type
    input_shape = [d.dim_value if d.HasField("dim_value") else None
                   for d in input_tensor.shape.dim]
    if input_shape != expected_input or input_tensor.elem_type != onnx.TensorProto.FLOAT:
        raise ValueError(f"Cannot specialize a non-static float32 input: {input_shape}")
    output = prefix.graph.output[0]
    tensor = output.type.tensor_type
    metadata = [int(d.dim_value) if d.HasField("dim_value") else (d.dim_param or None)
                for d in tensor.shape.dim]
    if tensor.elem_type != onnx.TensorProto.FLOAT or len(tensor.shape.dim) != 3:
        raise ValueError(f"Unexpected output type/rank: {metadata}")
    for dimension, expected in zip(tensor.shape.dim, expected_output):
        if dimension.HasField("dim_value") and dimension.dim_value != expected:
            raise ValueError(f"Concrete embedding output dimension disagrees with contract: {metadata}")
    onnx.checker.check_model(prefix)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(prefix.SerializeToString(), sess_options=options,
                                   providers=["CPUExecutionProvider"])
    checks = []
    for name, values in [
        ("zero", np.zeros(expected_input, dtype=np.float32)),
        ("ramp", np.linspace(-1, 1, int(np.prod(expected_input)), dtype=np.float32)
         .reshape(expected_input)),
    ]:
        actual = session.run([output.name], {prefix.graph.input[0].name: values})[0]
        if list(actual.shape) != expected_output or actual.dtype != np.float32:
            raise ValueError(f"Real prefix output on {name}: {actual.shape} {actual.dtype}")
        if not np.isfinite(actual).all():
            raise ValueError(f"Non-finite prefix output on {name}")
        checks.append({"probe": name, "shape": list(actual.shape), "dtype": str(actual.dtype)})
    # Input is fixed batch=1, positional Add is the selected boundary, and the
    # real output was checked above. No graph operations or weights are changed.
    static_info = onnx.helper.make_tensor_value_info(
        output.name, onnx.TensorProto.FLOAT, expected_output)
    output.CopyFrom(static_info)
    for value in prefix.graph.value_info:
        if value.name == output.name:
            value.CopyFrom(static_info)
    onnx.checker.check_model(prefix)
    return {"output_shape_metadata_before": metadata,
            "runtime_shape_checks": checks, "output_shape": expected_output}


def extract_embedding(model):
    import onnx
    from onnx import shape_inference
    from onnx.utils import Extractor
    if len(model.graph.input) != 1 or len(model.graph.output) != 1:
        raise ValueError("Expected a single-input, single-output original encoder")
    # The positional tensor identifies this boundary by graph dataflow, not a
    # guessed node number. Follow Identity aliases used by some ONNX exporters.
    positional = {v.name for v in model.graph.initializer
                  if "pos_embed" in v.name.lower() and list(v.dims) == [1, 128, 384]}
    if not positional:
        raise ValueError("Cannot find positional embedding initializer [1,128,384]; no cut was guessed")
    changed = True
    while changed:
        changed = False
        for node in model.graph.node:
            if node.op_type == "Identity" and len(node.input) == 1 and node.input[0] in positional:
                for name in node.output:
                    if name not in positional:
                        positional.add(name)
                        changed = True
    candidates = [node for node in model.graph.node
                  if node.op_type == "Add" and any(name in positional for name in node.input)]
    if len(candidates) != 1 or len(candidates[0].output) != 1:
        raise ValueError(f"Expected one positional Add, found {[n.name for n in candidates]}; refusing to guess")
    target = candidates[0]
    inferred = shape_inference.infer_shapes(model)
    prefix = Extractor(inferred).extract_model([model.graph.input[0].name], [target.output[0]])
    if any(n.op_type == "Softmax" for n in prefix.graph.node):
        raise ValueError("Selected prefix contains Softmax; refusing an ambiguous attention boundary")
    shape_info = verify_static_output(prefix)
    info = {"scope": "patch embedding and positional Add only, before Transformer blocks",
            "cut_node": target.name, "cut_tensor": target.output[0],
            "source_nodes": len(model.graph.node), "prefix_nodes": len(prefix.graph.node),
            "prefix_operators": dict(Counter(n.op_type for n in prefix.graph.node)),
            **shape_info}
    return prefix, info


def main():
    import onnx
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Original normalized-input encoder ONNX")
    parser.add_argument("--input", default="demo_images")
    parser.add_argument("--output", required=True, help="NEW prefix run directory")
    args = parser.parse_args()
    print("Embedding prefix extractor v2: verify real output shape before specialization", flush=True)
    source, output = Path(args.source).resolve(), Path(args.output).resolve()
    source_dir = output.with_name(output.name + "_source")
    if output.exists() or source_dir.exists():
        raise FileExistsError("Choose a NEW output name; existing artifacts will not be overwritten")
    prefix, info = extract_embedding(onnx.load(str(source)))
    source_dir.mkdir(parents=True, exist_ok=False)
    prefix_path = source_dir / "parseq_embedding_prefix.onnx"
    onnx.save(prefix, prefix_path)
    info.update(source_sha256=sha256(source), prefix_sha256=sha256(prefix_path))
    save_json(source_dir / "extraction.json", info)
    print(json.dumps(info, indent=2), flush=True)
    try:
        code = compile_model(argparse.Namespace(
            encoder=str(prefix_path), input=args.input, output=str(output)))
    finally:
        contract_path = output / "contract.json"
        if contract_path.is_file():
            contract = json.loads(contract_path.read_text(encoding="utf-8"))
            contract.update(scope="DEBUG PREFIX: patch embedding + positional Add only; NOT an OCR encoder",
                            artifact_kind="embedding_prefix", extraction=info)
            save_json(contract_path, contract)
    print("Prefix compiled. Next: audit_dxm1_io.py with this run as --reference-run.", flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
