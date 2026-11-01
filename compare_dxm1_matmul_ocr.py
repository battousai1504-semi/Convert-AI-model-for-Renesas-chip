#!/usr/bin/env python3
"""Full PARSeq: compare PyTorch OCR with DX-M1 raw-patch encoder + CPU decoder.

Requires a FULL encoder run from probe_dxm1_matmul_head.py, not an embedding run.
Keeps decoder autoregression, early stopping, max length and refinement unchanged.
"""
import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from probe_parseq_dxm1 import (
    discover, rgb_pixels, raw_nchw, normalize, ort_session, sha256,
    save_json, feature_metrics,
)
from probe_dxm1_matmul_head import pack_pixels


def load_run(run):
    import onnx
    run = Path(run).resolve()
    manifest = json.loads((run / "matmul_contract.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "compiled":
        raise ValueError("The MatMul encoder compilation did not finish")
    model, reference = run / manifest["dxnn"], run / "original_reference.onnx"
    for path, digest in [(model, manifest["dxnn_sha256"]),
                         (reference, manifest["original_sha256"]),
                         (run / "parseq_rawpatch_matmul.onnx", manifest["matmul_sha256"])]:
        if sha256(path) != digest:
            raise ValueError(f"Artifact hash differs from manifest: {path}")
    original = onnx.load(str(reference))
    if not any(node.op_type == "Softmax" for node in original.graph.node):
        raise ValueError("Reference contains no attention Softmax. Use a FULL encoder, not the embedding prefix")
    placement = manifest.get("placement", {})
    if not placement.get("projection_nodes") or "NPU" not in placement.get("partitions", {}).values():
        raise ValueError("Manifest does not confirm NPU projection placement")
    return manifest, model, reference


def check_runtime(engine):
    inputs, outputs = engine.get_input_tensors_info(), engine.get_output_tensors_info()
    info = {"inputs": [{**x, "dtype": np.dtype(x["dtype"]).name} for x in inputs],
            "outputs": [{**x, "dtype": np.dtype(x["dtype"]).name} for x in outputs]}
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError(f"Expected one encoder input/output: {info}")
    shape, out_shape = list(inputs[0]["shape"]), list(outputs[0]["shape"])
    if (len(shape) < 3 or shape[-2:] != [128, 96] or np.prod(shape[:-2]) != 1
            or np.dtype(inputs[0]["dtype"]) != np.dtype("float32")
            or engine.get_input_size() != 128 * 96 * 4):
        raise ValueError(f"Unexpected raw-patch input contract: {info}")
    if (len(out_shape) < 3 or out_shape[-2:] != [128, 384] or np.prod(out_shape[:-2]) != 1
            or np.dtype(outputs[0]["dtype"]) != np.dtype("float32")
            or engine.get_output_size() != 128 * 384 * 4):
        raise ValueError(f"Unexpected encoder memory contract: {info}")
    return shape, info


def decode_with_memory(model, images, memory):
    """Replace only the encoder call and always restore it, even on failure."""
    import torch
    core, original_encode = model.model, model.model.encode
    values = torch.from_numpy(np.array(memory, dtype=np.float32, copy=True))

    def cached_encode(current_images):
        if current_images.shape[0] != 1:
            raise ValueError("Only batch=1 is supported")
        return values.to(device=current_images.device, dtype=current_images.dtype)

    try:
        core.encode = cached_encode
        return model(images)
    finally:
        core.encode = original_encode


def main():
    import onnxruntime as ort
    import torch
    from PIL import Image, ImageOps
    from importlib.metadata import version
    from dx_engine import InferenceEngine
    from dx_engine.inference_option import InferenceOption
    from strhub.models.utils import load_from_checkpoint
    from strhub.data.module import SceneTextDataModule

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--input", default="demo_images", help="Single-line crops, not full-frame photos/video")
    parser.add_argument("--checkpoint", default="pretrained=parseq")
    parser.add_argument("--output", help="A NEW directory; defaults to timestamped results")
    args = parser.parse_args()
    manifest, dxnn, reference_file = load_run(args.run)
    paths = discover(args.input)
    output = Path(args.output or Path("results") /
                  datetime.now().strftime("dxm1_full_ocr_%Y%m%d_%H%M%S_%f")).resolve()
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    model = load_from_checkpoint(args.checkpoint).eval().cpu()
    if list(model.hparams.img_size) != [32, 128]:
        raise ValueError("Only standard PARSeq 32x128 is supported")
    core = model.model
    transform = SceneTextDataModule.get_transform(model.hparams.img_size)
    reference = ort_session(reference_file)
    options = InferenceOption()
    options.use_ort = True
    rows = []
    with InferenceEngine(str(dxnn), options) as engine, torch.inference_mode():
        if not options.use_ort:
            raise RuntimeError("DX-RT disabled ORT; CPU tasks would be skipped")
        shape, info = check_runtime(engine)
        save_json(output / "runtime_io.json", info)
        print("FULL ENCODER RUNTIME IO:", json.dumps(info), flush=True)
        for index, image in enumerate(paths, 1):
            row = {"image": str(image), "status": "error", "error": ""}
            try:
                pixels = rgb_pixels(image)
                x_array = normalize(raw_nchw(pixels))
                with Image.open(image) as original:
                    x = transform(ImageOps.exif_transpose(original).convert("RGB")).unsqueeze(0)
                if not np.allclose(x.numpy(), x_array, rtol=0, atol=1e-6):
                    raise ValueError("Pixel preparation differs from the repository transform")
                features_pt = core.encode(x).detach().numpy()
                features_ref = reference.run(None, {reference.get_inputs()[0].name: x_array})[0]
                if (features_pt.shape != features_ref.shape or not np.isfinite(features_pt).all()
                        or not np.isfinite(features_ref).all()
                        or not np.allclose(features_pt, features_ref, rtol=1e-3, atol=1e-4)):
                    raise ValueError("Source ONNX does not match the selected PyTorch encoder checkpoint")
                logits_pt = model(x)
                if not torch.isfinite(logits_pt).all():
                    raise ValueError("Non-finite reference decoder logits")
                row["pytorch_text"] = model.tokenizer.decode(logits_pt.softmax(-1))[0][0]
                feed = np.empty(shape, dtype=np.float32)
                np.copyto(feed.reshape(1, 128, 96), pack_pixels(pixels))
                begin = time.perf_counter()
                outputs = engine.run([feed])
                row["dxrt_call_ms"] = (time.perf_counter() - begin) * 1000
                if len(outputs) != 1 or outputs[0].dtype != np.float32:
                    raise ValueError("Unexpected DX-RT encoder output count/dtype")
                memory = np.array(outputs[0], copy=True)
                if memory.size != 128 * 384 or memory.shape[-2:] != (128, 384):
                    raise ValueError(f"Unexpected actual encoder memory shape: {memory.shape}")
                memory = memory.reshape(1, 128, 384)
                row.update(feature_metrics(features_pt, memory))
                logits_dx = decode_with_memory(model, x, memory)
                if not torch.isfinite(logits_dx).all():
                    raise ValueError("Non-finite hybrid decoder logits")
                text_dx = model.tokenizer.decode(logits_dx.softmax(-1))[0][0]
                equal = row["pytorch_text"] == text_dx
                row.update(status="match" if equal else "mismatch", dxm1_hybrid_text=text_dx,
                           text_equal=int(equal))
                print(f"[{index}/{len(paths)}] {row['status'].upper()} {image.name}: "
                      f"PyTorch={row['pytorch_text']!r}, DX-M1+PyTorch={text_dx!r}; "
                      f"cos={row['feature_cosine']:.6f}, relL2={row['feature_relative_l2']:.6f}", flush=True)
            except Exception as exc:
                row.update(status="error", error=f"{type(exc).__name__}: {exc}")
                print(f"[{index}/{len(paths)}] ERROR {image.name}: {row['error']}", flush=True)
            rows.append(row)
    fields = ["image", "status", "pytorch_text", "dxm1_hybrid_text", "text_equal",
              "feature_cosine", "feature_relative_l2", "feature_mae", "feature_max_abs",
              "dxrt_call_ms", "error"]
    with (output / "comparison.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    matches = sum(row["status"] == "match" for row in rows)
    errors = sum(row["status"] == "error" for row in rows)
    save_json(output / "summary.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(), "checkpoint": args.checkpoint,
        "dxnn_sha256": sha256(dxnn), "original_sha256": sha256(reference_file),
        "scope": "CPU RGB resize/patch packing + DX-M1 encoder graph + unchanged CPU PyTorch decoder",
        "placement": manifest["placement"], "images": len(rows), "text_matches": matches,
        "errors": errors, "all_texts_match": matches == len(rows), "ground_truth_accuracy": None,
        "note": "Agreement with PyTorch measures conversion fidelity, not OCR accuracy. "
                "INT8 feature errors are measured, not judged by FP32 allclose. "
                "dxrt_call_ms includes transfers/runtime and is not full-pipeline FPS.",
        "torch": torch.__version__, "numpy": np.__version__, "onnxruntime": ort.__version__,
        "dx_engine": version("dx-engine"),
        "decoder_settings": {"decode_ar": bool(core.decode_ar), "refine_iters": int(core.refine_iters),
                             "max_label_length": int(core.max_label_length)},
    })
    print(f"Text agreement: {matches}/{len(rows)}; errors: {errors}", flush=True)
    print("CSV/report:", output, flush=True)
    return 0 if matches == len(rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
