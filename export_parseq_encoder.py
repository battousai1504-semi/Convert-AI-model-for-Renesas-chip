#!/usr/bin/env python3
r"""Export and validate the PARSeq IMAGE ENCODER, not the complete OCR model.

Place beside read.py in the existing baudm/parseq checkout. Keep original repo
files unchanged. Tested dependency target: torch 2.2.1, timm 0.9.16, numpy 1.26.4,
onnx 1.16.1, onnxruntime 1.18.1, Python 3.10.

PowerShell:
  .\.venv\Scripts\python.exe -m pip install numpy==1.26.4 onnx==1.16.1 onnxruntime==1.18.1
  .\.venv\Scripts\python.exe export_parseq_encoder.py --input inputs

Outputs, under artifacts/encoder_<timestamp>/:
  parseq_encoder.onnx   static float32 image input, batch size 1
  encoder_validation.csv
  export_report.json   dimensions, operator inventory, hash and validation status

Validation compares:
  A. Original PyTorch encoder features vs ONNX Runtime encoder features.
  B. Original full PyTorch PARSeq vs ONNX encoder + ORIGINAL PyTorch decoder.
All comparisons use CPU and identical image preprocessing. The decoder retains
autoregressive stopping and refinement; no max_length override or decoding-mode
change is made. An export copy has fused attention disabled to expose ordinary
ONNX operations. The original model used for reference is left unchanged.

This is NOT a DEEPX-ready full model. The output graph is only the encoder.
It must still be compiled/tested with the actual DEEPX SDK and hardware.
Prediction agreement checks conversion fidelity, NOT OCR ground-truth accuracy.
Do not use the validation runtimes as a performance benchmark.
"""

import argparse
import copy
import csv
import hashlib
import json
import platform
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
CSV_FIELDS = ["image", "pytorch_text", "hybrid_text", "text_equal", "features_close",
              "features_max_abs_error", "logits_close", "logits_max_abs_error", "status", "error"]


def collect_images(value):
    root = Path(value).resolve()
    if root.is_file() and root.suffix.lower() in IMAGE_EXTENSIONS:
        return root.parent, [root]
    if root.is_dir():
        paths = sorted(p for p in root.rglob("*")
                       if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
        if paths:
            return root, paths
    raise ValueError(f"No supported images found at {root}")


def compare_arrays(reference, actual, *, rtol, atol):
    if reference.shape != actual.shape:
        return {"close": False, "max_abs_error": None,
                "reference_shape": list(reference.shape), "actual_shape": list(actual.shape)}
    finite = bool(np.isfinite(reference).all() and np.isfinite(actual).all())
    return {
        "close": finite and bool(np.allclose(reference, actual, rtol=rtol, atol=atol)),
        "max_abs_error": float(np.max(np.abs(reference - actual))) if finite else None,
        "reference_shape": list(reference.shape), "actual_shape": list(actual.shape),
    }


def save_validation(output, rows, report):
    with (output / "encoder_validation.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    (output / "export_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default="inputs", help="Single-line text crops: image file or directory")
    parser.add_argument("--output", help="Output directory; default creates a new timestamped directory")
    parser.add_argument("--checkpoint", default="pretrained=parseq")
    args = parser.parse_args()
    try:
        root, paths = collect_images(args.input)
    except ValueError as exc:
        parser.error(str(exc))

    try:
        import torch
        import timm
        import onnx
        import onnxruntime as ort
        from PIL import Image, ImageOps
        from strhub.data.module import SceneTextDataModule
        from strhub.models.utils import load_from_checkpoint
    except ImportError as exc:
        raise SystemExit(f"Missing dependency: {exc}. Use the SAME .venv as read.py and install ONNX dependencies.")

    output = Path(args.output or Path("artifacts") /
                  datetime.now().strftime("encoder_%Y%m%d_%H%M%S_%f")).resolve()
    output.mkdir(parents=True, exist_ok=True)
    model_path = output / "parseq_encoder.onnx"
    if model_path.exists():
        parser.error(f"Refusing to overwrite {model_path}. Choose a new --output directory.")
    print(f"Output directory: {output}")
    print("Loading original PARSeq on CPU ...")
    model = load_from_checkpoint(args.checkpoint).eval().cpu()
    core = model.model
    original_encode = core.encode
    transform = SceneTextDataModule.get_transform(model.hparams.img_size)

    def prepare(path):
        with Image.open(path) as original:
            image = ImageOps.exif_transpose(original).convert("RGB")
            return transform(image).unsqueeze(0).to(dtype=torch.float32, device="cpu")

    sample = None
    for path in paths:
        try:
            sample = prepare(path)
            break
        except (OSError, ValueError):
            continue
    if sample is None:
        parser.error("Every input image failed to open. Fix inputs before exporting.")

    # Export ONLY a copy. The original reference attention path is preserved.
    export_encoder = copy.deepcopy(core.encoder).eval().cpu()
    unfused = 0
    for module in export_encoder.modules():
        if hasattr(module, "fused_attn") and module.fused_attn:
            module.fused_attn = False
            unfused += 1
    print(f"Exporting encoder, input={list(sample.shape)}, opset=17 ...")
    with torch.no_grad():
        torch.onnx.export(export_encoder, sample, str(model_path), export_params=True,
                          opset_version=17, do_constant_folding=True,
                          input_names=["images"], output_names=["memory"],
                          training=torch.onnx.TrainingMode.EVAL)
    onnx.checker.check_model(str(model_path))
    graph = onnx.load(str(model_path))
    operators = Counter((node.domain or "ai.onnx") + "::" + node.op_type for node in graph.graph.node)
    del export_encoder
    options = ort.SessionOptions()
    options.intra_op_num_threads = max(1, torch.get_num_threads())
    session = ort.InferenceSession(str(model_path), sess_options=options,
                                   providers=["CPUExecutionProvider"])

    def onnx_encode(images):
        # The unchanged PyTorch decoder consumes these features on CPU.
        values = np.ascontiguousarray(images.detach().cpu().numpy(), dtype=np.float32)
        memory = session.run(["memory"], {"images": values})[0]
        return torch.from_numpy(memory).to(device=images.device, dtype=images.dtype)

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "checkpoint": args.checkpoint,
        "scope": "Encoder-only ONNX; hybrid validation uses the original PyTorch decoder.",
        "python": platform.python_version(), "torch": torch.__version__, "timm": timm.__version__,
        "onnx": onnx.__version__, "onnxruntime": ort.__version__, "numpy": np.__version__,
        "opset": 17, "providers": session.get_providers(), "input_shape": list(sample.shape),
        "input_dtype": "float32", "image_size_hw": list(model.hparams.img_size),
        "preprocessing": "RGB, EXIF orientation, repo bicubic resize + ToTensor + Normalize(0.5,0.5)",
        "decode_ar": bool(core.decode_ar), "refine_iters": int(core.refine_iters),
        "max_label_length": int(core.max_label_length), "decoding_settings_changed": False,
        "export_copy_unfused_attention_blocks": unfused,
        "feature_tolerance": {"rtol": 1e-3, "atol": 1e-4},
        "logit_tolerance": {"rtol": 1e-3, "atol": 1e-3},
        "operator_counts": dict(sorted(operators.items())), "onnx_sha256": file_sha256(model_path),
        "onnx_checker_passed": True,
        "ground_truth_accuracy": None, "deepx_compatibility": "NOT_TESTED",
    }
    rows = []
    with torch.no_grad():
        for index, path in enumerate(paths, 1):
            row = dict.fromkeys(CSV_FIELDS, "")
            row["image"] = path.relative_to(root).as_posix()
            try:
                image = prepare(path)
                features_pt = original_encode(image).detach().cpu().numpy()
                features_ort = onnx_encode(image).detach().cpu().numpy()
                features = compare_arrays(features_pt, features_ort, rtol=1e-3, atol=1e-4)
                logits_pt = model(image)
                text_pt = model.tokenizer.decode(logits_pt.softmax(-1))[0][0]
                # Preserve data-dependent AR loop and refinement in the original forward.
                try:
                    core.encode = onnx_encode
                    logits_hybrid = model(image)
                finally:
                    core.encode = original_encode
                text_hybrid = model.tokenizer.decode(logits_hybrid.softmax(-1))[0][0]
                logits = compare_arrays(logits_pt.detach().cpu().numpy(),
                                        logits_hybrid.detach().cpu().numpy(), rtol=1e-3, atol=1e-3)
                passed = features["close"] and logits["close"] and text_pt == text_hybrid
                row.update(pytorch_text=text_pt, hybrid_text=text_hybrid,
                           text_equal=int(text_pt == text_hybrid),
                           features_close=int(features["close"]),
                           features_max_abs_error=features["max_abs_error"],
                           logits_close=int(logits["close"]),
                           logits_max_abs_error=logits["max_abs_error"],
                           status="pass" if passed else "mismatch")
                print(f"[{index}/{len(paths)}] {row['status'].upper()} {row['image']}: PyTorch={text_pt!r}, hybrid={text_hybrid!r}")
            except Exception as exc:
                row.update(status="error", error=f"{type(exc).__name__}: {exc}")
                print(f"[{index}/{len(paths)}] ERROR {row['image']}: {row['error']}")
            rows.append(row)
    passed = sum(row["status"] == "pass" for row in rows)
    report.update(validation_images=len(rows), passed_images=passed,
                  all_validations_passed=passed == len(rows))
    with torch.no_grad():
        report["encoder_output_shape"] = list(original_encode(sample).shape)
    save_validation(output, rows, report)
    if report["all_validations_passed"]:
        print(f"\nPASS: encoder-only ONNX matches the reference on {passed}/{len(rows)} images.")
        print("The decoder still runs in PyTorch. DEEPX compilation/hardware are NOT verified.")
        print(f"Files: {output}")
        return 0
    print(f"\nFAIL: {passed}/{len(rows)} images passed. Inspect CSV/report before deployment.")
    print(f"Files: {output}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
