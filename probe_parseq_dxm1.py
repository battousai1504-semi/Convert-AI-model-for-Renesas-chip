#!/usr/bin/env python3
"""Compile an explicit RGB-input PARSeq encoder and compare real DX-M1 OCR.

Run compile in onekiwi/dxcom:2.4.1-cpu; run compare in the PARSeq venv.
Only the image encoder runs on the NPU. Decoder/AR/refinement stay in PyTorch.
"""
import argparse
import csv
import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PREFIX = "__onekiwi_rgb_v1__"
CONTRACT = "parseq-rgb-u8-nhwc-explicit-normalization-v1"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2,
                                   allow_nan=False), encoding="utf-8")


def discover(value):
    root = Path(value).resolve()
    paths = ([root] if root.is_file() else sorted(root.rglob("*")))
    paths = [p for p in paths if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    if not paths:
        raise ValueError(f"No supported images: {root}")
    return paths


def rgb_pixels(path):
    from PIL import Image, ImageOps
    with Image.open(path) as original:
        image = ImageOps.exif_transpose(original).convert("RGB")
        return np.array(image.resize((128, 32), Image.Resampling.BICUBIC),
                        dtype=np.uint8, copy=True)


def raw_nchw(pixels):
    return np.ascontiguousarray(pixels.transpose(2, 0, 1)[None], dtype=np.float32)


def normalize(raw):
    return ((raw / np.float32(255)) - np.float32(0.5)) / np.float32(0.5)


def wrap_graph(original):
    """Make the float32 ONNX interface explicitly consume pixel values 0..255.

    Raw ONNX interface: NCHW float32 pixels. DX deployment interface is required
    to be NHWC uint8 and is checked against the actual runtime before inference.
    Normalization is model arithmetic, never guessed from runtime null scales.
    """
    import copy
    import onnx
    from onnx import helper, numpy_helper, TensorProto
    graph = copy.deepcopy(original)
    if len(graph.graph.input) != 1 or len(graph.graph.output) != 1:
        raise ValueError("Expected one encoder input and one output")
    old_input = graph.graph.input[0]
    shape = [d.dim_value for d in old_input.type.tensor_type.shape.dim]
    if shape != [1, 3, 32, 128] or old_input.type.tensor_type.elem_type != TensorProto.FLOAT:
        raise ValueError(f"Expected float32 encoder [1,3,32,128], got {shape}")
    used = {x.name for x in graph.graph.initializer}
    for n in graph.graph.node:
        used.update([n.name, *n.input, *n.output])
    if any(x.startswith(PREFIX) for x in used):
        raise ValueError("Reserved wrapper prefix already exists; use the original encoder")
    original_name = old_input.name
    if original_name in {x.name for x in graph.graph.initializer}:
        raise ValueError("Encoder input cannot also be an initializer")
    raw_name = PREFIX + "pixels"
    div_name, half_name = PREFIX + "255", PREFIX + "half"
    graph.graph.initializer.extend([
        numpy_helper.from_array(np.array(255, np.float32), div_name),
        numpy_helper.from_array(np.array(0.5, np.float32), half_name),
    ])
    prefix_nodes = [
        helper.make_node("Div", [raw_name, div_name], [PREFIX + "unit"], name=PREFIX + "div255"),
        helper.make_node("Sub", [PREFIX + "unit", half_name], [PREFIX + "centered"], name=PREFIX + "subhalf"),
        helper.make_node("Div", [PREFIX + "centered", half_name], [original_name], name=PREFIX + "divhalf"),
    ]
    original_nodes = list(graph.graph.node)
    del graph.graph.node[:]
    graph.graph.node.extend(prefix_nodes + original_nodes)
    del graph.graph.input[:]
    graph.graph.input.extend([helper.make_tensor_value_info(raw_name, TensorProto.FLOAT, shape)])
    onnx.checker.check_model(graph)
    return graph


def ort_session(path):
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), sess_options=options,
                               providers=["CPUExecutionProvider"])


def compile_model(args):
    from importlib.metadata import version
    import onnx
    import torch
    import dx_com
    from torch.utils.data import Dataset, DataLoader
    encoder = Path(args.encoder).resolve()
    if not encoder.is_file():
        raise FileNotFoundError(encoder)
    paths = discover(args.input)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    original = onnx.load(str(encoder))
    source = output / "encoder_normalized_reference.onnx"
    wrapped = output / "parseq_encoder_rgb.onnx"
    onnx.save(original, str(source))
    onnx.save(wrap_graph(original), str(wrapped))
    reference, rgb_model = ort_session(source), ort_session(wrapped)
    checks = []
    for path in paths:
        raw = raw_nchw(rgb_pixels(path))
        expected = reference.run(None, {reference.get_inputs()[0].name: normalize(raw)})[0]
        actual = rgb_model.run(None, {rgb_model.get_inputs()[0].name: raw})[0]
        if (expected.shape != actual.shape or not np.isfinite(expected).all()
                or not np.isfinite(actual).all()
                or not np.allclose(expected, actual, rtol=1e-3, atol=1e-4)):
            raise ValueError(f"Explicit normalization wrapper differs from original: {path}")
        checks.append({"image": path.name, "sha256": sha256(path),
                       "max_abs_error": float(np.max(np.abs(expected - actual)))})
        print(f"ONNX WRAPPER OK: {path.name}", flush=True)
    del reference, rgb_model
    manifest = {
        "contract": CONTRACT,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "DX-M1 encoder + unchanged CPU PyTorch decoder; not full-model conversion",
        "normalization_in_onnx": "((RGB / 255) - 0.5) / 0.5",
        "onnx_input": "float32 NCHW raw RGB pixel values [0,255]",
        "expected_runtime_input": {"shape": [1, 32, 128, 3], "dtype": "uint8", "color": "RGB"},
        "exported_encoder_sha256": sha256(encoder),
        "source_sha256": sha256(source), "wrapper_sha256": sha256(wrapped),
        "calibration_images": checks, "status": "wrapper_verified",
        "calibration_note": "Smoke calibration; demo images are not a representative VN plate dataset",
        "dx_com": version("dx-com"),
        "torch": torch.__version__, "numpy": np.__version__,
    }
    save_json(output / "contract.json", manifest)

    class RawRgbDataset(Dataset):
        def __len__(self):
            return len(paths)

        def __getitem__(self, index):
            raw = raw_nchw(rgb_pixels(paths[index]))[0]
            return torch.from_numpy(raw)

    torch.set_num_threads(4)
    try:
        dx_com.compile(
            model=str(wrapped), output_dir=str(output / "compiled"),
            dataloader=DataLoader(RawRgbDataset(), batch_size=1, shuffle=False, num_workers=0),
            calibration_method="ema", calibration_num=len(paths),
            quantization_device="cpu", opt_level=0, gen_log=True, export_html=True,
        )
        files = sorted((output / "compiled").rglob("*.dxnn"))
        if len(files) != 1 or files[0].stat().st_size == 0:
            raise RuntimeError(f"Expected exactly one nonempty DXNN, got {files}")
        manifest.update(status="compiled", dxnn=files[0].relative_to(output).as_posix(),
                        dxnn_sha256=sha256(files[0]))
    except BaseException:
        manifest["status"] = "compile_failed"
        save_json(output / "contract.json", manifest)
        raise
    save_json(output / "contract.json", manifest)
    print("DXNN:", files[0], flush=True)
    print("Next: compare real images on the host using the PARSeq venv.", flush=True)
    return 0


def feature_metrics(reference, actual):
    if reference.shape != actual.shape:
        raise ValueError(f"Feature shape mismatch: {reference.shape} vs {actual.shape}")
    if not np.isfinite(reference).all() or not np.isfinite(actual).all():
        raise ValueError("Non-finite encoder features")
    a, b = reference.astype(np.float64).ravel(), actual.astype(np.float64).ravel()
    norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
    cosine = (np.dot(a, b) / (norm_a * norm_b) if norm_a and norm_b
              else float(np.array_equal(a, b)))
    return {"feature_cosine": float(np.clip(cosine, -1, 1)),
            "feature_relative_l2": float(np.linalg.norm(a - b) / max(norm_a, 1e-12)),
            "feature_mae": float(np.mean(np.abs(a - b))),
            "feature_max_abs": float(np.max(np.abs(a - b)))}


def compare_model(args):
    from importlib.metadata import version
    import torch
    import PIL
    from PIL import Image, ImageOps
    from dx_engine import InferenceEngine
    from strhub.models.utils import load_from_checkpoint
    from strhub.data.module import SceneTextDataModule
    run = Path(args.run).resolve()
    manifest = json.loads((run / "contract.json").read_text(encoding="utf-8"))
    if manifest.get("contract") != CONTRACT or manifest.get("status") != "compiled":
        raise ValueError("Use the run produced by this script's compile command")
    dxnn = run / manifest["dxnn"]
    for path, expected in [(dxnn, manifest["dxnn_sha256"]),
                           (run / "encoder_normalized_reference.onnx", manifest["source_sha256"]),
                           (run / "parseq_encoder_rgb.onnx", manifest["wrapper_sha256"])]:
        if sha256(path) != expected:
            raise ValueError(f"Artifact hash differs from contract: {path}")
    paths = discover(args.input)
    output = Path(args.output or Path("results") /
                  datetime.now().strftime("dxm1_compare_%Y%m%d_%H%M%S_%f")).resolve()
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    model = load_from_checkpoint(args.checkpoint).eval().cpu()
    if list(model.hparams.img_size) != [32, 128]:
        raise ValueError("Only standard PARSeq 32x128 is supported")
    core, original_encode = model.model, model.model.encode
    transform = SceneTextDataModule.get_transform(model.hparams.img_size)
    reference_onnx = ort_session(run / "encoder_normalized_reference.onnx")
    rows = []
    with InferenceEngine(str(dxnn)) as engine, torch.inference_mode():
        inputs, outputs = engine.get_input_tensors_info(), engine.get_output_tensors_info()
        if (len(inputs) != 1 or list(inputs[0]["shape"]) != [1, 32, 128, 3]
                or np.dtype(inputs[0]["dtype"]) != np.dtype("uint8")
                or engine.get_input_size() != 12288):
            raise ValueError(f"Unexpected runtime input; do not feed images: {inputs}")
        if (len(outputs) != 1 or list(outputs[0]["shape"]) != [1, 128, 384]
                or np.dtype(outputs[0]["dtype"]) != np.dtype("float32")
                or engine.get_output_size() != 196608):
            raise ValueError(f"Unexpected runtime output: {outputs}")
        for path in paths:
            row = {"image": str(path), "status": "error", "error": ""}
            try:
                pixels = rgb_pixels(path)
                normalized = normalize(raw_nchw(pixels))
                with Image.open(path) as original:
                    x = transform(ImageOps.exif_transpose(original).convert("RGB")).unsqueeze(0)
                if not np.allclose(x.numpy(), normalized, rtol=0, atol=1e-6):
                    raise ValueError("RGB preparation differs from the repository transform")
                features_pt = original_encode(x).detach().numpy()
                features_onnx = reference_onnx.run(
                    None, {reference_onnx.get_inputs()[0].name: normalized})[0]
                if not np.allclose(features_pt, features_onnx, rtol=1e-3, atol=1e-4):
                    raise ValueError("Reference checkpoint does not match the source ONNX within export tolerance")
                logits_pt = model(x)
                text_pt = model.tokenizer.decode(logits_pt.softmax(-1))[0][0]
                # Explicit copy writes physical memory pages before PCIe DMA.
                feed = np.empty((1, 32, 128, 3), dtype=np.uint8)
                np.copyto(feed[0], pixels)
                start = time.perf_counter()
                values = engine.run([feed])
                elapsed_ms = (time.perf_counter() - start) * 1000
                if len(values) != 1:
                    raise ValueError("Expected one runtime output")
                memory = np.array(values[0], copy=True)
                if memory.dtype != np.float32 or memory.shape != (1, 128, 384):
                    raise ValueError(f"Unexpected actual memory: {memory.shape} {memory.dtype}")
                row.update(feature_metrics(features_pt, memory))
                memory_tensor = torch.from_numpy(memory)

                def dx_encode(images):
                    return memory_tensor.to(device=images.device, dtype=images.dtype)

                try:
                    core.encode = dx_encode
                    logits_dx = model(x)
                finally:
                    core.encode = original_encode
                if not torch.isfinite(logits_pt).all() or not torch.isfinite(logits_dx).all():
                    raise ValueError("Non-finite decoder logits")
                text_dx = model.tokenizer.decode(logits_dx.softmax(-1))[0][0]
                row.update(status="match" if text_pt == text_dx else "mismatch",
                           pytorch_text=text_pt, dxm1_hybrid_text=text_dx,
                           text_equal=int(text_pt == text_dx), dxrt_call_ms=elapsed_ms)
                print(f"{row['status'].upper()} {path.name}: PyTorch={text_pt!r}, DX-M1={text_dx!r}, "
                      f"cos={row['feature_cosine']:.6f}, relL2={row['feature_relative_l2']:.6f}", flush=True)
            except Exception as exc:
                row.update(status="error", error=f"{type(exc).__name__}: {exc}")
                print(f"ERROR {path.name}: {row['error']}", flush=True)
            rows.append(row)
    fields = ["image", "status", "pytorch_text", "dxm1_hybrid_text", "text_equal",
              "feature_cosine", "feature_relative_l2", "feature_mae", "feature_max_abs",
              "dxrt_call_ms", "error"]
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    matches = sum(r["status"] == "match" for r in rows)
    errors = sum(r["status"] == "error" for r in rows)
    save_json(output / "summary.json", {
        "contract": CONTRACT, "dxnn_sha256": sha256(dxnn), "checkpoint": args.checkpoint,
        "images": len(rows), "text_matches": matches, "errors": errors,
        "all_texts_match": matches == len(rows), "ground_truth_accuracy": None,
        "note": "Text agreement measures conversion fidelity, not ground-truth OCR accuracy. "
                "INT8 feature errors are reported, not judged by FP32 allclose. "
                "Single-call times include transfers/runtime and are not an FPS benchmark.",
        "python": platform.python_version(), "torch": torch.__version__,
        "dx_engine": version("dx-engine"),
        "numpy": np.__version__, "pillow": PIL.__version__,
        "decoder_settings": {"decode_ar": bool(core.decode_ar),
                             "refine_iters": int(core.refine_iters),
                             "max_label_length": int(core.max_label_length)},
    })
    print(f"Text agreement: {matches}/{len(rows)}; errors: {errors}")
    print("Results:", output)
    return 0 if matches == len(rows) else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    compiler = sub.add_parser("compile", help="CPU container: wrap, verify ONNX equivalence, compile")
    compiler.add_argument("--encoder", required=True)
    compiler.add_argument("--input", required=True, help="Single-line calibration crops; demo_images for smoke only")
    compiler.add_argument("--output", required=True, help="A NEW directory under artifacts")
    compiler.set_defaults(handler=compile_model)
    compare = sub.add_parser("compare", help="Host PARSeq venv: real NPU + unchanged CPU decoder")
    compare.add_argument("--run", required=True)
    compare.add_argument("--input", required=True, help="Single-line text crops, not full-frame photos")
    compare.add_argument("--checkpoint", default="pretrained=parseq")
    compare.add_argument("--output", help="NEW result directory; default creates timestamped results")
    compare.set_defaults(handler=compare_model)
    args = parser.parse_args()
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
