#!/usr/bin/env python3
"""Compare DX-RT Python and native dxrun using identical raw bytes and DXNN.

This is a diagnostic experiment, not a performance benchmark or OCR accuracy test.
Requires probe_parseq_dxm1.py, the host PARSeq venv, and the installed DX-RT CLI.
"""
import argparse
import csv
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np

from probe_parseq_dxm1 import (
    discover, rgb_pixels, raw_nchw, normalize, ort_session,
    sha256, save_json, feature_metrics,
)


def read_cpp_output(path, shape):
    expected_bytes = int(np.prod(shape)) * np.dtype("<f4").itemsize
    if not path.is_file() or path.stat().st_size != expected_bytes:
        actual = path.stat().st_size if path.exists() else "missing"
        raise ValueError(f"Native output byte count: {actual}; expected {expected_bytes}")
    result = np.fromfile(path, dtype="<f4").reshape(shape)
    if not np.isfinite(result).all():
        raise ValueError("Native output contains NaN/Inf")
    return result


def centered_diagnostic(reference, actual):
    """Describe channel-offset/gain patterns; never modify inference outputs."""
    feature_metrics(reference, actual)
    a = reference.astype(np.float64).reshape(-1, reference.shape[-1])
    b = actual.astype(np.float64).reshape(-1, actual.shape[-1])
    ac, bc = a - a.mean(axis=0), b - b.mean(axis=0)
    na, nb = np.linalg.norm(ac), np.linalg.norm(bc)
    if na < 1e-12 or nb < 1e-12:
        return {"centered_cosine": None, "centered_gain": None,
                "affine_fit_residual_relative_l2": None}
    cosine = float(np.clip(np.sum(ac * bc) / (na * nb), -1, 1))
    gain = float(np.sum(ac * bc) / (na * na))
    offset = b.mean(axis=0) - gain * a.mean(axis=0)
    residual = float(np.linalg.norm(b - (gain * a + offset)) / max(np.linalg.norm(b), 1e-12))
    return {"centered_cosine": cosine, "centered_gain": gain,
            "affine_fit_residual_relative_l2": residual}


def main():
    from dx_engine import InferenceEngine
    from dx_engine.inference_option import InferenceOption
    from importlib.metadata import version
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Exact .dxnn tested by BOTH runtimes")
    parser.add_argument("--reference-run", required=True,
                        help="RGB contract directory containing the validated original encoder")
    parser.add_argument("--input", default="demo_images")
    parser.add_argument("--output", help="NEW output directory; defaults to timestamped results")
    args = parser.parse_args()
    cli = shutil.which("dxrun") or shutil.which("run_model")
    if not cli:
        raise FileNotFoundError("Cannot find native dxrun/run_model in PATH")
    help_result = subprocess.run([cli, "--help"], capture_output=True, text=True,
                                 check=False, timeout=15)
    if help_result.returncode or "--use-ort" not in help_result.stdout + help_result.stderr:
        raise RuntimeError("Native CLI does not advertise --use-ort; stop rather than skip CPU groups. "
                           "Check the DX-RT build. " + help_result.stderr)
    model = Path(args.model).resolve()
    if not model.is_file():
        raise FileNotFoundError(model)
    reference_run = Path(args.reference_run).resolve()
    source = reference_run / "encoder_normalized_reference.onnx"
    contract = json.loads((reference_run / "contract.json").read_text(encoding="utf-8"))
    if sha256(source) != contract["source_sha256"]:
        raise ValueError("Reference ONNX hash does not match the contract")
    paths = discover(args.input)
    output = Path(args.output or Path("results") /
                  datetime.now().strftime("dxm1_io_audit_%Y%m%d_%H%M%S_%f")).resolve()
    output.mkdir(parents=True, exist_ok=False)
    reference = ort_session(source)
    samples = []
    print("Output:", output, flush=True)
    print("Same model for Python/native:", model, flush=True)
    print("Model SHA256:", sha256(model), flush=True)
    print("Native CLI:", cli, flush=True)
    print("Audit v2: Python use_ort=True; native --use-ort (CPU + NPU tasks)", flush=True)

    # Finish and release the Python engine before starting any native process.
    options = InferenceOption()
    options.use_ort = True
    with InferenceEngine(str(model), options) as engine:
        if not options.use_ort:
            raise RuntimeError("Python DX-RT disabled ORT; stop rather than skip CPU groups")
        inputs, outputs = engine.get_input_tensors_info(), engine.get_output_tensors_info()
        if (len(inputs) != 1 or inputs[0]["shape"] != [1, 32, 128, 3]
                or np.dtype(inputs[0]["dtype"]) != np.dtype("uint8")
                or engine.get_input_size() != 12288):
            raise ValueError(f"Unexpected runtime input: {inputs}")
        if (len(outputs) != 1 or outputs[0]["shape"] != [1, 128, 384]
                or np.dtype(outputs[0]["dtype"]) != np.dtype("float32")
                or engine.get_output_size() != 196608):
            raise ValueError(f"Unexpected runtime output: {outputs}")
        for index, image in enumerate(paths):
            folder = output / f"{index:03d}_{image.stem}"
            folder.mkdir()
            pixels = rgb_pixels(image)
            feed = np.empty((1, 32, 128, 3), dtype=np.uint8)
            np.copyto(feed[0], pixels)
            raw_file = folder / "input_rgb_nhwc_u8.bin"
            feed.tofile(raw_file)
            # The exact serialized bytes given to C++ must match Python's feed.
            if raw_file.read_bytes() != feed.tobytes(order="C"):
                raise ValueError("Input serialization differs from Python feed")
            expected = reference.run(
                None, {reference.get_inputs()[0].name: normalize(raw_nchw(pixels))})[0]
            result = engine.run([feed])
            if len(result) != 1:
                raise ValueError("Expected one Python output")
            py = np.array(result[0], copy=True)
            feature_metrics(expected, py)  # Reject unexpected shape or NaN/Inf.
            if py.dtype != np.float32:
                raise ValueError(f"Expected Python float32 output, got {py.dtype}")
            repeated = np.array(engine.run([feed])[0], copy=True)
            feature_metrics(py, repeated)
            repeat_delta = float(np.max(np.abs(py - repeated)))
            np.save(folder / "reference_onnx.npy", expected)
            np.save(folder / "python_dxrt.npy", py)
            samples.append((image, folder, raw_file, expected, py, repeat_delta))

    rows = []
    for image, folder, raw_file, expected, py, repeat_delta in samples:
        cpp_file = folder / "native_dxrt_f32.bin"
        command = [cli, "--model", str(model), "--single", "--use-ort", "--loops", "1",
                   "--input", str(raw_file), "--output", str(cpp_file)]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        (folder / "native_console.log").write_text(
            completed.stdout + "\n" + completed.stderr, encoding="utf-8")
        if completed.returncode:
            print(completed.stdout, completed.stderr, flush=True)
            raise RuntimeError(f"Native CLI returned {completed.returncode}; see {folder / 'native_console.log'}")
        cpp = read_cpp_output(cpp_file, expected.shape)
        a, b = feature_metrics(expected, py), feature_metrics(expected, cpp)
        delta = float(np.max(np.abs(py - cpp)))
        row = {
            "image": image.name, "input_sha256": sha256(raw_file),
            "onnx_python_cosine": a["feature_cosine"],
            "onnx_native_cosine": b["feature_cosine"],
            "onnx_python_relative_l2": a["feature_relative_l2"],
            "onnx_native_relative_l2": b["feature_relative_l2"],
            "python_native_max_abs": delta,
            "python_native_close": bool(np.allclose(py, cpp, rtol=1e-5, atol=1e-5)),
            "python_repeat_max_abs": repeat_delta,
            **centered_diagnostic(expected, py),
        }
        rows.append(row)
        print(f"{image.name}: ONNX/Python cos={a['feature_cosine']:.6f}; "
              f"ONNX/C++ cos={b['feature_cosine']:.6f}; "
              f"Python/C++ max_abs={delta:.6g}; repeat max_abs={repeat_delta:.6g}", flush=True)
        if row["centered_cosine"] is not None:
            print(f"  Offset/gain diagnostic only: centered cos={row['centered_cosine']:.6f}; "
                  f"gain={row['centered_gain']:.6g}; fit residual={row['affine_fit_residual_relative_l2']:.6g}", flush=True)
    with (output / "io_audit.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    save_json(output / "io_audit.json", {
        "model": str(model), "model_sha256": sha256(model),
        "reference_onnx_sha256": sha256(source), "native_cli": cli,
        "dx_engine": version("dx-engine"), "rows": rows,
        "runtime_options": {"python_use_ort": True, "native_use_ort": True},
        "note": "Controlled I/O test. Same model hash, RGB NHWC bytes, and reference for both runtimes. "
                "No ground-truth accuracy or FPS claim. Compiler diagnosis metrics need not be per-image. "
                "Centered/affine metrics are diagnostic fits only, not corrections or validation. "
                "Runtime differences are observations, not a hardware-fault diagnosis.",
    })
    print("Audit saved:", output, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
