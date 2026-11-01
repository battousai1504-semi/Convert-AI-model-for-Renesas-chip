#!/usr/bin/env python3
"""Existing automatic ALPR pipeline with DX-M1 encoder and PyTorch decoder.

Example: python alpr_auto_dxm1.py --dx-run artifacts/FULL_ENCODER_RUN --source image.jpg --device cpu
All detector/crop/line-split/tracker/preview/CSV behavior comes from alpr_auto.py.
"""
import argparse
import hashlib
import json
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from dxm1_parseq_backend import Dxm1Encoder


@contextmanager
def reader_override(base, run, factory=None):
    original = base.ParseqReader
    factory = factory or Dxm1Encoder
    readers = []

    class Reader(original):
        def __init__(self, device="cpu", checkpoint="pretrained=parseq"):
            super().__init__(device, checkpoint)
            self.dxm1_backend = factory(self.model, run, device)
            readers.append(self)

    base.ParseqReader = Reader
    try:
        yield readers
    finally:
        base.ParseqReader = original
        for reader in readers:
            reader.dxm1_backend.close()


def annotate_summary(destination, reports, base_path, run):
    path = Path(destination) / "summary.json"
    if not path.is_file() or not reports:
        return
    report = json.loads(path.read_text(encoding="utf-8"))
    report.update(
        scope="Automatic ALPR: ORT CPU detector, CPU crop/patch packing, compiled DX-M1 encoder "
              "(see placement for any CPU fallback), unchanged PyTorch decoder and tracking",
        ocr_encoder_backend="DX-M1",
        ocr_decoder_device=reports[0]["decoder_device"],
        deepx_encoder=reports[0],
        alpr_base_sha256=hashlib.sha256(Path(base_path).read_bytes()).hexdigest(),
    )
    report.setdefault("arguments", {})["dx_run"] = str(Path(run).resolve())
    report["ocr_device_note"] = "Legacy ocr_device refers to PyTorch decoder; encoder backend is explicit above."
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def main(argv=None):
    import alpr_auto as base
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--dx-run")
    parser.add_argument("--output")
    args, remaining = parser.parse_known_args(argv)
    if "--help" in remaining or "-h" in remaining:
        print(__doc__)
        return base.main(["--help"])
    if not args.dx_run:
        parser.error("--dx-run must point to a successfully compiled FULL MatMul encoder run")
    run = Path(args.dx_run).resolve()
    if not (run / "matmul_contract.json").is_file():
        parser.error(f"Cannot find {run / 'matmul_contract.json'}")
    destination = Path(args.output or Path("results") /
                       datetime.now().strftime("auto_dxm1_%Y%m%d_%H%M%S_%f")).resolve()
    print("Encoder: DX-M1; --device selects PyTorch DECODER only; detector: ORT CPU", flush=True)
    with reader_override(base, run) as readers:
        try:
            result = base.main([*remaining, "--output", str(destination)])
        finally:
            annotate_summary(destination, [r.dxm1_backend.report() for r in readers], base.__file__, run)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
