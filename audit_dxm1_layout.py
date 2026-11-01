#!/usr/bin/env python3
"""Check explicit output-layout hypotheses on saved arrays; never fixes outputs."""
import argparse
import csv
from datetime import datetime
from pathlib import Path

import numpy as np
from probe_parseq_dxm1 import feature_metrics


def hypotheses(values):
    if values.shape != (1, 128, 384):
        raise ValueError(f"Expected [1,128,384], got {values.shape}")
    raw = values.ravel()
    result = {
        "identity": values,
        "channel_first": raw.reshape(384, 128).T.reshape(1, 128, 384),
        "patch_axes_swapped": raw.reshape(16, 8, 384).transpose(1, 0, 2).reshape(1, 128, 384),
        "channel_first_patch_axes_swapped": raw.reshape(384, 16, 8).transpose(2, 1, 0).reshape(1, 128, 384),
    }
    for block in (8, 16, 32, 64):
        result[f"channel_block_{block}"] = (
            raw.reshape(384 // block, 8, 16, block)
            .transpose(1, 2, 0, 3).reshape(1, 128, 384))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", required=True)
    args = parser.parse_args()
    root = Path(args.audit).resolve()
    folders = sorted(p.parent for p in root.glob("*/reference_onnx.npy"))
    if not folders:
        raise ValueError(f"No saved feature pairs in {root}")
    rows, candidates = [], []
    for folder in folders:
        reference = np.load(folder / "reference_onnx.npy", allow_pickle=False)
        actual = np.load(folder / "python_dxrt.npy", allow_pickle=False)
        scores = {name: feature_metrics(reference, value)["feature_cosine"]
                  for name, value in hypotheses(actual).items()}
        best = max(scores, key=scores.get)
        print(f"{folder.name}: identity={scores['identity']:.6f}; best={best} {scores[best]:.6f}", flush=True)
        # A triage flag, NOT a model validation threshold or a deployment fix.
        strong = {name for name, score in scores.items()
                  if name != "identity" and score >= 0.98 and score - scores['identity'] >= 0.2}
        candidates.append(strong)
        rows.extend({"sample": folder.name, "hypothesis": name, "cosine": score}
                    for name, score in scores.items())
    report = root / datetime.now().strftime("layout_hypotheses_%Y%m%d_%H%M%S_%f.csv")
    with report.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sample", "hypothesis", "cosine"])
        writer.writeheader()
        writer.writerows(rows)
    print("Layout report:", report, flush=True)
    consistent = set.intersection(*candidates)
    if consistent:
        print("CONSISTENT LAYOUT CANDIDATE:", sorted(consistent), flush=True)
        print("Stop for inspection; original outputs were NOT modified.", flush=True)
        return 10
    print("No common high-cosine layout candidate among the tested hypotheses.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
