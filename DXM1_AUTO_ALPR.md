# Automatic ALPR with DX-M1 full encoder

## Current measured milestone

The full raw-patch MatMul encoder run
`artifacts/dxm1_full_encoder_mm_20261101_134233_21170` compiled as 1 NPU group /
0 CPU groups. Full encoder features had cosine 0.971733–0.989557 against
PyTorch and the unchanged CPU PyTorch decoder returned identical text on 6/6
demo crops (0 execution errors). Those images were also calibration smoke
samples, so this is not a holdout accuracy result or a VN ALPR benchmark.

## Architecture of this entry point

- Existing `alpr_auto.py`: ORT CPU plate detector, crop/rectification, one/two
  line handling, tracking, annotated preview and CSV writing.
- New `alpr_auto_dxm1.py`: scoped reader subclass for this CLI process. It uses
  the existing reader's `read()` implementation and attaches an encoder adapter.
- New `dxm1_parseq_backend.py`: checks the full-run manifest/model hashes and
  validates its checkpoint against the original encoder ONNX on zero/seeded
  random inputs. These are startup CPU checks, not runtime fallback.
- At inference, normalized RGB tensors are converted back to exact uint8
  pixels, packed as raw float32 patches, sent to the compiled encoder, and its
  float32 memory is passed to the original decoder on `--device`.
- `--device cpu/cuda` selects the PyTorch DECODER device in this entry point.
  Encoder execution is DX-M1, with any compiler CPU partitions listed in the
  manifest. The known successful full run has none. Detector remains ORT CPU.
- If NPU inference fails, the adapter raises an error; it does not substitute
  a PyTorch encoder. Existing pipeline error reporting remains active.

The original `alpr_auto.py` is not edited. The subclass registration is scoped
to this CLI invocation and restored in a finally block, along with the original
encoder method. This entry point is intended for the current sequential CLI,
not concurrent application/server use.

## First automatic VN test

```bash
cd "$HOME/OneKiwi/renesas-alpr"
bash run_vn_dxm1_smoke.sh
```

Runner:

1. Downloads `test_image/3.jpg` from the pinned public repository below, verifies
   SHA256 and caches it under `inputs/vn_dxm1_smoke`. It does not overwrite an
   existing file whose checksum differs. The image is a full scene with plate
   `30A-339.18`, visually checked for this single-image test.
2. Runs the original CPU baseline and the DX-M1 entry point on the same image.
3. Saves each pipeline's CSV/preview/summary under a fresh
   `results/vn_dxm1_smoke_<timestamp>_<pid>/{cpu,dxm1}` directory.
4. Writes `smoke_comparison.json` with detection box/text agreement, observed
   plate texts and whether the expected single-image text `30A33918` was seen.

Exit 2 means an observed mismatch, OCR error, or missing expected plate. It is
not a generic hardware-failure diagnosis. If both systems miss every plate,
the expected-text check prevents treating their empty outputs as a pass.

Sample provenance:

- Source: https://github.com/trungdinh22/License-Plate-Recognition/blob/0b08d1ad9d3da5b0569243a77f068138678f516b/test_image/3.jpg
- SHA256: `8345ddb7c0124ad6e823f5ea77544af44bd4fe7636f358035ad53c10d2c1afbd`
- Only the download recipe is included in this package, not the image or dataset.
- This is a reproducible demo case, not representative VN test data.

## Other photos/videos

Use an existing file or directory path on Ubuntu (quote paths with spaces):

```bash
.venv-linux/bin/python alpr_auto_dxm1.py \
  --dx-run artifacts/dxm1_full_encoder_mm_20261101_134233_21170 \
  --source /actual/path/to/video.mp4 \
  --device cpu \
  --max-frames 30
```

Or run CPU/NPU comparison with a custom source as argument 1, optional compiled
run as argument 2:

```bash
bash run_vn_dxm1_smoke.sh /actual/path/to/video.mp4
```

These examples require a real input file. The default runner above downloads
its own verified image and needs no manually created test_video.mp4.
For video, existing `--every-n-frames` defaults to 3 and `--max-frames 30` means
30 processed frames, not the first 30 raw frames. Existing timestamps are
nominal-FPS based; preview is sampled and silent. Event counts are not unique
vehicle counts. The comparison checks detection rows/boxes/text, not tracker
identity quality or all event semantics.

## Output provenance and limits

`summary.json` adds `ocr_encoder_backend`, `ocr_decoder_device`, and
`deepx_encoder`: model hashes, placement, runtime IO, successful/failed encoder
calls. Call counts include warmup. The legacy `ocr_device` field describes the
PyTorch decoder; the report says this explicitly. `accuracy` stays null without
ground-truth labels. Confidence and CPU/NPU agreement are not accuracy.

Original CSV schemas remain unchanged: `detections.csv`, `events.csv`,
`frames.csv`. Automatic detection/crop/line handling can fail independently of
encoder conversion. Test the full pipeline on representative labeled VN images
and videos before calling it customer-ready. No end-to-end FPS claim is made.

## Checks done while creating this package

```bash
.venv-linux/bin/python -m unittest discover -s tests -p test_alpr_dxm1_adapter.py -v
```

Four tests passed: full byte-range inverse preprocessing, rejection of invalid
transforms, inherited reader/cleanup on failure, entry-point CSV preservation and
backend provenance annotation. These use stubs for the baseline/model/runtime;
they are not an actual detector/video/NPU end-to-end test. The user subsequently
ran the real card on one VN photo and 30 sampled VN video frames successfully;
see docs/DXM1_DEPLOYMENT.md for the measured milestone and evidence collection.
