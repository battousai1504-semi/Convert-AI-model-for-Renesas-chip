# Encoder đầy đủ với MatMul head + decoder PyTorch

## Mốc đã đo trên máy thật

Embedding prefix dùng MatMul đạt cosine 0.999865–0.999979 trên 6 ảnh demo.
Compiler 2.4.1 xác nhận projection ở NPU, 1 NPU group / 0 CPU group.
Đầu vào thực tế là float32 [1,128,96]; đầu ra prefix float32 [1,128,384].
Đây chỉ là kết quả prefix. Nó chưa xác nhận encoder đầy đủ hoặc OCR đã đúng.

## Lần chạy này

```bash
cd "$HOME/OneKiwi/renesas-alpr"
bash run_dxm1_full_encoder_probe.sh
```

Runner dùng encoder đầy đủ gốc:
`artifacts/encoder_20261031_225454_044655/parseq_encoder.onnx`.
Nó thay head bằng chính phép MatMul tương đương đã thử, giữ phần Transformer
phía sau, so sánh CPU ONNX gốc/mới trên 6 ảnh trước khi compile. Mỗi lần chạy
tạo thư mục `artifacts/dxm1_full_encoder_mm_<timestamp>_<pid>` mới.
Compilation summary xác nhận phép chiếu nằm ở NPU; số nhóm CPU/NPU của tail
được lưu trong manifest và in ra để phát hiện fallback.

Sau đó script host chạy:

1. Kiểm tra SHA256 model và source encoder thật có attention (từ chối prefix).
2. Kiểm tra dtype/shape/số byte runtime, bật ORT rõ ràng cho CPU fallback nếu có.
3. Kiểm tra tiền xử lý khớp transform của repo.
4. Kiểm tra source ONNX khớp encoder của checkpoint PyTorch trên từng ảnh.
5. Chạy toàn model PyTorch lấy text tham chiếu.
6. CPU xếp patch RGB, compiled encoder chạy trên DX-M1/CPU theo partition;
   giữ đặc trưng float32 trả về.
7. Chạy decoder PyTorch gốc với đặc trưng đó; giữ AR early stopping, max length,
   refinement và tokenizer. Chỉ thay encoder call, luôn khôi phục sau đó.
8. Xuất `comparison.csv`, `summary.json`, `runtime_io.json` ở
   `results/dxm1_full_ocr_<timestamp>`.

Exit 0: mọi text khớp trên tập vừa chạy; exit 2: có mismatch hoặc lỗi ảnh.
Không chấm feature INT8 bằng ngưỡng allclose float32: cosine, relative L2,
MAE, max-absolute-error được giữ trong CSV. Thời gian dxrt_call_ms gồm runtime
và truyền dữ liệu, không phải FPS cả pipeline hay benchmark đã warmup.

## Giới hạn

- NPU chạy compiled encoder graph, CPU vẫn resize/xếp patch và chạy decoder.
  Nếu compiler sinh CPU groups, ORT chạy chúng; phải xem partition trước khi
  gọi encoder là hoàn toàn NPU.
- 6 ảnh demo đang dùng cho cả calibration và smoke comparison. Kết quả này
  không phải holdout/test set có nhãn, không đo độ chính xác biển số Việt Nam.
- Text khớp PyTorch đo độ trung thực chuyển đổi, không chứng minh text đúng ảnh.
- Chưa tích hợp backend mới vào `alpr_auto.py` hoặc video.

## Kiểm chứng helper

```bash
.venv-linux/bin/python -m unittest discover -s tests -p test_dxm1_full_encoder_helpers.py -v
```

Trong môi trường tạo gói đã chạy 4 test: thay head vẫn bảo toàn phép tính của
tail có attention và LayerNorm; từ chối prefix/hash đổi; khôi phục encoder khi
decoder thành công hoặc thất bại; từ chối đầu vào uint8 ngoài contract.
Test decoder dùng stub nhỏ, không phải PyTorch/PARSeq đầy đủ. Test số học ONNX
dùng graph có attention/LayerNorm, không phải toàn encoder 12 block của người dùng.
Kết quả full PARSeq/DX-M1 phải lấy từ lần chạy trên máy thật.
