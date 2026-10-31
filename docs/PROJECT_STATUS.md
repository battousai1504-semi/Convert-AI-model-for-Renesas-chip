# Phạm vi và tiến độ

Cập nhật: 2026-10-31 (Asia/Ho_Chi_Minh).

Trang này tổng hợp mã trên nhánh `truc/parseq-cross-platform` và log Trực cung cấp. Các kết quả trên laptop không phải lần chạy lại trong CI.

## Mục tiêu

- Ưu tiên biển Việt Nam, một dòng và hai dòng.
- Nhận ảnh crop, ảnh toàn cảnh và video toàn cảnh; tự tìm biển và đọc chữ.
- Đầu ra ban đầu là CSV; video tổng hợp nhiều lần đọc theo lượt xuất hiện.
- Chạy thử trên DEEPX DX-M1 M.2, sau đó tích hợp Renesas theo kiến trúc công ty xác nhận.
- Mở rộng quốc tế/nhận diện quốc gia ở giai đoạn sau.
- Chưa có ngưỡng nghiệm thu accuracy hoặc tốc độ được công ty xác nhận.

## Thành phần và bằng chứng

| Thành phần | Kết quả hiện có | Giới hạn |
| --- | --- | --- |
| PARSeq trên Windows | CPU và RTX 4050 CUDA chạy được | Chạy được không chứng minh OCR đúng |
| PARSeq trên Ubuntu | Python 3.10, CPU; demo `coco-1166773.jpg` trả `Chevrol` | Chưa có xác nhận chạy CUDA trên Ubuntu laptop |
| `ocr_batch.py` | CSV CPU/CUDA và báo cáo so sánh | Đầu vào là crop một dòng |
| Encoder ONNX | Trên laptop Ubuntu, PASS 6/6 ảnh demo so với PyTorch; input FP32 `[1,3,32,128]`, opset 17 | Chỉ encoder; decoder vẫn PyTorch. Đây là độ khớp chuyển đổi, không phải accuracy |
| `alpr_auto.py` | Tự detect, xử lý một/hai dòng, OCR, tổng hợp track và xuất CSV | Detector ONNX Runtime CPU; OCR PyTorch CPU/CUDA; chưa có backend DX-RT |
| Ảnh toàn cảnh VN trên Windows | 6 ảnh, 14 detections, 14 events | Chưa đối chiếu toàn bộ với nhãn chuẩn |
| Video trên Windows | `test_1.mp4`: 30 khung được xử lý, 47 detections, 10 events | Không suy ra 10 xe duy nhất, accuracy hoặc FPS toàn luồng |
| Dữ liệu/đánh giá | `annotate_plates.py`, `evaluate_plate_groups.py` | Khoanh vùng/nhập nhãn chỉ phục vụ tạo ground truth; không bắt buộc khi dùng ALPR tự động |
| Unit tests | Trực đã gửi log 5/5 baseline tests OK trên Windows | Không ghi nhận tổng số tests khác hay kết quả Linux khi chưa có log mới |
| DX-COM | 2.4.1, CLI báo target M1 | Chưa compile encoder PARSeq thành `.dxnn` |
| DX-RT | 3.4.2, `dxrt-cli -h` hoạt động, `import dx_engine` OK, `pip check` sạch | Chưa gọi inference trên card |
| Driver trên Ubuntu | Build và cài `dx_dma` + `dxrt_driver`; signer là MOK đã enroll; vermagic `7.0.0-38-generic` | Chưa xác minh nạp module, PCIe hoặc giao tiếp phần cứng |
| Model mẫu | MobileNetV2 `.dxnn` đã tải từ DEEPX, khoảng 7.4 MiB | Chưa chạy trên DX-M1 |
| Renesas | Chưa tích hợp | Cần xác nhận kiến trúc đích |

Log export trên laptop Ubuntu: `artifacts/encoder_20261031_225454_044655`.
Các output mẫu: CHEWBACCA, Chevrol, SALMON, Verbandsteffe, Kappa, 3rdAve.
Giữ nguyên output gốc trong báo cáo; không dùng chúng làm ground truth.

## Mốc tiếp theo

1. Gắn DX-M1, xác nhận PCIe, nạp driver có chữ ký và đọc trạng thái runtime.
2. Chạy model mẫu hãng để kiểm tra inference phần cứng.
3. Thử compile encoder PARSeq bằng DX-COM; lưu log phép toán, cấu hình và phiên bản.
4. Chuẩn bị dữ liệu calibration đại diện, tách biệt bộ đánh giá; đo sai lệch sau lượng tử hóa.
5. Tích hợp DX-RT vào ALPR theo kết quả compiler; đánh giá cả detector và encoder. Decoder hiện vẫn PyTorch.
6. Đo accuracy cả biển và hiệu năng toàn luồng trên dữ liệu VN có nhãn; đánh giá lỗi tracking, biển nghiêng/mờ, hai dòng.
7. Tích hợp Renesas sau khi thống nhất phần cứng và giao diện.

Hướng dẫn: [hai OS](SETUP_DUAL_BOOT.md), [ALPR tự động](ALPR_AUTO_WINDOWS.md), [chuẩn bị DX-M1 trên Ubuntu](DEEPX_UBUNTU_PREP.md).

ONNX hợp lệ không đồng nghĩa compiler hỗ trợ. PASS đối chiếu không đồng nghĩa OCR đúng. Driver build/cài được không đồng nghĩa đã nạp được hoặc đã chạy trên NPU.
