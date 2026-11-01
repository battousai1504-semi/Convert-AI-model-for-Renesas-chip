# Trạng thái dự án

Cập nhật: **2026-11-01**, theo log thực thi của Trực trên card thật.

Mục tiêu: nhận diện biển số Việt Nam một/hai dòng từ ảnh toàn cảnh và video,
xuất CSV; bước hiện tại là DEEPX DX-M1 M.2 trên Ubuntu, sau đó mới tích hợp Renesas.

| Hạng mục | Đã xác minh | Giới hạn |
| --- | --- | --- |
| Baseline Windows | CPU/CUDA OCR, detector tự động, ảnh/video, CSV | Chưa có benchmark VN đại diện |
| Ubuntu / phần cứng | Kernel 7.0.0-38, Secure Boot bật, driver ký MOK nạp được, PCIe Gen3 x4 | Driver cài thủ công theo kernel |
| SDK | DX-COM 2.4.1; DX-RT 3.4.2; RT driver 2.6.0; PCIe driver 2.5.0; FW card 2.7.0 | Không flash firmware |
| Encoder ONNX FP32 | Khớp PyTorch 6/6 ảnh demo | Chưa phải triển khai NPU |
| Encoder DX-M1 | Đầu raw-patch MatMul tương đương; 1 NPU group, 0 CPU groups | CPU vẫn resize/pack patch; INT8 có sai lệch feature |
| OCR ghép encoder NPU + decoder CPU | 6/6 text giống PyTorch; không lỗi thực thi | Cùng 6 ảnh dùng calibration; không phải holdout |
| ALPR ảnh toàn cảnh VN | Tự detect/cắt/đọc 30A33918, 1 detection / 1 event, giống CPU | Một ảnh smoke test |
| ALPR video VN | 30 frame đã xử lý, 30 detections, 1 event; 59V184625 ở cả CPU/DX-M1 | Frame lấy mẫu, không phải 30 biển độc lập |
| Renesas / sản phẩm khách hàng | Chưa xác minh | Cần tập dữ liệu gán nhãn và benchmark toàn pipeline |

## Mô hình thực sự đang chạy ở đâu?

- Detector: ONNX Runtime CPU.
- Resize, crop, tách dòng, pack patch: CPU.
- Encoder PARSeq đã biến đổi tương đương: DX-M1, một NPU group trong run thành công.
- Decoder PARSeq giữ nguyên autoregression/refinement: PyTorch CPU trong các log đã xác minh.
- Gộp sự kiện, preview, CSV: CPU.

`--device` của `alpr_auto_dxm1.py` chọn thiết bị **decoder**, không chọn encoder.

## Bằng chứng và khả năng chạy lại

Xem [DXM1_DEPLOYMENT.md](DXM1_DEPLOYMENT.md) cho đường dẫn run, lệnh và nguồn mẫu.
`scripts/collect_dxm1_evidence.py` tạo `docs/evidence/dxm1_20261101.json` từ báo cáo
thực tế trên laptop: kiểm tra hash model xuyên suốt các lần thử, đối chiếu lại CSV,
lưu số lượng và provenance. File chỉ được tạo khi báo cáo có đủ và nhất quán.
Model, checkpoint, video, SDK và môi trường Python không đưa vào Git.

Helper tests dùng stub/ONNX trên máy phát triển tách biệt với log test card thật.
Kết quả chạy tests của laptop được lưu khi chuẩn bị checkpoint Git; không coi tests
stub là bằng chứng thực thi NPU.

## Điều đã sửa trong quá trình chuyển model

Đường RGB/patch Conv ban đầu compile được nhưng text sai 6/6. Audit với cùng binary,
input và ORT bật ở cả Python/C++ cho thấy đầu ra giống nhau giữa hai API, nhưng khác
ONNX. Thử đầu embedding đã thu hẹp vấn đề về đường đầu encoder. Pack patch trên CPU
và đổi projection sang MatMul với normalization gộp vào weights/bias cho kết quả tốt.
Đây là workaround đã đo; chưa chứng minh nguyên nhân gốc thuộc compiler, runtime
hay firmware. Giữ các script probe/audit để tái hiện, không dùng run RGB cũ cho ALPR.

## Việc tiếp theo

1. Giữ checkpoint code, Dockerfile và báo cáo nhỏ để các thành viên chạy lại.
2. Xây tập đánh giá VN có nhãn đúng: xe máy/ô tô, hai dòng, góc nghiêng, đêm, mờ,
   nhiều xe; tách calibration/training/test.
3. Đo độ đúng cả biển và lỗi detector/OCR riêng; kiểm tra sự kiện/tracking trên video.
4. Đo thời gian từng bước và toàn pipeline trong điều kiện đã ghi rõ; chưa công bố FPS thực tế.
5. Sau khi có baseline đáng tin, tối ưu và tích hợp với máy chủ Renesas của công ty.

[Setup Windows/Ubuntu](SETUP_DUAL_BOOT.md) · [Baseline tự động](ALPR_AUTO_WINDOWS.md) ·
[Chuẩn bị Ubuntu](DEEPX_UBUNTU_PREP.md)
