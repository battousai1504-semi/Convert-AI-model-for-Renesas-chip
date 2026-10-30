# Phạm vi và tiến độ

Cập nhật: 2026-10-30 (Asia/Ho_Chi_Minh).

## Mục tiêu đã thống nhất trong nhóm làm việc

- Ưu tiên biển số Việt Nam: ảnh crop, ảnh toàn cảnh, rồi video toàn cảnh.
- Cần xử lý biển một dòng và hai dòng ở giai đoạn ứng dụng đầy đủ.
- Đầu ra ban đầu là CSV; giữ thiết kế để có thể bổ sung giao diện khác.
- Model nhận dạng ban đầu là PARSeq pretrained. Bổ sung detector để tìm biển trong toàn cảnh.
- Video cần tracking/tổng hợp nhiều lần đọc và hạn chế dòng CSV trùng theo lượt xuất hiện.
- Phần cứng thử trước là card DEEPX M.2, rồi tích hợp Renesas theo kiến trúc công ty xác nhận.
- Nhận diện quốc gia/mở rộng quốc tế là hướng sau, không thuộc baseline đang có.
- Độ chính xác/tốc độ cần được đo trên dữ liệu đại diện; chưa có ngưỡng nghiệm thu được công ty xác nhận.

## Những gì hiện có

| Thành phần | Trạng thái và giới hạn |
| --- | --- |
| PARSeq gốc | Đã chạy CPU và RTX 4050 trên Windows trong môi trường Python 3.10 của Trực |
| `ocr_batch.py` | Đã chạy trên máy Trực, xuất CSV CPU/GPU và summary; chỉ nhận crop một dòng |
| `export_parseq_encoder.py` | Máy Trực báo PASS trên 2 file cùng cho chuỗi `AD799KB`; điều này không xác minh nhãn thật |
| Đối chiếu encoder | Đã kiểm tra trên CPU Linux với PyTorch 2.2.1/timm 0.9.16 và 6 ảnh demo upstream: đặc trưng và output nhận dạng khớp trong ngưỡng kiểm tra |
| Bootstrap hai OS | Windows bootstrap CUDA đã được Trực xác nhận: 5/5 tests, demo Chevrol trên RTX 4050. Linux CPU đã cài mới bằng `scripts/setup.sh cpu`, Python 3.10.22, `pip check` sạch; batch thật và encoder đối chiếu PASS 6/6. Linux CUDA trên laptop chưa kiểm tra |
| Unit tests | 10 tests vượt qua trên Python 3.10: các test baseline và kiểm tra nhãn/crop hai dòng, đổi tọa độ, hủy/lỗi ghi file, ảnh trùng/bị sửa, lấy khung hình video, ghép cả biển và nhãn lệch |
| Accuracy biển VN | Chưa có kết quả nghiệm thu trên tập nhãn đại diện |
| Detector / tách dòng / nhận dạng video tự động | Chưa triển khai |
| Chuẩn bị dữ liệu VN | Thêm `annotate_plates.py`: chọn vùng và nhập nhãn bằng tay từ ảnh/video lấy mẫu, giữ nhóm nguồn và thứ tự hai dòng. `evaluate_plate_groups.py` đánh giá cả biển từ CSV OCR. Chưa chạy GUI trên máy Windows đích; chưa có số liệu biển VN thật |
| DX-COM / `.dxnn` / card thật | Chưa xác minh với model của dự án |
| Tích hợp Renesas | Chưa thực hiện |

## Thứ tự tiếp theo

1. Chạy lại baseline/demo từ repo nhóm trên Windows và Ubuntu; ghi phiên bản cùng log.
2. Xác nhận card, phiên bản DEEPX compiler/runtime/firmware ở lab; chạy model mẫu của hãng.
3. Thử compiler với encoder ONNX để phát hiện phép toán chưa hỗ trợ và đánh giá lượng tử hóa bằng dữ liệu hiệu chuẩn phù hợp. Dữ liệu hiệu chuẩn và bộ đánh giá độc lập.
4. Quyết định cách triển khai decoder dựa trên kết quả compiler và số liệu thực tế; hiện vẫn là PyTorch, chưa chốt encoder-only làm kiến trúc sản phẩm.
5. Bổ sung detector biển VN, xử lý bố cục hai dòng và đánh giá ảnh toàn cảnh.
6. Bổ sung video/tracking, CSV sự kiện và đo hiệu năng toàn luồng.

Một file ONNX được tạo thành công không đồng nghĩa chuyển đổi đúng. `PASS` đối chiếu không đồng nghĩa OCR đúng nhãn; chạy trên PC không đồng nghĩa đã chạy trên NPU. Mỗi mốc phải có bằng chứng riêng.

Luồng dữ liệu mới đã được thử trên ảnh chữ tổng hợp có nhãn biết trước với PARSeq
CPU thật: cắt/lưu nhãn → OCR CSV → ghép cả biển. Video lấy mẫu đã thử với file
MJPG tạo trong test. Đây là kiểm tra chức năng, không phải accuracy biển VN thực tế.
