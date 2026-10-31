# Học và tự debug ALPR trên DEEPX DX-M1

Tài liệu dành cho nhóm intern OneKiwi: học từ các việc đang làm với PARSeq, ảnh/video biển Việt Nam, ONNX và card DX-M1. Nội dung bằng tiếng Việt; giữ thuật ngữ tiếng Anh để tra tài liệu và đọc log.

Mục tiêu là tự giải thích pipeline, tái hiện lỗi, kiểm tra giả thuyết và viết bản vá có bằng chứng. Hoàn thành lộ trình không đồng nghĩa đã thành chuyên gia; cần tiếp tục thực hành với dữ liệu và lỗi thật.

## Bắt đầu ở đâu?

1. Đọc [bản đồ dự án](00_PROJECT_MAP.md) để biết từng thành phần nhận và trả gì.
2. Theo [lộ trình 8 chặng](ROADMAP.md), chọn chặng bằng năng lực hiện tại.
3. Làm [bài thực hành](labs/README.md), ghi kết quả bằng [nhật ký học](templates/LEARNING_LOG.md).
4. Khi gặp lỗi, dùng [quy trình debug](08_DEBUGGING.md), không đổi nhiều thứ cùng lúc.
5. Tra [tài liệu gốc](RESOURCES.md) theo mã R01–R17. Mỗi nguồn có phần cần đọc, mục đích và ưu tiên.

## Các bài học

| Bài | Chủ đề | Năng lực cần chứng minh |
| --- | --- | --- |
| [00](00_PROJECT_MAP.md) | Bản đồ hệ thống | Chỉ đúng nơi có thể gây ra một lỗi |
| [01](01_ENV_GIT_PYTHON.md) | Shell, Git, Python, môi trường | Chạy đúng interpreter và đọc traceback |
| [02](02_ML_PYTORCH.md) | ML, tensor, inference và training | Giải thích shape/dtype/device, weights và logits |
| [03](03_DETECTION_GEOMETRY.md) | Detector, ảnh, tọa độ, hai dòng | Từ bbox vẽ đúng crop trên ảnh gốc |
| [04](04_PARSEQ_TRANSFORMER.md) | Attention, ViT, PARSeq, tokenizer | Theo được ảnh → encoder → decoder → chuỗi |
| [05](05_EVALUATION_VIDEO.md) | Nhãn, metrics, video, tracking | Đánh giá độc lập và hiểu giới hạn events |
| [06](06_ONNX_QUANTIZATION.md) | Export, fidelity, INT8, calibration | Phân biệt ONNX hợp lệ, output đúng và NPU hỗ trợ |
| [07](07_DEEPX_LINUX.md) | PCIe, driver, Secure Boot, DX-RT | Chẩn đoán từ PCIe tới inference |
| [08](08_DEBUGGING.md) | Quy trình tìm và vá lỗi | Đưa ra minimal reproduction và regression check |
| [09](09_OPTIMIZATION_DELIVERY.md) | Profiling, tối ưu, bàn giao | Cải thiện có đo và giữ được độ đúng |

## Quy ước chạy lệnh

Chạy từ root repo, sau khi đã setup theo [hướng dẫn hai OS](../docs/SETUP_DUAL_BOOT.md).

| OS | Interpreter ứng dụng |
| --- | --- |
| Ubuntu / Bash | `.venv-linux/bin/python` |
| Windows / PowerShell | `.\.venv-win\Scripts\python.exe` |

Các lệnh CLI trong bài ưu tiên Ubuntu. Trên Windows thay tiền tố interpreter theo bảng; các lệnh Linux như `lspci`, `modprobe`, `journalctl` chỉ dùng trên Ubuntu. Các khối `python` minh họa có thể chạy trong file thử riêng, bằng đúng interpreter ứng dụng.

Không nâng cấp dependencies chỉ vì tài liệu mới dùng phiên bản khác. Baseline hiện ghim Python 3.10 / PyTorch 2.2.1; môi trường SDK tách riêng. Đọc [trạng thái thực tế](../docs/PROJECT_STATUS.md) trước khi đưa ra tuyên bố về phần cứng.

## Cách cùng học

- Mỗi chặng có một người làm và một người chạy lại/giải thích chéo; đổi vai.
- Mỗi thành viên phải hiểu toàn luồng dù có phụ trách sâu một phần.
- Ghi ghi chú đã kiểm chứng vào `knowledge/notes/<ten-thanh-vien>/` khi có nội dung thật. Dùng mẫu [learning log](templates/LEARNING_LOG.md); không commit dữ liệu lớn.
- Thay đổi kiến thức bằng nhánh nhỏ và PR: nêu điều sửa, nguồn hoặc log, kết quả kiểm tra.
- Không đưa dự đoán AI hoặc lời giải thích chưa kiểm tra thành ground truth.
- Giữ logs/ảnh/model cục bộ trong thư mục đã ignore; chỉ commit ví dụ nhỏ được phép chia sẻ và kết quả tổng hợp đã kiểm tra.

## Việc nên học ngay với tiến độ hiện tại

1. [Bài 07](07_DEEPX_LINUX.md): vì sao build/cài driver chưa chứng minh card hoạt động.
2. [Bài 04](04_PARSEQ_TRANSFORMER.md): encoder đã chuyển ONNX, decoder còn PyTorch nghĩa là gì.
3. [Bài 08](08_DEBUGGING.md): luyện với lỗi file không tồn tại và lỗi patch nhóm đã gặp.

Nhánh kiến thức được xây trên commit `068fc2e` của nhánh ứng dụng. Các trang mô tả code ở mốc đó; khi code đổi, cần cập nhật bài học tương ứng.
