# Tài liệu gốc: đọc phần nào và dùng để làm gì?

Ưu tiên tài liệu của tác giả/thư viện/hãng. Các link cốt lõi được kiểm tra khi soạn ngày 2026-10-31. Tài liệu online có thể đổi; code dự án vẫn theo requirements và SHA của SDK trong docs.

**Cách đọc:** mở đúng mục được chỉ định → chạy một ví dụ → áp vào một hàm trong repo → ghi điều đã hiểu/chưa kiểm chứng. Không cần đọc hết mọi trang trước khi bắt đầu.

## R01

**[Python 3.10 Tutorial](https://docs.python.org/3.10/tutorial/)** — bắt buộc cho Bài 01.

Đọc control flow, data structures, modules, input/output, errors/exceptions, classes và virtual environments. Bắt đầu bằng dict/list/function/path/exception.

Đây là tài liệu đúng nhánh Python của baseline; Python 3.10 hiện là phiên bản cũ. Việc học/tái hiện baseline không phải lý do giữ nguyên mọi dependency cho sản phẩm mãi mãi; nâng cấp cần kế hoạch kiểm thử.

## R02

**[Pro Git](https://git-scm.com/book/en/v2)** — bắt buộc cho làm nhóm.

Đọc chương 2 Git Basics, chương 3 Git Branching. Tập trung working tree/index/commit/remote, xem diff, branch và merge. Đầu ra: biết file nào thực sự ở commit đang push.

## R03

**[MIT Missing Semester: Shell](https://missing.csail.mit.edu/2020/course-shell/)** — bắt buộc cho Linux.

Học working directory, PATH, quoting, redirection, pipes và exit status. Dùng để hiểu vì sao một khối lệnh lỗi giữa chừng vẫn có thể chạy lệnh phía sau nếu không kiểm soát.

## R04

**[uv: environments](https://docs.astral.sh/uv/pip/environments/)** và **[installation](https://docs.astral.sh/uv/getting-started/installation/)**.

Đọc chọn interpreter/venv và dùng Python-managed installation. Dự án hiện dùng requirements cùng pip trong venv, không tự chuyển sang một workflow lockfile khác khi chưa đánh giá.

## R05

**[NumPy 1.26 quickstart](https://numpy.org/doc/1.26/user/quickstart.html)** — Bài 02–03.

Đọc ndarray, shape, dtype, indexing, reshape, broadcasting và copies/views. Áp dụng vào ảnh, bbox, layout và buffer; dùng phiên bản1.26 theo baseline PC.

## R06

**[PyTorch Learn the Basics](https://docs.pytorch.org/tutorials/beginner/basics/intro.html)** — Bài 02.

Theo các phần tensor → datasets/dataloaders → transforms → build model → autograd → optimization → save/load. Tài liệu hiện tại có thể dùng API mới; giữ baseline torch 2.2.1 khi làm repo này, thử API mới trong môi trường riêng nếu cần.

## R07

**[OpenCV 4.9: geometric transforms](https://docs.opencv.org/4.9.0/da/d6e/tutorial_py_geometric_transformations.html)** — Bài 03.

Đọc resize, affine và perspective transforms; hiểu ma trận đầu vào, kích thước output và interpolation. Dùng để kiểm tra rectification và biến đổi tọa độ; không suy ra một heuristic chọn góc luôn đúng.

## R08

**[OpenCV 4.9: videos](https://docs.opencv.org/4.9.0/dd/d43/tutorial_py_video_display.html)** — Bài 05.

Đọc VideoCapture, VideoWriter, xử lý lỗi đọc frame, size và FPS. Liên hệ timestamp xấp xỉ trong repo; chưa phải thiết kế streaming nhiều camera.

## R09

**[open-image-models — tác giả detector](https://github.com/ankandrew/open-image-models)** — Bài 03.

Đọc phần object detection và model ONNX; kiểm tra preprocessing và output contract trước khi đổi model. Detector repo nhóm có URL/hash riêng trong alpr_auto.py. Giữ attribution/license khi dùng hoặc phân phối.

## R10

**[Dive into Deep Learning: Attention & Transformers](https://d2l.ai/chapter_attention-mechanisms-and-transformers/index.html)** — Bài 04.

Đọc attention scoring, scaled dot-product, multi-head, positional encoding, Transformer và Vision Transformer. Thiếu toán thì quay về chương 2: linear algebra, calculus, autodiff, probability. Chưa cần học toàn bộ NLP để hiểu encoder ảnh.

## R11

**[PARSeq paper — tác giả Bautista & Atienza](https://arxiv.org/abs/2207.06966)** và **[reference implementation](https://github.com/baudm/parseq)** — Bài 04.

Lượt1: abstract, hình kiến trúc, input/output, AR/NAR/refinement. Lượt2: permutation language modeling và công thức loss/masking. Lượt3: thí nghiệm, ablation và giới hạn dữ liệu.

Đối chiếu với strhub/models/parseq/model.py và configs/model/parseq.yaml của nhóm; không chỉ đọc paper rồi giả định code đang dùng mọi cấu hình trong đó.

## R12

**[ONNX introduction](https://onnx.ai/onnx/intro/)** — Bài 06.

Đọc Concepts, ONNX with Python, model inputs/outputs và operator/opset. Tập in graph metadata từ file thật trước khi tối ưu.

## R13

**[torch.onnx — PyTorch 2.2](https://docs.pytorch.org/docs/2.2/onnx.html)** — Bài 06.

Đọc export, tracing, shape/dynamic axes và unsupported operations. Khớp với phiên bản torch của repo; không trộn lệnh exporter mới nhất với baseline cũ mà không kiểm thử.

## R14

**[ONNX Runtime quantization](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)** — Bài 06.

Đọc quantization overview, static/dynamic, calibration và accuracy debugging. Dùng để học khái niệm; quy trình DX-COM cần cấu hình của DEEPX, không mặc định chạy ORT quantize rồi chuyển thẳng lên NPU.

## R15

**[DEEPX official tutorials](https://github.com/DEEPX-AI/dx-tutorials)** và **[SDK environment guide](https://github.com/DEEPX-AI/dx-all-suite/blob/main/docs/source/02_Setting_Up_Environment.md)** — Bài 06–07.

Thứ tự đề nghị trong tutorial: T01 môi trường → T02 model demo → T05 compiler/calibration → T06 runtime/Python/profiling. Đọc tên và phiên bản tutorial tại thời điểm sử dụng; không tự nâng SDK đang dùng chỉ để khớp một trang mới.

Repo nhóm có bản ghi [setup/phiên bản thật](../docs/DEEPX_UBUNTU_PREP.md). Khi debug, ưu tiên mã/tài liệu tại SHA đã cài thay vì main mới nhất.

## R16

**[Linux kernel: module signing](https://docs.kernel.org/admin-guide/module-signing.html)** — Bài 07.

Đọc ký module, public/private key, sign-file, việc không strip sau khi ký và kiểm tra khi nạp. Tập giải thích signer/vermagic và lỗi trust; không chia sẻ khóa riêng để “giúp debug”.

## R17

**[Ubuntu: UEFI Secure Boot](https://documentation.ubuntu.com/security/security-features/platform-protections/secure-boot/)** — Bài 07.

Đọc quan hệ firmware → shim → kernel/module, MOK và DKMS. Kết hợp R16 để phân biệt cert đã enroll với module thực sự được kernel chấp nhận.

## Học sâu nhưng chưa cần làm trước

Sau khi qua Lab 12, chọn một hướng ở Bài 09. Đọc source driver/runtime theo nhu cầu thực tế, đo một bottleneck hoặc thử một operator graph nhỏ. Chất lượng câu hỏi và phép đo quan trọng hơn số link đã lưu.
