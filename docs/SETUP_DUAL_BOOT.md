# Chạy cùng mã nguồn trên Windows và Ubuntu

## Nguyên tắc

- Dùng cùng nhánh Git cho mã nguồn, cấu hình và tài liệu.
- Mỗi OS tạo môi trường riêng: `.venv-win` hoặc `.venv-linux`. Không copy hoặc dùng chung `.venv` giữa hai OS.
- Driver NVIDIA, driver/runtime DEEPX và thư viện hệ thống được cài riêng trên từng OS.
- Ảnh, video, weights, ONNX, DXNN và kết quả chạy được lưu riêng, mặc định không commit.
- Khuyến nghị hai bản clone: Windows trong thư mục người dùng Windows; Ubuntu trong `~/OneKiwi`. Đồng bộ code bằng commit/push/pull.
- Nếu cần chuyển model ONNX giữa hai máy/OS, copy file cùng `export_report.json` để giữ thông tin kích thước, preprocessing và hash.

PC baseline dùng Python 3.10, PyTorch 2.2.1, NumPy 1.26.4 và các phiên bản ONNX đã ghim trong `requirements/pc-*.txt`. CUDA 12.1 là baseline cho RTX 4050 đã dùng trong dự án; cần kiểm tra riêng với các dòng GPU khác. Đây không phải môi trường cài DX-COM hay runtime DEEPX.

## 1. Lấy nhánh làm việc

Từ thư mục cha muốn chứa dự án, chạy trên mỗi OS:

```text
git clone --branch truc/parseq-cross-platform https://github.com/battousai1504-semi/Convert-AI-model-for-Renesas-chip.git renesas-alpr
cd renesas-alpr
```

Thư mục PARSeq cũ vẫn có thể dùng để đối chiếu. Chép các ảnh đang thử vào `renesas-alpr/inputs` khi cần; không chuyển `.venv` cũ sang đây.

## 2. Windows: cài môi trường

Cần Git và uv. Máy đã cài thì bỏ qua:

```powershell
winget install --id Git.Git -e
winget install --id astral-sh.uv -e
```

Sau khi cài mới, mở terminal mới để nhận PATH. Trong thư mục repo, chọn **một** cấu hình:

```powershell
# Máy có NVIDIA GPU phù hợp, muốn chạy CUDA
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/setup.ps1 -Device cuda

# Hoặc chỉ chạy CPU
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/setup.ps1 -Device cpu
```

`ExecutionPolicy Bypass` chỉ áp dụng cho tiến trình PowerShell vừa gọi. Script tạo `.venv-win`, cài dependencies và chạy `pip check`. Khi chọn CUDA, cần thấy `CUDA available: True` trước khi chạy tác vụ GPU. Có thể chạy lại setup để đổi CPU/CUDA trong môi trường này.

Chạy các lệnh từ terminal PowerShell của VS Code:

```powershell
.\.venv-win\Scripts\python.exe read.py pretrained=parseq --images demo_images/coco-1166773.jpg --device cpu
.\.venv-win\Scripts\python.exe read.py pretrained=parseq --images demo_images/coco-1166773.jpg --device cuda
.\.venv-win\Scripts\python.exe ocr_batch.py --input demo_images --device both
.\.venv-win\Scripts\python.exe export_parseq_encoder.py --input demo_images
```

Với máy chỉ có CPU, dùng `--device cpu` thay cho `both` và bỏ lệnh CUDA.

## 3. Ubuntu: cài môi trường

Cần Git, curl và uv. Nếu chưa có Git/curl, cài bằng trình quản lý gói của OS. Cài uv theo hướng dẫn chính thức:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Mở terminal mới, kiểm tra `uv --version`, rồi vào thư mục repo. Chọn **một** cấu hình:

```bash
bash scripts/setup.sh cpu
# Hoặc, sau khi nvidia-smi hoạt động trên Ubuntu:
bash scripts/setup.sh cuda
```

Driver NVIDIA đã hoạt động trên Windows không đồng nghĩa Ubuntu đã có driver phù hợp. Script chỉ cài môi trường Python; nó không cài driver GPU.

```bash
.venv-linux/bin/python read.py pretrained=parseq --images demo_images/coco-1166773.jpg --device cpu
.venv-linux/bin/python read.py pretrained=parseq --images demo_images/coco-1166773.jpg --device cuda
.venv-linux/bin/python ocr_batch.py --input demo_images --device both
.venv-linux/bin/python export_parseq_encoder.py --input demo_images
```

Với máy chỉ có CPU, dùng `--device cpu` cho batch và bỏ lệnh CUDA.

## 4. Dùng VS Code trên cả hai OS

1. Mở **thư mục repo** bằng File > Open Folder hoặc `code .`.
2. Cài các extension được đề xuất: Python, Python Debugger và Pylance của Microsoft.
3. Chọn `Python: Select Interpreter` trong Command Palette:
   - Windows: `<repo>\.venv-win\Scripts\python.exe`.
   - Ubuntu: `<repo>/.venv-linux/bin/python`.
4. Mở Run and Debug, chọn cấu hình có sẵn rồi nhấn F5.

`launch.json` chọn đúng executable theo OS. Cấu hình GPU hoặc CPU/GPU chỉ chạy sau khi môi trường nhận CUDA. Terminal không tự kích hoạt venv; các lệnh trên gọi trực tiếp executable để tránh chạy nhầm Python. Các thư mục venv được ẩn ở Explorer để tránh mở nhầm `python.exe` thành văn bản.

## 5. Chạy ảnh biển số của nhóm

```text
inputs/
  plate_01.png
  plate_02.jpg
```

Mỗi ảnh trong baseline hiện tại cần là **một dòng chữ đã cắt**, chưa phải ảnh toàn xe hoặc biển hai dòng chưa tách. Thay `--input demo_images` bằng `--input inputs`.

Nếu có nhãn đã xác minh, tạo CSV UTF-8 với tên tương đối theo thư mục đầu vào:

```csv
image,text
plate_01.png,59A12345
plate_02.jpg,30F67890
```

Các giá trị trên chỉ minh họa định dạng, không phải nhãn của ảnh có sẵn. Dùng `--labels <duong-dan-CSV>` với `ocr_batch.py` sau khi điền đáp án thật. Không dùng dự đoán model làm nhãn.

CSV lưu dự đoán nguyên gốc và phiên bản chuẩn hóa. Chuẩn hóa chỉ viết hoa và bỏ ký tự phân cách; không tự thay `O/0`, `B/8`, `I/1`.

## 6. Đọc kết quả

- Batch: thư mục mới trong `results/`, gồm `ocr_cpu.csv` và/hoặc `ocr_cuda.csv`, cùng `summary.json`.
- ONNX: thư mục mới trong `artifacts/`, gồm `parseq_encoder.onnx`, `encoder_validation.csv`, `export_report.json`.
- Không có nhãn thật thì accuracy/CER là `null`. CPU/GPU đồng ý không chứng minh OCR đúng.
- `model_ms` chỉ đo forward của model. `pipeline_ms` gồm đọc ảnh, preprocessing, chuyển dữ liệu, forward và giải mã; chưa phải FPS video.
- Encoder ONNX cố định batch 1, float32, đầu vào `[1,3,32,128]` ở cấu hình PARSeq mặc định. Preprocessing nằm ngoài graph.
- Export hiện chỉ chuyển **encoder**. Decoder vẫn chạy PyTorch, giữ vòng lặp AR và refinement. `PASS` là đối chiếu trên các ảnh đã chạy, không phải chứng nhận DEEPX hay accuracy sản phẩm.

## 7. Đồng bộ khi chuyển OS

Trước khi chuyển OS, lưu các file code rồi kiểm tra:

```text
git status
git add ocr_batch.py export_parseq_encoder.py docs
git commit -m "Describe the actual change"
git push
```

Chỉ add những đường dẫn đã sửa; nếu không có thay đổi thì không cần commit. Ở OS còn lại, vào bản clone của nó:

```text
git status
git pull --ff-only
```

Nếu có thay đổi chưa commit hoặc nhánh đã phân kỳ, xử lý chúng trước khi pull; không dùng reset/force để bỏ qua. Sau khi requirements thay đổi, chạy lại script setup của OS đó. Dữ liệu/model bị ignore không được Git tự đồng bộ.

## 8. Chuyển sang DEEPX

DX-COM là môi trường compiler riêng trên Linux x86_64; phải chọn phiên bản phù hợp runtime/firmware của card lab. DX-RT có tài liệu cho Windows và Linux, nhưng cần cài driver/runtime riêng và kiểm thử từng tổ hợp thực tế. Khả năng dùng chung `.dxnn` phụ thuộc tương thích SDK/runtime, không được suy ra chỉ từ việc dùng chung code Python.

Không cài DX-COM vào môi trường PARSeq này nếu các requirements của hãng khác bộ đã ghim. Không upload SDK, weights, ảnh/video khách hàng hoặc model nhị phân lớn vào Git thông thường.

Tài liệu chính thức:

- https://docs.astral.sh/uv/getting-started/installation/
- https://pytorch.org/get-started/previous-versions/
- https://developer.deepx.ai/tech-docs/
- https://github.com/DEEPX-AI/dx_rt

## 9. Kiểm tra code phụ trợ

```powershell
.\.venv-win\Scripts\python.exe -m unittest discover -s tests -v
```

```bash
.venv-linux/bin/python -m unittest discover -s tests -v
```

Unit tests kiểm tra CSV, nhãn, cách tính chỉ số và báo cáo. Đối chiếu model thật dùng lệnh export với `demo_images`; kiểm thử phần cứng DEEPX vẫn là mốc riêng.
