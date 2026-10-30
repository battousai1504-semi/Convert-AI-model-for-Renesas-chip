# Đo OCR biển Việt Nam trên Windows

Mục tiêu của bước này là có ảnh thật, nhãn do người kiểm tra và kết quả CSV.
Công cụ không tự tìm biển số, không tự chia dòng và chưa chạy trên DEEPX.
Nó đọc được ảnh toàn cảnh/video để người dùng chọn vùng chữ bằng tay.

## 1. Chuẩn bị

Mở PowerShell trong repo `renesas-alpr`, ở nhánh `truc/parseq-cross-platform`.
Môi trường `.venv-win` hiện tại đã chạy PARSeq trên GPU thành công.

```powershell
.\.venv-win\Scripts\python.exe -m pip install opencv-python==4.9.0.80
.\.venv-win\Scripts\python.exe -m unittest discover -s tests -v
New-Item -ItemType Directory -Force inputs\vn_sources
```

Chép một số ảnh/video Việt Nam mà nhóm có quyền sử dụng vào `inputs/vn_sources`.
Đợt đầu nên chọn 10–20 biển đọc rõ, có cả biển dài một dòng và biển hai dòng.
Đây là số mẫu để kiểm tra quy trình, không đủ chứng minh chất lượng sản phẩm.
Không cần các ảnh đã cắt sẵn: có thể dùng ảnh toàn xe, hoặc video ngắn.
Các file `AD799KB` trước đây chưa thay thế được bộ dữ liệu Việt Nam này.

## 2. Cắt và gán nhãn từng dòng

```powershell
.\.venv-win\Scripts\python.exe annotate_plates.py --source inputs/vn_sources --output data/vn_ocr_v1 --every-seconds 2 --max-frames 20
```

Tại mỗi ảnh/khung hình, nhập lựa chọn ở terminal:

- `1`: biển một dòng; `2`: biển hai dòng.
- Cửa sổ ảnh mở ra: kéo chuột khoanh sát MỘT dòng chữ, giữ đủ nét các ký tự.
- Nhấn Enter để xác nhận; C để hủy vùng đang chọn. Quay lại terminal.
- Nhập đúng chữ nhìn thấy, giữ cả số 0 đầu dòng. Enter rỗng hủy cả biển.
- Với biển hai dòng: chọn và nhập dòng TRÊN trước, rồi dòng DƯỚI.
- Có thể chọn nhiều biển trong cùng ảnh. Nhập `n` khi muốn sang ảnh/khung tiếp.
- `q` thoát; Ctrl+C cũng giữ phần đã lưu. Chỉ chạy một phiên gán nhãn vào cùng một thư mục dữ liệu.

Nếu không đọc chắc ký tự thì bỏ qua biển đó. Không lấy dự đoán của model làm nhãn.
Không chọn logo/ốc vít; chưa dùng ảnh nghiêng nhiều cần nắn phối cảnh ở bước đầu.
Video mặc định lấy tối đa 20 khung hình cho mỗi file, cách khoảng 2 giây theo FPS
khai báo. Timestamp chỉ gần đúng, không dùng cho đo thời gian sự kiện chính xác.

Kết quả trong `data/vn_ocr_v1`:

| File | Nội dung |
| --- | --- |
| `images/*.png` | Mỗi ảnh là một dòng, được lưu ở độ phân giải gốc |
| `labels.csv` | `image,text`, dùng trực tiếp với `ocr_batch.py` |
| `dataset.json` | Các dòng thuộc biển nào, thứ tự, vùng cắt, nguồn và hash |

Ảnh nguồn không bị sửa. Crop có pixel trùng hệt sẽ bị từ chối để tránh đếm lặp.
Các khung khác nhau của cùng video vẫn có thể là cùng một biển: không coi chúng
là các quan sát độc lập. `group_id` mặc định là SHA256 file nguồn; cùng video
luôn cùng nhóm. Hai video/ảnh khác nhau của cùng xe vẫn cần nhóm kiểm tra và
gộp theo lượt quay/xe khi chia dữ liệu về sau. Chưa tự chia train/calibration/test.

Muốn sửa nhãn: mở `data/vn_ocr_v1/dataset.json` trong VS Code, sửa trường `text`
ở đúng dòng, giữ nguyên tên ảnh/hash/vùng cắt. Sau đó tạo lại CSV:

```powershell
.\.venv-win\Scripts\python.exe annotate_plates.py --output data/vn_ocr_v1 --export-only
```

Không chỉ sửa `labels.csv`: JSON là nguồn nhãn gốc. Sau khi sửa nhãn, chạy lại OCR
và evaluator. Công cụ kiểm tra pixel crop khi mở lại bộ dữ liệu và trước đánh giá.

## 3. Chạy OCR thật và xuất CSV

```powershell
.\.venv-win\Scripts\python.exe ocr_batch.py --input data/vn_ocr_v1/images --labels data/vn_ocr_v1/labels.csv --device both --output results/vn_ocr_01
```

Kết quả: `ocr_cpu.csv`, `ocr_cuda.csv`, `summary.json`. Accuracy trong summary này
là độ chính xác **từng dòng crop**; chưa phải cả biển hai dòng. Không tự đổi O↔0,
I↔1 để ép kết quả hợp định dạng. CPU/GPU giống nhau không chứng minh đọc đúng.
Giữ file CSV nhãn như văn bản; Excel tự nhận dạng có thể làm mất số 0 đầu chuỗi.

## 4. Đo chính xác cả biển

```powershell
.\.venv-win\Scripts\python.exe evaluate_plate_groups.py --dataset data/vn_ocr_v1/dataset.json --predictions results/vn_ocr_01/ocr_cuda.csv --output results/vn_plate_cuda_01
```

`plate_evaluation.csv` có một dòng cho mỗi biển đã gán nhãn; `plate_summary.json`
báo exact accuracy/CER chung và riêng biển một/hai dòng. Dòng trên ghép trước
dòng dưới, chỉ bỏ dấu phân cách và viết hoa. Cả hai dòng phải đọc thành công
và toàn bộ ký tự phải đúng thì biển đó mới tính là đúng. Dòng lỗi không được bỏ
khỏi mẫu số. CSV thiếu/trùng ảnh hoặc nhãn lệch bộ dữ liệu sẽ bị từ chối.

Các số đo này chỉ phản ánh **OCR với vùng cắt do người chọn**. Không bao gồm
detector, tự tách hai dòng, tracking, khử trùng sự kiện hay tốc độ toàn video.
Khi đọc báo cáo cần xem số nguồn/xe độc lập, không chỉ tổng số crop.

## Bước sau dữ liệu đầu tiên

1. Xem các biển sai trong CSV để phân biệt crop/nhãn sai và OCR sai.
2. Mở rộng dữ liệu có kiểm soát: một/hai dòng, ngày/đêm, mờ/xa/nghiêng.
3. Chọn một tập hiệu chuẩn và một tập đánh giá tách nguồn/xe; không dùng chung.
4. Dùng ảnh thật để kiểm tra encoder ONNX; thử DX-COM tại lab/Linux sau khi xác
   nhận phiên bản compiler/runtime/firmware. Decoder hiện vẫn chạy PyTorch.
5. Bổ sung detector và tự chia dòng, rồi video/tracking dựa trên cùng thước đo.

Không commit ảnh/video/dataset/kết quả/model vào Git. Các thư mục mặc định đã
được ignore. Chỉ commit code, tests và tài liệu của công cụ.
