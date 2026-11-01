# Chạy tự động: ảnh/video → biển số → CSV

`alpr_auto.py` không yêu cầu khoanh vùng hoặc nhập biển số. Luồng thực tế:

1. YOLOv9 ONNX tìm các vùng biển trong ảnh/khung hình.
2. Cắt tự động, thử nắn theo viền tứ giác khi thấy viền phù hợp.
3. Chọn bố cục một/hai dòng bằng hình học; với hai dòng, tìm khoảng trống ngang.
4. PARSeq đọc từng dòng; ghép dòng trên trước, dòng dưới sau.
5. Lưu CSV và ảnh/video có khung + chữ dự đoán.
6. Video dùng tracker IoU đơn giản để gom các lần đọc và bỏ phiếu chuỗi phổ biến.

Đây là baseline chạy trên PC. Detector chạy ONNX Runtime CPU, còn `--device cuda`
chỉ chọn GPU cho PARSeq. Chưa biên dịch detector/encoder/decoder sang DEEPX,
chưa chạy card M.2 hoặc tích hợp Renesas. Đây chưa phải bản nghiệm thu sản phẩm.

## Cài phần bổ sung

Đứng trong `C:\Users\Admin\OneKiwi\renesas-alpr`, dùng môi trường hiện có:

```powershell
.\.venv-win\Scripts\python.exe -m pip install -r requirements/alpr-auto.txt
.\.venv-win\Scripts\python.exe -m unittest discover -s tests -v
```

Không cần cài thư viện OCR khác; vẫn dùng PARSeq. Các file mới này hoạt động độc
lập với công cụ gán nhãn `annotate_plates.py`, có thể dùng dù chưa cài gói gán nhãn.
Để chạy code trên Ubuntu về sau, thay đường dẫn Python bằng `.venv-linux/bin/python`.

## Chạy một ảnh toàn cảnh

```powershell
.\.venv-win\Scripts\python.exe alpr_auto.py --source inputs/vn_sources/3.jpg --device cuda
```

Lần đầu, chương trình tải detector khoảng 7.8 MB, kiểm tra SHA256 rồi giữ trong
`artifacts/models`. PARSeq dùng checkpoint đã có trong cache hoặc tự tải nếu thiếu.
Không cần gắn card DEEPX. Toàn bộ suy luận chạy trên PC.

Để chạy tất cả ảnh/video trong một thư mục:

```powershell
.\.venv-win\Scripts\python.exe alpr_auto.py --source inputs/vn_sources --device cuda
```

Mỗi lần chạy tạo thư mục mới `results/auto_<timestamp>`, được in ở terminal.
Không dùng ảnh gán nhãn/labels làm điều kiện để chạy suy luận.

## Chạy video

```powershell
.\.venv-win\Scripts\python.exe alpr_auto.py --source inputs/test_video.mp4 --device cuda --every-n-frames 3
```

Mặc định xử lý các khung 0, 3, 6, …; không phải mọi khung. Muốn xử lý mọi khung:

```powershell
.\.venv-win\Scripts\python.exe alpr_auto.py --source inputs/test_video.mp4 --device cuda --every-n-frames 1
```

Thử ngắn trước khi chạy video dài bằng `--max-frames 30` (30 khung được xử lý).
Preview MP4 là video không âm thanh gồm các khung được xử lý, FPS giảm tương ứng.
Nếu bộ mã hóa video trên máy không mở được, chương trình báo lỗi rõ ràng;
thêm `--no-preview` để chỉ xuất CSV. Không tự bỏ qua lỗi rồi báo đã chạy thành công.
Nhấn Ctrl+C khi cần dừng; báo cáo sẽ ghi `interrupted`, không giả thành chạy đủ clip.

## Đầu ra

| File | Ý nghĩa |
| --- | --- |
| `detections.csv` | Từng vùng được phát hiện: nguồn, khung, tọa độ, chuỗi thô/chuỗi chuẩn hóa, một/hai dòng, điểm số, track ID |
| `events.csv` | Một kết quả cho mỗi track: thời gian đầu/cuối, chuỗi phổ biến, số phiếu và số lần quan sát |
| `frames.csv` | Mỗi khung đã xử lý, gồm cả `no_plate` và lỗi; có số vùng và thời gian tính toán |
| `summary.json` | Phiên bản môi trường/model, SHA256 detector, tham số, số khung/vùng/track và trạng thái chạy |
| `001_<tên ảnh>.jpg` | Ảnh toàn cảnh có khung và chữ dự đoán |
| `001_preview.mp4` | Video có khung và chữ, nếu đầu vào là video |

`text` chỉ viết hoa và bỏ dấu phân cách, không tự đổi O↔0, I↔1… `raw_text`,
`line1` và `line2` giữ dự đoán gốc. `split_y` là tọa độ trong crop sau tiền xử lý,
không phải tọa độ y của ảnh toàn cảnh. `rectified=1` ghi nhận đã nắn theo viền.
Tọa độ x1/y1/x2/y2 luôn thuộc ảnh toàn cảnh đã áp dụng EXIF orientation.

`ocr_score` là trung bình nhân xác suất token của PARSeq (gồm token kết thúc),
không phải xác suất biển đúng. `needs_review=1` là gợi ý xem lại, không phải một
bộ xác nhận chính xác. `accuracy` để null khi không có nhãn thật để chấm.
Nhập nhãn bằng tay chỉ cần khi xây bộ đánh giá, không cần khi vận hành luồng này.

## Giới hạn cần biết khi đọc kết quả

- Detector pretrained tổng quát, chưa fine-tune hoặc đo recall trên bộ biển VN của nhóm.
  Có thể bỏ sót biển nhỏ/che khuất; phát hiện được một số biển không có nghĩa tìm đủ.
- Nắn viền, phân loại bố cục và cắt hai dòng là heuristic. Ảnh nghiêng, mờ, viền
  mất, biển bị che hoặc bố cục lạ có thể đọc sai. `--no-rectify` giúp đối chiếu;
  `--layout one` hoặc `--layout two` chỉ để thử một tập có bố cục đã biết.
- Tracker chỉ ghép vùng bằng IoU và khoảng cách thời gian. Xe bị che, quay lại
  khung hình hoặc đi quá nhanh có thể tạo track mới/ghép nhầm. Không đảm bảo một
  dòng cho mỗi xe duy nhất trên toàn video. Một track có thể liên tục đọc sai.
- Timestamps lấy từ chỉ số khung/FPS danh nghĩa, gần đúng với video FPS biến đổi.
- Chạy trên file video không chứng minh đạt thời gian thực trên camera hay NPU.
  `compute_ms` không gồm decode, đọc/ghi file, tải model hay warm-up. Thời gian
  `stream_wall_seconds` gồm xử lý luồng và ghi preview, đã loại warm-up đầu tiên.
- CUDA lỗi sẽ báo thất bại; không tự chuyển CPU rồi vẫn báo đang dùng GPU.

## Kiểm tra đã thực hiện khi viết công cụ

- Unit tests cho tọa độ detector, RGB/letterbox, tách dòng, nắn tứ giác/fallback,
  tracker một-một/bỏ phiếu/hết hạn, kiểm checksum tải model, CSV ảnh không có
  biển, video lấy mẫu/ghi preview và báo lỗi. Model được thay bằng stub trong tests CLI.
- Chạy detector ONNX và PARSeq CPU thật trên 6 ảnh VN công khai được liên kết
  dưới đây: hoàn thành, xuất 14 vùng. Đây là kiểm tra chạy được, không phải mAP,
  recall hoặc accuracy. Không lấy output của model làm nhãn.
- Chạy video kiểm tra tạo từ ảnh bằng dịch chuyển hình học để kiểm luồng thực;
  không dùng kết quả này để tuyên bố hiệu năng trên video giao thông thực tế.
- Chưa chạy native Windows/CUDA hoặc card DEEPX trong môi trường kiểm tra này;
  cần đối chiếu bằng kết quả trên máy của nhóm.

## Nguồn detector và ảnh thử

Detector ONNX từ Open Image Models:

- Repo: https://github.com/ankandrew/open-image-models
- Model: https://github.com/ankandrew/open-image-models/releases/download/assets/yolo-v9-t-640-license-plates-end2end.onnx
- SHA256: `c3c1026ca7d0585dd88084d68182dd897113712fa734ae1557ca70174440c076`
- Quy ước letterbox và output `[batch,x1,y1,x2,y2,class,score]` đối chiếu với
  `open_image_models/detection/core/yolo_v9` ở commit
  `f22000e02b30642f317cdba7755c0631638b109e`.
- Thông báo MIT của nguồn được giữ tại `third_party/open-image-models-LICENSE.txt`.

Ảnh thử: https://github.com/trungdinh22/License-Plate-Recognition/tree/main/test_image
(các file `1.jpg`, `3.jpg`, `4.jpg`, `101.jpg`, `117.jpg`, `119.jpg`). Không đóng gói
ảnh hoặc trọng số vào repo; không suy diễn quyền dùng dữ liệu thương mại từ việc
file có thể tải công khai.

## Commit sau khi chạy ổn trên Windows

Chỉ thêm các file code/tài liệu sau, không thêm ảnh/model/output:

```powershell
git add alpr_auto.py requirements/alpr-auto.txt tests/test_alpr_auto.py docs/ALPR_AUTO_WINDOWS.md third_party/open-image-models-LICENSE.txt
git commit -m "Add automatic plate detection and PARSeq OCR for images and video"
git push
```

Nếu vẫn còn các file gán nhãn staged từ bản patch trước, xem `git status` trước
khi commit để biết toàn bộ nội dung sẽ lên cùng commit; không xóa các file đó.
