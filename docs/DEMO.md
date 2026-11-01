# Flow demo DX-M1: chọn ảnh hoặc video mới

**Mở trang này khi demo.** Flow dùng laptop Ubuntu đã setup DX-M1, encoder PARSeq
đã compile và `.venv-linux` đang chạy được. Người xem có thể đưa file ảnh/video mới;
không cần khoanh biển, nhập text chuẩn hoặc sửa source code.

## 1. Trước buổi demo: làm một lần

Lấy code đúng nhánh trên laptop:

```bash
cd "$HOME/OneKiwi/renesas-alpr"
git switch truc/parseq-cross-platform
git pull --ff-only origin truc/parseq-cross-platform
```

Cài công cụ chọn file, chuyển codec và phát video nếu máy chưa có:

```bash
sudo apt-get update
sudo apt-get install -y zenity ffmpeg vlc
```

Không cần chạy lại lệnh cài mỗi lượt demo. `zenity` mở hộp chọn file; nếu không có,
launcher cho dán/kéo thả đường dẫn trong terminal. Video được xuất thêm H.264 và
mở VLC để tránh phụ thuộc ứng dụng MP4 mặc định. Ảnh/HTML mở qua trình duyệt.

Sau khi khởi động lại Ubuntu, nạp driver và kiểm tra:

```bash
sudo modprobe dx_dma
sudo modprobe dxrt_driver
cd "$HOME/OneKiwi/renesas-alpr"
.venv-linux/bin/python demo_alpr.py --check
```

Cần thấy `DX-M1: OK`; `--check` chỉ kiểm tra môi trường/card, không chạy inference.
Driver phải khớp kernel đang boot. Nếu clone repo trên máy mới, làm
[setup Ubuntu](DEEPX_UBUNTU_PREP.md), [setup Python](SETUP_DUAL_BOOT.md) và
[compile full encoder](DXM1_DEPLOYMENT.md) trước; model/SDK/venv không nằm trong Git.
Detector và checkpoint cần được tải/cached từ lượt chạy đầu. Chạy thử một ảnh có sẵn
trước khi đến công ty để kiểm tra cả pipeline, không chỉ trạng thái card:

```bash
.venv-linux/bin/python demo_alpr.py --source inputs/vn_eval_v1/images/vn_dev_01.jpg
```

Ảnh ví dụ ở trên có sẵn sau lượt `run_vn_eval.sh`. Nó chỉ là tự kiểm tra; lúc demo
người xem có thể chọn file hoàn toàn mới ở bước 2.

## 2. Tại buổi demo: một lệnh cho nguồn mới

```bash
cd "$HOME/OneKiwi/renesas-alpr"
.venv-linux/bin/python demo_alpr.py
```

1. Hộp chọn file hiện ra: chọn ảnh hoặc video người khác đưa, ở Downloads, USB
   hoặc thư mục bất kỳ. Bấm Cancel sẽ thoát mà không chạy inference.
2. Hệ thống chạy **ảnh/frame mới** qua detector + OCR. Terminal in tiến độ.
3. Chạy xong tự mở trang kết quả có ảnh/khung/text và link CSV. Với video,
   tự tạo bản H.264 và mở VLC phát lặp để dễ trình diễn; dừng bằng nút Stop/đóng VLC.
4. Chọn file khác: quay lại terminal và chạy lại cùng lệnh. Mỗi lượt tạo output mới.

Nếu không hiện hộp chọn file, kéo file từ Files vào terminal (hoặc dán đường dẫn)
rồi Enter. Launcher hỗ trợ đường dẫn có dấu cách, dấu nháy do kéo thả và file URI.
Cũng có thể truyền `--source` kèm đường dẫn thực của file/thư mục; thư mục sẽ được
pipeline duyệt ảnh/video. Đầu vào luôn là file cục bộ, không phải link YouTube/webcam.

Không cần compile lại encoder khi đổi ảnh/video. Đổi checkpoint hoặc model là việc
khác: phải kiểm tra/compile lại artifact phù hợp.

## 3. Video dài: chọn cách xử lý rõ ràng

Mặc định xử lý **mọi frame, toàn video** (`--max-frames 0`, `--every-n-frames 1`).
Nên chọn clip ngắn để buổi demo ít phải chờ. Muốn thử nhanh 120 frame đầu được xử lý:

```bash
.venv-linux/bin/python demo_alpr.py --max-frames 120
```

Muốn lấy mỗi 3 frame một frame:

```bash
.venv-linux/bin/python demo_alpr.py --every-n-frames 3
```

`max-frames` tính theo số frame được xử lý, không phải số giây. Khi lấy mẫu,
preview có FPS lấy mẫu theo nguồn. Video mở sau khi xử lý xong, **không phải camera
live**; tốc độ phát không phải tốc độ inference. Không dùng FPS của trình phát để
báo hiệu năng hệ thống.

## 4. Kết quả nằm ở đâu?

Terminal in `KẾT QUẢ: .../results/demo_<timestamp>` và `MỞ: .../index.html`.

| File | Ý nghĩa |
| --- | --- |
| index.html | Trang xem kết quả trên máy, không cần web server |
| 001_…jpg | Ảnh model tự vẽ khung biển và text OCR; không có GT overlay |
| 001_preview.mp4 | Preview video gốc của pipeline |
| 001_preview_H264.mp4 | Bản H.264 để mở/xem/gửi thuận tiện |
| detections.csv | Kết quả mỗi biển ở mỗi frame |
| events.csv | Sự kiện gộp theo tracking heuristic |
| frames.csv | Trạng thái và compute time theo frame |
| summary.json | Backend, model hash, bộ đếm và metadata pipeline |
| demo_run.json | Lệnh/nguồn/run model và trạng thái DX-RT của lượt demo |

Nếu dùng thư mục nhiều nguồn, prefix 001/002/... sẽ khác nhau theo nguồn. Trang
kết quả hiển thị toàn bộ ảnh/video, VLC tự mở video đầu tiên. Có thể chạy
`--no-open` để chỉ tạo file khi không có desktop hoặc muốn tự mở.

Ảnh/video mới chưa có nhãn nên không có accuracy. Trang kết quả sẽ hiển thị trung
thực trường hợp không tìm thấy biển hoặc OCR báo lỗi; không tự đổi sang encoder CPU.
Màu khung trong preview: xanh lá là không bị heuristic gắn cờ, vàng/cam là cần xem
lại. Màu này không phải ground-truth accuracy hoặc xác suất đọc đúng đã hiệu chuẩn.

## 5. Model đang dùng và CPU/NPU làm gì?

| Bước | Model/xử lý | Thiết bị |
| --- | --- | --- |
| Tìm biển | YOLOv9-t ONNX | CPU |
| Chuẩn bị vùng chữ | Crop, chỉnh phối cảnh, tách dòng, resize, pack patch | CPU |
| Trích xuất đặc trưng | Encoder PARSeq, raw-patch MatMul INT8 | DX-M1 |
| Đọc ra ký tự | Decoder PARSeq, giữ autoregression/refinement | PyTorch CPU |
| Kết quả | Tracking, vẽ khung/chữ, ghi CSV; chuyển codec video | CPU |

Launcher tự tìm full encoder có hash khớp mốc phần cứng trong
`docs/evidence/dxm1_20261101.json`. Không tự lấy bản compile thử mới nhất. Nếu có
nhiều bản phù hợp hoặc không tìm thấy artifact, truyền `--dx-run` kèm đường dẫn
full run thực tế. Trên laptop đã xác minh có thể chỉ rõ:

```bash
.venv-linux/bin/python demo_alpr.py \
  --dx-run artifacts/dxm1_full_encoder_mm_20261101_134233_21170
```

CLI backend vẫn kiểm tra checkpoint/ONNX/DXNN và runtime IO. Tham số `--device cpu`
được launcher truyền cho **decoder**, encoder vẫn đi qua DX-M1.

## 6. Nói gì trong lúc chờ xử lý?

“Đây là ảnh/video toàn cảnh mới, em không khoanh sẵn biển hay nhập đáp án. YOLO tìm
biển, CPU chuẩn bị vùng chữ, encoder PARSeq chạy DX-M1, decoder CPU đọc ra ký tự.
Chạy xong mình xem khung/text và CSV của chính đầu vào này.”

Khi được hỏi kết quả đã kiểm chứng: CPU và DX-M1 cùng đúng 5/5 ảnh có nhãn trong
bộ development nhỏ; conversion fidelity khớp text PyTorch ở 6/6 demo crops. Đây
không phải độ chính xác tổng quát. Chưa chốt FPS toàn pipeline và chưa tích hợp host
Renesas. Với file mới đọc sai, giữ output để phân tích lỗi thay vì sửa đáp án trên ảnh.

## 7. Lỗi thường gặp: xử lý nhanh

| Hiện tượng | Cách làm |
| --- | --- |
| Chưa nhận M1 | Chạy lại hai lệnh modprobe ở bước 1, rồi dxrt-cli -s; kiểm tra kernel |
| Thiếu dx_engine/torch/cv2 | Dùng đúng .venv-linux/bin/python và hướng dẫn setup |
| Không thấy encoder đã kiểm chứng | Giữ thư mục artifacts trên laptop, hoặc compile theo DXM1_DEPLOYMENT.md; chọn --dx-run |
| File không tồn tại | Chọn lại bằng hộp chọn file, tránh gõ đường dẫn ví dụ |
| Không có app MP4 | Cài VLC ở bước 1; launcher sẽ gọi VLC trực tiếp |
| Trình duyệt không phát H.264 | Xem cùng file H264.mp4 bằng VLC |
| Không phát hiện biển | Xem đúng phần clip đã xử lý, thử biển đủ lớn/rõ; đây là kết quả cần đánh giá, không coi là pass OCR |
| Khung có nhưng chữ sai | Giữ ảnh, detections.csv và summary.json để phân biệt crop/tách dòng/OCR |
| Pipeline STOP | Giữ log; không flash/reset firmware để thử sửa khi chưa xác định nguyên nhân |

Nếu git pull bị chặn do file đang sửa/untracked, giữ các file đó và xem git status;
không dùng reset --hard/clean để chuẩn bị demo.

## 8. Kiểm tra phần launcher

```bash
.venv-linux/bin/python -m unittest discover -s tests -p test_demo_flow.py -v
```

Tests dùng inference giả, file mới thật trong thư mục tạm và FFmpeg thật cho
conversion video. Không coi các tests này là phép chạy NPU. Luồng inference gốc
`alpr_auto.py`/`alpr_auto_dxm1.py` không bị sửa; lần chạy bằng card ở laptop cần
được kiểm tra thêm qua bước 1. Giữ nguyên toolchain Ruff/Pyink và unittest của repo;
không đưa migration type checking toàn repo vào thay đổi flow demo.
