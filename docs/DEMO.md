# Demo nhanh: menu và lệnh ngắn

Trên laptop Ubuntu đã setup card/model, mở terminal tại repo rồi chạy:

```bash
./alpr
```

Menu có các nút số để **chuẩn bị card → thêm ảnh/video → quét → mở lại kết quả**.
Mỗi lần chọn số rồi Enter. Không cần nhớ tên artifact, đường dẫn output hay lệnh Python.
Đổi file đầu vào không cần compile lại encoder.

## 1. Các lệnh dùng lúc demo

| Lệnh | Việc thực hiện |
| --- | --- |
| `./alpr` | Mở menu, dùng số để thao tác |
| `./alpr start` | Nạp hai driver rồi kiểm tra DX-M1; có thể hỏi mật khẩu sudo |
| `./alpr add` | Chọn ảnh/video mới và sao chép vào hàng đợi |
| `./alpr status` | Xem các file đang chọn và đường dẫn kết quả gần nhất |
| `./alpr scan` | Quét hàng đợi và tự mở kết quả; tối đa 120 frame/video |
| `./alpr scan --all` | Quét mọi frame, toàn bộ các video trong hàng đợi |
| `./alpr open` | Mở lại trang ảnh/video của lần quét thành công gần nhất |
| `./alpr open --folder` | Mở thư mục chứa ảnh/video, CSV và summary |
| `./alpr clear` | Bỏ chọn toàn bộ input; giữ file gốc, bản sao và kết quả |
| `./alpr check` | Chỉ kiểm tra môi trường/card, không nạp driver/inference |
| `./alpr help` | Xem trợ giúp |

Lượt demo đầu sau khi boot:

```bash
./alpr start
./alpr add
./alpr scan
```

Lượt tiếp theo với input khách khác:

```bash
./alpr clear
./alpr add
./alpr scan
```

Hoặc chỉ giữ cửa sổ menu `./alpr`, chọn lần lượt 1 → 2 → 3. Menu số 4 mở lại kết quả,
số 5 xem input, số 6 bỏ chọn input và số 0 thoát.

`add` thêm vào hàng đợi hiện tại, không tự xóa các file đã chọn. Có thể thêm nhiều
lần rồi quét chung. Nếu thiếu zenity, terminal cho dán/kéo thả đường dẫn rồi Enter.
Đường dẫn có dấu cách được hỗ trợ. Nếu biết đường dẫn, đặt sau `./alpr add`; có thể
đưa nhiều đường dẫn hoặc thư mục ảnh/video. Ví dụ tự kiểm tra với ảnh đã có:

```bash
./alpr add inputs/vn_eval_v1/images/vn_dev_01.jpg
```

Nguồn này có sẵn sau lượt `run_vn_eval.sh`; ảnh/video khách đưa được chọn bằng hộp
chọn file, không cần đặt một tên cố định. Input là file cục bộ; chưa phải webcam/URL.

## 2. Quét nhanh hay quét toàn bộ?

`./alpr scan` xử lý mọi frame trong phần đã chọn, nhưng **giới hạn 120 frame đầu
mỗi video** để lượt demo ngắn. Ảnh tĩnh vẫn xử lý từng ảnh. Muốn toàn bộ video,
dùng `./alpr scan --all`; có thể mất lâu tùy độ dài và số biển.

Muốn giới hạn khác hoặc chỉ lưu output:

```bash
./alpr scan --max-frames 300
./alpr scan --all --no-open
```

120 frame không có nghĩa là 120 giây. Video được xử lý xong rồi mở phát lặp bằng VLC;
đây không phải camera live và FPS của trình phát không phải FPS inference.
Nếu biển chỉ xuất hiện ở đoạn sau, dùng --all hoặc --max-frames lớn hơn.

## 3. Kết quả và input nằm đâu?

- File gốc không bị sửa. Bản sao input nằm ở `inputs/demo_inbox/files`.
- Danh sách đang chọn và kết quả gần nhất được ghi ở `inputs/demo_inbox/state.json`.
- Mỗi lần scan có snapshot chỉ gồm input đang chọn ở `inputs/demo_batches`.
  File cũ đã bỏ chọn không bị quét lẫn vào lượt mới.
- Kết quả nằm trong thư mục mới `results/alpr_<timestamp>`.
- `clear` chỉ bỏ chọn, không xóa các bản sao, snapshot hoặc kết quả cũ.
- `open` nhớ lần quét thành công tạo bởi `./alpr`, không tự đoán từ các run cũ.
  Nếu lượt mới lỗi, vẫn giữ con trỏ tới lần thành công trước và báo lỗi rõ.

Output có index.html, ảnh đã vẽ khung/text, video H.264, detections.csv, events.csv,
frames.csv, summary.json và demo_run.json. Trang kết quả mở bằng trình duyệt; nếu
có VLC thì video đầu tiên mở phát lặp. Các video khác cũng nằm trên trang kết quả.
`./alpr open --folder` để lấy file gửi công ty.

Input và results đã được gitignore. Không git add ảnh/video khách đưa. Code wrapper
không upload dữ liệu hay gửi tin nhắn. Menu được thiết kế cho một phiên thao tác
trên laptop, chưa phải hàng đợi đồng thời cho nhiều người dùng.

## 4. Chuẩn bị máy một lần

Lấy code đúng nhánh:

```bash
cd "$HOME/OneKiwi/renesas-alpr"
git switch truc/parseq-cross-platform
git pull --ff-only origin truc/parseq-cross-platform
```

Cần `.venv-linux` đã chạy được pipeline và full encoder đã compile trên laptop.
Các công cụ cho chọn file và phát video có thể cài một lần:

```bash
sudo apt-get update
sudo apt-get install -y zenity ffmpeg vlc
```

Sau đó dùng `./alpr start`. Driver phải khớp kernel đang boot. Wrapper không cài SDK,
flash firmware hoặc đổi kernel. Nếu clone repo trên máy mới, làm
[setup Ubuntu](DEEPX_UBUNTU_PREP.md), [Python](SETUP_DUAL_BOOT.md),
[compile encoder](DXM1_DEPLOYMENT.md) trước. Model/SDK/venv không nằm trong Git.
Chạy thử trước với một ảnh để cache detector/checkpoint và kiểm tra cả pipeline.

Nếu file `alpr` chưa có quyền chạy, dùng `bash alpr` cho menu hoặc `chmod +x alpr`.
Gọi script từ thư mục khác bằng đường dẫn đầy đủ tới file alpr vẫn được; input tương
đối được hiểu theo thư mục terminal hiện tại.

## 5. Giải thích với khách

“Đây là ảnh/video mới do anh/chị đưa. Hệ thống tự tìm biển và đọc chữ, không cần
khoanh hay nhập đáp án. Encoder PARSeq chạy trên DX-M1; detector, xử lý ảnh và
decoder chạy CPU. Kết quả hiển thị trên ảnh/video và có CSV để tích hợp.”

| Thành phần | Thiết bị |
| --- | --- |
| YOLOv9-t tìm biển | CPU |
| Crop/chỉnh ảnh/tách dòng/resize/pack patch | CPU |
| Encoder PARSeq đã chuyển đổi | DX-M1 |
| Decoder PARSeq, tracking, vẽ kết quả và CSV | CPU |

Trên bộ development nhỏ đã có nhãn, CPU và DX-M1 cùng đúng 5/5 ảnh. Với input mới
chưa có đáp án chuẩn thì chưa công bố accuracy. Video đã xử lý rồi phát lại chưa
chứng minh thời gian thực. Host hiện là laptop Ubuntu; chưa tích hợp host Renesas.

Màu xanh/vàng trong preview là heuristic cần xem lại, không phải xác suất đọc đúng
đã hiệu chuẩn. Nếu không thấy biển hoặc OCR sai, giữ output để phân tích; không coi
hai bên cùng không đọc được là conversion pass.

## 6. Lỗi và công cụ chi tiết

| Hiện tượng | Làm gì |
| --- | --- |
| Chưa nhận DX-M1 | `./alpr start`, xem kernel và dxrt-cli -s nếu driver không nạp được |
| Input cache thiếu/đã đổi | Giữ file gốc; `./alpr clear` rồi thêm lại file nguyên vẹn, kiểm tra cache nếu hash báo khác |
| Chưa có output để mở | Chạy `./alpr add` rồi `./alpr scan`; open chỉ nhớ lượt thành công của script này |
| Video không mở trong browser | Dùng VLC hoặc mở thư mục bằng `./alpr open --folder` |
| Quét không thấy biển | Kiểm tra phần đầu clip đã được xử lý hay dùng scan --all; xem kích thước/độ rõ của biển |
| State bị hỏng | Giữ inputs/demo_inbox/state.json để kiểm tra, không tự ghi đè/reset |
| Git bị chặn bởi file đang sửa | Giữ git status và thay đổi đó; không reset --hard/clean để chuẩn bị demo |

Launcher chi tiết vẫn dùng được: `.venv-linux/bin/python demo_alpr.py` cho một
nguồn trực tiếp không qua hàng đợi; có --source, --dx-run, --max-frames, --every-n-frames,
--layout, --output và --no-open. Khác với menu nhanh, launcher chi tiết mặc định xử lý
**toàn bộ** video. Menu tự chọn encoder theo hash mốc đã kiểm chứng như launcher;
truyền --dx-run sau lệnh start/check/scan nếu cần chọn full run khác đã kiểm tra.

Tests cho hàng đợi và launcher:

```bash
.venv-linux/bin/python -m unittest discover -s tests -p test_alpr_console.py -v
.venv-linux/bin/python -m unittest discover -s tests -p test_demo_flow.py -v
```

Tests dùng inference giả và file tạm, FFmpeg thật cho kiểm tra codec; chưa chạy menu
mới trên card thật của laptop. Giữ pipeline/backend gốc và toolchain unittest,
Ruff/Pyink hiện tại; không gộp migration type checking toàn repo vào tính năng này.
