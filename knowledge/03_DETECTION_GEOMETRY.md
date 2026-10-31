# 03 — Từ ảnh toàn cảnh đến crop đúng

## Detector và recognizer trả hai loại output

Detector trả vị trí biển và score. Recognizer trả chuỗi ký tự từ crop. Cần đánh giá riêng: detector bỏ sót → OCR không bao giờ thấy biển; OCR tốt trên crop chuẩn chưa chứng minh pipeline toàn cảnh tốt.

Detector mặc định trong repo là YOLOv9 end-to-end ONNX do open-image-models phát hành. Không thay một model khác vào `--detector` chỉ vì tên file cùng .onnx: decoder hiện đòi output [N,7].

## Ảnh và màu

OpenCV thường dùng BGR/HWC/uint8; ảnh PIL đưa vào PARSeq dùng RGB. Chuyển màu hai lần có thể đảo về màu sai. Lưu một crop để nhìn chỉ kiểm tra hình ảnh; cần kiểm tra thêm metadata của tensor thực đưa vào model.

Ảnh EXIF có thể cần xoay theo metadata. Crop tính trên ảnh chưa xoay mà hiển thị trên ảnh đã xoay sẽ lệch tọa độ.

## Letterbox và phép đổi ngược

Resize giữ tỉ lệ rồi thêm padding vào input detector vuông. Với ảnh W×H, cạnh đích S:

```math
r=\min(S/W,S/H),\quad p_x=(S-Wr)/2,\quad p_y=(S-Hr)/2
```

Ví dụ lý tưởng không có sai số làm tròn: W=1280, H=720, S=640 → r=0.5, padding y=140. Điểm gốc (100,200) thành (50,240). Đổi ngược:

```math
x=(x'-p_x)/r,\qquad y=(y'-p_y)/r
```

Code thật có làm tròn resize/border. Đọc cách `letterbox` lưu ratio/padding và `decode_boxes` dùng chúng; đừng trộn một hàm letterbox mới với decoder cũ mà chưa kiểm tra.

Bbox trong baseline: `x1,y1,x2,y2`; crop NumPy theo `frame[y1:y2,x1:x2]`. Đảo x/y hoặc nhầm xywh với xyxy sẽ cho crop sai dù detector có vẻ hoạt động.

## Output detector hiện tại

Mỗi hàng output gồm batch, x1, y1, x2, y2, class, score. Baseline lọc score/class, đổi ngược padding và scale, rồi clip theo ảnh gốc. Kiểm tra NaN/Inf và kích thước dương.

Model end-to-end đã có xử lý chọn detection trong graph; khi đổi sang raw-output model cần thiết kế decoder/NMS đúng. Không thêm NMS tùy tiện vào graph cũ để thử chữa bbox lệch.

IoU là diện tích giao chia diện tích hợp, trong [0,1]. Dùng cho đánh giá/ghép bbox, không phải xác suất hai xe là một.

## Chỉnh phối cảnh và hai dòng

Homography là phép biến đổi phối cảnh từ bốn điểm tương ứng. Nó hữu ích khi chọn đúng góc; chọn viền nền thay vì viền biển có thể làm chữ tệ hơn.

`rectify_plate` và `line_regions` hiện là heuristic. Aspect ratio và khoảng trống theo hàng không chứng minh được mọi biển hai dòng. Với ảnh khó, so sánh:

- Crop chưa rectify và đã rectify.
- `--layout auto`, `one`, `two`.
- Từng dòng có bị mất nét, chứa nhầm dòng khác hoặc sai thứ tự không.

Chuỗi của biển hai dòng phải được ghép theo chính sách thống nhất. Regex có thể đánh dấu cần xem lại, nhưng không tự biết “O” nào thật ra là “0”.

## Bài thực hành

Đọc R07–R09; làm Lab 4.

Dùng ảnh được phép sử dụng; lưu bbox overlay, crop trước/sau và từng dòng. Một thành viên tự tính tay phép đổi cho một bbox; thành viên khác đối chiếu code.

Đạt khi bạn phân biệt được detector sai với decode tọa độ sai bằng chứng cụ thể.
