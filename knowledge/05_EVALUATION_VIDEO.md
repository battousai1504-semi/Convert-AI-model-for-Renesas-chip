# 05 — Đánh giá đúng và hiểu video/events

## Bốn loại bằng chứng khác nhau

| Kiểm tra | Trả lời được | Không chứng minh |
| --- | --- | --- |
| Unit tests | Helper xử lý đúng các case đã viết | Model nhận dạng đúng ngoài thực tế |
| PyTorch ↔ ONNX | Hai đường chạy khớp trên dữ liệu đã thử | Chuỗi khớp nhãn thật hoặc NPU chạy được |
| Đánh giá có ground truth | Độ đúng theo protocol/bộ dữ liệu đã định nghĩa | Độ đúng cho mọi camera/điều kiện |
| Inference card thật | Tổ hợp card/driver/runtime/model hoạt động | Sẵn sàng cho mọi khách hàng |

Gọi đúng tên kiểm tra để nhóm không bàn giao quá mức bằng chứng.

## Tạo bộ đánh giá

Ground truth là đáp án được xác minh độc lập. Không dùng dự đoán của chính model làm đáp án. Giữ ID ảnh/video, nguồn, nhãn đọc được, bbox/layout khi cần, trạng thái không đọc được và phiên bản dữ liệu.

Tách train, validation, calibration và test theo nguồn/xe/video/camera khi phù hợp. Hai frame gần nhau không phải hai quan sát độc lập để rải ngẫu nhiên sang train/test. Giữ một test set không dùng để chọn threshold hay sửa heuristic.

Chọn mẫu đa dạng: ô tô/xe máy, một/hai dòng, ngày/đêm, góc nghiêng, motion blur, biển nhỏ, nhiều xe, che khuất, không có biển. Đây là danh sách tình huống để lấy mẫu, không phải tập dữ liệu đã được cung cấp đầy đủ.

Nhóm cần thống nhất cách ghi chữ, phân cách, biển không đọc được và quyền dùng dữ liệu. Không tự loại ảnh khó khỏi mẫu số sau khi thấy model sai.

## Các chỉ số

Với N ảnh crop có nhãn:

```math
\mathrm{ExactAccuracy}=\frac{\#\{\hat y_i=y_i\}}{N}
```

```math
\mathrm{CER}=\frac{\sum_i\mathrm{EditDistance}(\hat y_i,y_i)}{\sum_i|y_i|}
```

CER có thể lớn hơn 1 nếu nhiều insertion. “90% ký tự đúng” không bằng “90% biển đúng”.

Trong `ocr_batch.py`, chính sách so sánh chuẩn hóa là uppercase ASCII chữ/số, bỏ phân cách; không tự thay O/0, I/1. Lỗi đọc với nhãn thật vẫn tính sai. Thiếu nhãn thì accuracy/CER là null.

Ví dụ: 10 biển, 9 biển đúng hoàn toàn và 1 biển sai 1 ký tự trên tổng 80 ký tự → exact accuracy 90%, CER 1.25%. Mẫu rất nhỏ nên chưa đủ làm cam kết khách hàng.

Với toàn cảnh, cần tách:
- Precision/recall detector theo ngưỡng ghép bbox đã chọn trước.
- OCR trên crop chuẩn và OCR trên crop do detector tạo.
- Tỉ lệ cả biển đúng trên tất cả biển ground truth; bỏ sót detector cũng là lỗi end-to-end.
- False positive ở ảnh không biển.

Các chỉ số end-to-end này là protocol cần làm; đừng mặc định summary hiện tại đã tính chúng.

## Frame, detection, track và event

Frame là khung ảnh. Detection là một vùng tìm được trong frame. Track là nhóm quan sát được gán cùng ID qua thời gian. Event là hàng tổng hợp khi track kết thúc.

Tracker hiện tại trong `PlateTracks` ghép một-một bằng IoU và giới hạn khoảng trống, rồi vote chuỗi. Nó không phải mô hình nhận diện danh tính xe. Mất detection, xe giao nhau hoặc nhảy bbox có thể làm một xe thành nhiều track, hoặc ghép nhầm.

Vote nhiều frame giúp giảm lỗi ngẫu nhiên; nếu mọi crop đều sai hệ thống, vote có thể làm sai rất chắc chắn. `vote_fraction` và `ocr_score` không phải accuracy đã hiệu chuẩn.

## Thời gian và lấy mẫu video

`--every-n-frames 3` chỉ xử lý mỗi 3 frame; `--max-frames 30` giới hạn 30 frame được xử lý, không nhất thiết 30 frame đầu của file. Kiểm tra `frames` nếu đổi semantics.

Thời gian theo frame index/FPS của file là xấp xỉ, nhất là video variable-frame-rate. Không lấy số frame đã xử lý chia thời lượng video rồi gọi đó là tốc độ xử lý của máy.

Lưu cả frame không biển/lỗi để biết hệ thống bỏ qua gì. Xem:
- `detections.csv`: từng vùng;
- `events.csv`: tổng hợp track;
- `frames.csv`: tình trạng từng frame được xử lý.

## Bài thực hành

Làm Lab 6–7. Đọc [video OpenCV](RESOURCES.md#r08), xem `PlateTracks` và `summarize`.

Đạt khi bạn có một ví dụ vote sai có thể tái hiện, phân biệt được detection/event/xe thật, và tính được accuracy/CER bằng tay trên bộ nhãn nhỏ.
