# Đánh giá ALPR Việt Nam: bộ khởi đầu v1

## Mục tiêu và phạm vi

Đo riêng khả năng tìm biển và đọc đúng toàn biển của pipeline đang có. Giữ nguyên
baseline `alpr_auto.py`, adapter `alpr_auto_dxm1.py` và encoder đã qua smoke test.
Nhãn chỉ được evaluator đọc sau inference; không đưa text hoặc box chuẩn vào model.

Bộ đầu gồm **5 ảnh xe máy hai dòng**, lấy từ sample của
[phthanhtung/alpr-vietnam-system tại commit cố định](https://github.com/phthanhtung/alpr-vietnam-system/tree/3455c86eb2903072133860ad40419dcb4727b92c/data/sample).
Khung biển lấy từ YOLO labels gốc; chữ được trợ lý kiểm tra trực tiếp từ pixels,
không dùng đầu ra OCR làm đáp án. Script sinh `label_review.jpg` để nhóm kiểm tra
lại độc lập trước khi dùng số liệu làm tiêu chí nghiệm thu.

| ID | Text đã chuẩn hóa |
| --- | --- |
| vn_dev_01 | 59D208525 |
| vn_dev_02 | 37B296286 |
| vn_dev_03 | 59E166764 |
| vn_dev_04 | 54V29672 |
| vn_dev_05 | 54R23076 |

Đây là **development starter**, chưa phải test set đại diện: cả 5 có bối cảnh bãi
xe tương tự; chưa có ô tô một dòng, cảnh đêm, nhiều xe hoặc mẫu âm tính thật.
Chưa biết ảnh có trùng dữ liệu huấn luyện model pretrained nào hay không. Nhóm
nguồn được đặt bảo thủ thành một group; không gọi 5 ảnh là 5 phiên quay độc lập.

Recipe lưu URL commit cố định, SHA256 của ảnh/nhãn gốc và annotation provenance.
Gói chỉ chứa recipe/code, không phân phối lại ảnh hoặc toàn bộ dataset.

## Chạy trên laptop hiện tại

```bash
cd "$HOME/OneKiwi/renesas-alpr"
bash run_vn_eval.sh
```

Runner thực hiện:

1. Chạy tests của evaluator, kiểm tra môi trường/model run đã có.
2. Kiểm tra không trùng hash ảnh với calibration trong matmul_contract.json,
   tải/kiểm tra checksum 5 ảnh và tạo manifest + tờ kiểm nhãn.
3. Chạy baseline CPU và encoder DX-M1 + decoder CPU trên cùng thư mục ảnh,
   auto layout, cùng detector và thiết lập. Không đọc nhãn trong bước này.
4. Chấm hai CSV bằng cùng nhãn; ghi báo cáo, lỗi và ảnh có khung đối chiếu.

Muốn dùng một full encoder run khác:

```bash
bash run_vn_eval.sh artifacts/YOUR_FULL_ENCODER_RUN
```

Runner mặc định dùng full encoder thành công
`artifacts/dxm1_full_encoder_mm_20261101_134233_21170`. Không có bước compile/tune
trong lượt đánh giá. `--device cpu` chỉ thiết bị decoder ở nhánh DX-M1.

## File đầu ra

```text
inputs/vn_eval_v1/
  dataset.json           # ảnh + box + text chuẩn
  label_review.jpg       # kiểm chữ trực tiếp trên ảnh, trước khi xem dự đoán
  images/                # model chỉ nhận nội dung thư mục này
  upstream_labels/       # nhãn YOLO nguồn, đã kiểm checksum
results/vn_eval_<time>_<pid>/
  cpu/                   # CSV/summary pipeline CPU
  dxm1/                  # CSV/summary pipeline DX-M1
  evaluation/
    summary.json
    cpu_plates.csv
    dxm1_plates.csv
    cpu_errors.csv
    dxm1_errors.csv
    cpu_dxm1_comparison.csv
    review.html
    review/              # ảnh: khung chuẩn xanh lá, khung dự đoán đỏ
```

Mở review.html trong trình duyệt hoặc mở riêng ảnh JPG. CSV dùng UTF-8 BOM để
Excel đọc được. Trạng thái chấm điểm thành công không đồng nghĩa accuracy 100%:
script trả 0 khi có báo cáo hợp lệ; chạy lỗi/thiếu dữ liệu sẽ dừng với lỗi.

## Cách tính

Khung chuẩn và khung dự đoán được ghép **một-một theo hình học**, IoU >= 0.5.
Dùng matching có số cặp tối đa, duyệt theo IoU giảm dần và tie-break xác định;
không dùng text để chọn cặp, không tự nhận là matching COCO mAP hay tối đa tổng IoU.

| Chỉ số | Định nghĩa |
| --- | --- |
| Detection TP / FP / FN | Số khung ghép được / khung dự đoán thừa / biển chuẩn chưa ghép |
| Detection precision | TP / tất cả khung dự đoán |
| Detection recall | TP / tất cả biển chuẩn |
| End-to-end exact recall | Biển tìm được và đọc đúng toàn bộ / tất cả biển chuẩn |
| End-to-end exact precision | Biển tìm được và đọc đúng toàn bộ / tất cả khung dự đoán |
| OCR exact on matched boxes | Đọc đúng / số cặp khung ghép được; không tính detector miss vào mẫu số này |
| OCR CER on matched boxes | Tổng edit distance / tổng ký tự chuẩn của các cặp đã ghép |
| GT CER including missed plates | Tương tự nhưng biển bị bỏ sót được coi là chuỗi rỗng; không đo phạt text từ FP |

Số có mẫu số bằng 0 là null, không tự biến thành 100%. CER có thể lớn hơn 1 vì
có lỗi chèn ký tự. Chuẩn hóa bằng chữ HOA ASCII và chữ số, bỏ dấu cách/chấm/gạch;
không thay 0/O, 1/I hay sửa theo mẫu biển. OCR báo lỗi được chấm là chuỗi rỗng kể
cả khi trong CSV còn text. Hai hệ cùng miss/error không được tính là đồng thuận đọc.

Lỗi trong CSV:

- `detector_miss`: không có khung ghép đạt ngưỡng; cũng có thể do localization
  thấp hoặc frame lỗi, xem ảnh đối chiếu và `frame_errors`/frames.csv để phân biệt.
- `ocr_wrong`: đã ghép khung nhưng text sai.
- `ocr_error`: khung ghép được nhưng OCR báo lỗi thực thi.
- `false_positive`: khung dự đoán chưa ghép, kể cả detection trùng.

Frame lỗi và biển bị bỏ sót vẫn nằm trong số mẫu cần đánh giá. Run không kết thúc,
thiếu frame, source lạ, hash ảnh sai, box hỏng, model detector/checkpoint hoặc
preprocessing setting khác nhau giữa CPU và DX-M1 đều làm evaluator dừng.

## Thời gian và provenance

Báo cáo giữ SHA256 dataset, evaluator, báo cáo nguồn và DXNN; mean/p50/p95 của
`compute_ms` lấy từ frames.csv. Đây là compute theo frame của baseline, không gồm
đọc/giải mã ảnh, I/O, load/warmup; không suy ra FPS toàn pipeline. Với 5 ảnh, p95
nearest-rank là giá trị lớn nhất và chưa có tính ổn định thống kê.

Đây là đánh giá **ảnh toàn cảnh**, chưa đo chất lượng tracking hay event trên video.
Khi lấy frame từ video làm dữ liệu ảnh, dùng chung group_id của video và giữ group
nguyên trong một split; không coi nhiều frame của một biển là nhiều biển độc lập.

## Mở rộng dữ liệu mà không đổi model

Manifest schema v1 gồm name/scope/schema_version và images. Mỗi ảnh cần id duy nhất,
file tương đối, SHA256, width/height, split (development/regression/test), group_id
và annotations. Mỗi annotation có id trong ảnh, box_xyxy [x1,y1,x2,y2], text, layout
one/two. Dùng annotations=[] cho ảnh âm tính đã được kiểm tra toàn cảnh. Một ảnh có
thể có nhiều annotation. Không dùng nhãn trống để biểu diễn biển chưa đọc được;
chỉ thêm ảnh đã gán nhãn đầy đủ hoặc xây cơ chế ignore có đặc tả riêng trước.

Không chia cùng group sang nhiều split. Nhóm phải kiểm tra cả ảnh gần trùng/xe trùng
và phiên quay khi tách tập; hash chỉ phát hiện bản sao byte-for-byte.

Chạy inference vào hai thư mục output mới rồi chấm:

```bash
.venv-linux/bin/python evaluate_alpr_dataset.py \
  --dataset inputs/YOUR_DATASET/dataset.json \
  --cpu-run results/YOUR_RUN/cpu \
  --dxm1-run results/YOUR_RUN/dxm1 \
  --output results/YOUR_RUN/evaluation
```

Nếu sửa đáp án, giữ trace thay đổi nhãn và chấm lại vào output mới. Rerun prepare
sẽ từ chối ghi đè manifest đã sửa; cập nhật recipe có version mới và thư mục dữ
liệu mới cho các vòng chính thức.

Sau lượt starter: mở rộng theo điều kiện xe máy/ô tô, một/hai dòng, nghiêng, mờ,
đêm, nhiều xe và ảnh không có biển. Dành tập test độc lập chưa dùng để calibration,
chọn heuristic/threshold hay tune model. Sau đó mới đo video/event và hiệu năng.

## Kiểm tra phát triển

16 unittest: matching không dùng text và không greedy mất cặp, FP trùng, OCR lỗi,
miss, negative, CER/normalization, checksum, source group, thiếu frame, chấm CLI
và review artifact. Tests dùng dự đoán giả với ảnh PNG thật; không chạy model/NPU.
Có kiểm tra preparation trên 5 ảnh nguồn thật và nhãn checksum, ngoài tests giả.

Dùng Ruff 0.2.2 và Pyink 23.10.0 theo cấu hình repo đã có; không thay toolchain.
Repo hiện chưa có gate ty/basedpyright hoặc coverage threshold chung; bổ sung các
gate đó được để thành việc toolchain riêng, tránh migration toàn repo trong bước
evaluation này. Các kiểm tra ở đây không phải chứng nhận production readiness.
