# Bài thực hành theo năng lực

Chạy từ root repo, dùng interpreter ứng dụng trong [quy ước](../README.md). Ảnh/video bên dưới là dữ liệu nhóm tự chuẩn bị được phép dùng; đường dẫn ví dụ không tự tạo file. Thực hành trên branch cá nhân, không làm lỗi môi trường hoặc firmware dùng chung để “thử cho biết”.

Lưu vật trung gian ở `results/learning_<ten>/` (đã ignore). Ghi báo cáo đã kiểm chứng bằng mẫu trong [templates](../templates/LEARNING_LOG.md). Bài có card phải ghi kết quả thực; chưa có card thì chỉ hoàn thành phần phân tích.

## Lab 1 — Chứng minh đúng môi trường

**Đọc:** Bài 01. **Cần:** venv ứng dụng đã setup.

```bash
git rev-parse HEAD
.venv-linux/bin/python -c "import sys, torch, numpy; print(sys.executable); print(torch.__version__); print(numpy.__version__); print(torch.cuda.is_available())"
.venv-linux/bin/python -m pip check
```

**Nộp:** OS, commit, interpreter, versions, ý nghĩa của từng dòng. So sánh với Python hệ thống nhưng không sửa Python hệ thống.

**Đạt:** chỉ ra được package nào thuộc ứng dụng, compiler và runtime; biết lệnh nào chưa kiểm tra GPU/NPU.

## Lab 2 — File lỗi và traceback

**Đọc:** Bài 01,08. **Cần:** ảnh demo có sẵn.

Dùng pathlib resolve/is_file kiểm tra một đường dẫn cố ý không tồn tại. Sau đó đọc ảnh demo bằng read.py. Không xóa hoặc đổi tên ảnh thật để tạo lỗi.

```bash
.venv-linux/bin/python -c "from pathlib import Path; p=Path('inputs/learning_missing_01.png'); print(p.resolve()); print(p.is_file())"
.venv-linux/bin/python read.py pretrained=parseq --images demo_images/coco-1166773.jpg --device cpu
```

**Nộp:** phân tích khác nhau giữa file không có, model tải lần đầu và OCR trả sai chuỗi.

**Đạt:** không đề xuất cài lại torch cho lỗi đường dẫn; giải thích tại sao output demo “Chevrol” không phải nhãn chuẩn.

## Lab 3 — Theo dấu tensor

**Đọc:** Bài 02. Chạy ví dụ tensor, rồi đổi ảnh zero thành toàn 255, không đổi normalization.

**Kỳ vọng:** zero → −1; 255 → +1; shape sau permute/unsqueeze không đổi theo giá trị.

**Nộp:** shape/dtype/device/range ở mỗi bước, ví dụ khác biệt reshape và permute. Không dùng ảnh uniform để kết luận hai phép biến đổi tương đương; dùng tensor có các giá trị khác nhau.

## Lab 4 — Bbox và hai dòng

**Đọc:** Bài 03. Tính tay ví dụ 1280×720 → 640 trước. Sau đó chạy một ảnh toàn cảnh VN đã có thật:

```bash
.venv-linux/bin/python alpr_auto.py --source inputs/vn_sources/3.jpg --device cpu
```

**Nộp:** bbox trên ảnh gốc, crop, ratio/padding; nếu biển hai dòng, so sánh layout auto/one/two trên cùng ảnh. Dùng file crop đã lưu hoặc debug ở ranh giới, không đổi code runtime chỉ để thêm mọi loại log cùng lúc.

**Đạt:** phân biệt được bbox lệch do detector với phép đổi tọa độ; không coi score là accuracy.

## Lab 5 — Giải thích encoder/decoder

**Đọc:** Bài 04 và code liên quan.

**Nộp:** tự vẽ input 32×128, patch 4×8, 128 token, 384 feature. Tính bytes của input/memory FP32. Chỉ ra dòng load model, transform, encoder, logits và tokenizer.

**Đạt:** giải thích được vì sao hybrid còn PyTorch và thay decode_ar có thể đổi output.

## Lab 6 — Metrics bằng tay và bằng công cụ

**Đọc:** Bài 05. Chuẩn bị ít nhất 10 crop một dòng có nhãn người kiểm tra. Dữ liệu nhỏ chỉ dùng luyện protocol.

Lưu labels CSV dưới inputs với header image,text và tên tương đối. Chạy:

```bash
.venv-linux/bin/python ocr_batch.py --input inputs/learning_crops --labels inputs/learning_labels.csv --device cpu
```

**Nộp:** exact accuracy, CER tính tay trên các rows, tổng mẫu/lỗi, raw và normalized. Thử kiểm tra duplicate label trong file CSV thử riêng.

**Đạt:** mẫu số không thay đổi vì model đọc sai; biết công cụ crop chưa tự tính end-to-end detection recall.

## Lab 7 — Video và track

**Đọc:** Bài 05. Chuẩn bị video có biển thật và nhãn một đoạn ngắn; đặt vào inputs/learning_video.mp4.

```bash
.venv-linux/bin/python alpr_auto.py --source inputs/learning_video.mp4 --device cpu --every-n-frames 3 --max-frames 30
```

**Nộp:** đối chiếu detections/events/frames; giải thích 30 frame được xử lý, track_id, vote_fraction và needs_review. Chỉ ra một tình huống có thể tách/ghép track sai; nếu không tìm thấy, mô tả case thử thêm, không bịa lỗi đã quan sát.

**Đạt:** không gọi events là số xe thật khi chưa đối chiếu.

## Lab 8 — ONNX và fidelity

**Đọc:** Bài 06.

```bash
.venv-linux/bin/python export_parseq_encoder.py --input demo_images
```

**Nộp:** export_report.json, shape/opset/hash, feature/logit tolerance, các dòng mismatch nếu có. Không commit model nhị phân.

Chỉ ra một thay đổi preprocessing bạn dự đoán làm lệch feature và cách thử nó trong file thực nghiệm riêng. Thử với nhiều ảnh; không sửa reference để “khớp” đường sai.

**Đạt:** phân biệt checker/fidelity/accuracy/compiler support; không khẳng định NPU đã chạy từ báo cáo này.

## Lab 9 — Phần cứng theo từng tầng

**Đọc:** Bài 07 và docs/DEEPX_UBUNTU_PREP.md.

**Có card:** thu kernel, PCIe, signer/vermagic, kết quả nạp module và status; chỉ chạy inference sau khi các bước trước ổn. Dùng model mẫu đã chuẩn bị.

**Chưa có card:** lập bảng từ log setup hiện có: chứng cứ nào đủ/thiếu, phép thử tiếp theo là gì. Không cố tạo lỗi kernel bằng cách thay firmware/unload driver dùng chung.

**Đạt:** báo đúng mốc đã chạy; hiểu smoke test input giả khác accuracy OCR.

## Lab 10 — Viết bug report có thể tái hiện

**Đọc:** Bài 08. Chọn lỗi đã gặp thật: file path, dependency, patch, bbox hoặc OCR.

**Nộp:** [bug report](../templates/BUG_REPORT.md) đầy đủ, có tối thiểu 2 giả thuyết và phép thử phân biệt. Giữ log gốc cùng expected/actual, không chỉ ảnh chụp terminal bị cắt.

**Đạt:** đồng đội tái hiện được từ môi trường mô tả.

## Lab 11 — Một bản vá nhỏ có regression check

Chọn bug code thật đã khoanh nguyên nhân. Nếu chưa có, người hướng dẫn cung cấp một bản lỗi trên branch thực hành riêng; không làm hỏng nhánh chung.

**Nộp:** patch nhỏ, case fail trước/pass sau, case tốt vẫn chạy, phạm vi chưa kiểm tra. Unit test khi kiểm tra được hành vi hữu ích; không viết test chỉ lặp lại từng dòng implementation.

Dùng `python -m unittest discover -s tests -v` theo interpreter đang chọn để chạy tests repo liên quan. Test pass không thay phần kiểm tra model/hardware cần thiết.

## Lab 12 — Tối ưu có số liệu

**Đọc:** Bài 09. Chọn đúng một giả thuyết như sampling, preview hoặc một stage chậm đã đo.

**Nộp:** [experiment report](../templates/EXPERIMENT.md): baseline, config, input cố định, thời gian/accuracy trước sau, trade-off và quyết định. Tách load/warm-up khỏi thời gian steady-state khi báo loại số đó.

**Đạt:** không đánh đổi accuracy mà không đo, không gọi latency encoder là FPS sản phẩm, và đồng đội chạy lại được.
