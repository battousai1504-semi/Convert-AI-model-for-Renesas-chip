# 08 — Một quy trình để tự dò và vá lỗi

## Trước hết: định nghĩa lỗi

“Không chạy” chưa đủ. Ghi:
- Expected: điều gì phải xảy ra, theo hợp đồng nào?
- Actual: điều gì xảy ra, log/ảnh/CSV nào chứng minh?
- Reproduce: lệnh đầy đủ, commit, environment, input tối thiểu.
- Scope: một ảnh, mọi ảnh, CPU, CUDA, ORT hay DX-RT?

Dùng [bug report](templates/BUG_REPORT.md). Giữ một bản input lỗi được phép dùng, và một case đang chạy tốt để đối chứng.

## Vòng lặp debug

1. Tái hiện bằng ít bước nhất, giữ log đầu tiên.
2. Chọn ranh giới thất bại đầu tiên theo Bài 00.
3. Đề xuất 2–3 giả thuyết có thể phân biệt.
4. Chọn phép thử ít thay đổi nhất để loại giả thuyết.
5. Khi có bằng chứng nguyên nhân, sửa phạm vi nhỏ.
6. Chạy lại case lỗi và case tốt liên quan.
7. Ghi rõ nguyên nhân, thay đổi, kiểm tra và giới hạn; commit.

Không sửa cùng lúc dependency, weights, threshold, preprocessing và decoder. Khi output đổi sẽ không biết yếu tố nào gây ra.

## Bảng triệu chứng của chính dự án

| Triệu chứng | Kiểm tra trước | Vì sao |
| --- | --- | --- |
| FileNotFoundError | cwd, tên/đuôi, đường dẫn resolve, is_file | Chưa mở được ảnh thì chưa tới inference |
| No module named torch | sys.executable và pip của cùng interpreter | Thường cài vào venv khác |
| CUDA unavailable | torch build, GPU/driver, torch.cuda.is_available | Có NVIDIA-SMI không đồng nghĩa wheel có CUDA |
| Flash attention warning nhưng vẫn trả chuỗi | Kết quả, latency và implementation fallback | Warning không tự là crash |
| Crop nhìn sai vị trí | Input shape, ratio/padding, xyxy, ảnh đã xoay chưa | Phân biệt detector và biến đổi tọa độ |
| Crop đúng nhưng OCR sai | RGB, normalize, layout/dòng, domain dữ liệu | Không có một nguyên nhân mặc định |
| ONNX shape mismatch | Input/output contract và config | Cùng số phần tử không đủ |
| ONNX NaN/Inf | Feature hữu hạn, dtype/range từng ranh giới | Không nới tolerance cho dữ liệu vô nghĩa |
| Unsupported operator | Node/opset/shape và đúng phiên bản compiler | Export thành công khác compile thành công |
| Một xe nhiều events | Detection gap, bbox IoU, sampling, max gap | Track ID không phải danh tính thật |
| Push up-to-date nhưng thiếu file | git status, staged diff, commit log | File chưa commit không được push |
| Patch does not apply | Base commit, context, EOL/EOF | Không phải yêu cầu reset hoặc force |

Ví dụ thực tế của nhóm: patch PROJECT_STATUS từng lỗi vì bản nền tạo patch thừa dòng trống cuối file. Kiểm tra trên bản nền đó vẫn PASS; chỉ kiểm tra trên đúng bytes upstream mới tái hiện được lỗi. Bài học: test fixture phải đại diện dữ liệu thực.

## Quan sát thay vì đoán

Ở Python, log metadata không nhạy cảm:
- đường dẫn resolve, kích thước ảnh và bbox;
- shape/dtype/device/min/max;
- phiên bản và hash model;
- provider ORT thực tế;
- raw text, normalized text, status/error.

Không in toàn bộ tensor lớn khi chỉ cần shape. Đặt breakpoint trước thao tác lỗi trong VS Code; xem call stack và locals. Dùng bước qua lời gọi để hiểu dữ liệu, không chạy lại liên tục với cùng giả thuyết.

Nếu cần xem import đến từ đâu:

```bash
.venv-linux/bin/python -c "import sys, torch; print(sys.executable); print(torch.__file__)"
git diff --stat
git diff --cached --stat
```

## Minimal reproduction khác với tắt kiểm tra

Ví dụ lỗi feature:
- Tách chỉ một ảnh và một lệnh encoder.
- Lưu shape/dtype/range trước và sau mỗi backend.
- Đối chiếu cùng input tensor và cùng preprocessing.
- Giữ phần kiểm tra NaN/shape; không bỏ assert chỉ để chương trình đi tiếp.

Với compiler: cố thu nhỏ graph giữ node lỗi; với driver: log kernel + PCIe + module version; với CSV: một row và header thật có thể đủ. Đừng gửi toàn bộ dữ liệu khách hàng chỉ vì report thiếu cấu trúc.

## Một bản vá tốt

Ví dụ lỗi decode bbox do quên trừ padding: sửa ở phép đổi tọa độ, không cộng bù một hằng riêng cho một ảnh.

Cần có:
- Case trước sửa thất bại (ví dụ ảnh tỉ lệ rộng có padding dọc).
- Sau sửa tọa độ về đúng ảnh gốc và clip đúng.
- Case tỉ lệ vuông/ảnh nhỏ/biên vẫn đúng.
- Không thay output schema hoặc device âm thầm.

Nếu lỗi chưa rõ, kết quả đúng là “chưa xác định; đã loại trừ A/B, cần C”, không phải một bản vá may mắn.

Đạt khi một thành viên khác tái hiện được lỗi, hiểu nguyên nhân và xác nhận bản vá từ ghi chú của bạn. Làm Lab 10–11.
