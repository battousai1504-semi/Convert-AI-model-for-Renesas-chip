# Thử thay patch Conv bằng MatMul tương đương

## Bằng chứng từ ZIP ngày 2026-11-01

Nguồn: embedding-prefix run `dxm1_embedding_probe_20261101_131111_19767`,
audit đã bật CPU fallback `dxm1_io_audit_20261101_131653_909563`.
DXNN SHA256: `c08eb07a48c1b03a694e6716f98202d5c95f01ca1bc179de75e0f07a02dd0f07`.

- Prefix có Conv kernel/stride [4,8], flatten/transpose, cộng positional embedding.
- Tính lại bằng NumPy từ trọng số và raw input khớp ONNX: max absolute error
  3.424e-6 (art-01107) và 2.244e-6 (coco-1166773).
- W có kích thước [384,96], rank 96, condition number khoảng 16.72. Dùng
  pseudo-inverse để kiểm tra ngược tín hiệu đầu vào là một phép chẩn đoán có dư
  thừa, không phải dump trực tiếp input nội bộ NPU.
- Cả hai mẫu có 80/128 vector gần bias + positional embedding, sai khác tuyệt
  đối trung bình khoảng 0.0002855. 48 vector còn lại có thể khớp với phép chiếu
  của các patch ảnh gốc; một số vị trí đã bị thay đổi.
- Cosine của các vector còn dữ liệu với phép chiếu patch phù hợp: 0.999983 và
  0.999971. Đây KHÔNG phải cosine toàn prefix, cũng không phải độ chính xác OCR.

Các quan sát hướng việc điều tra vào đường normalize/formatter/patch projection
và cách thực thi, chưa xác định chắc lỗi cụ thể thuộc compiler, runtime hay firmware.
Không thể suy ra card hỏng từ các quan sát này.

## Biểu diễn mới

Patch vector có 96 phần tử: 3 kênh x 4 hàng x 8 cột, thứ tự C,dy,dx.
128 patch đi theo thứ tự patch_y,patch_x, tạo tensor raw pixel float32 [1,128,96].

Với W [384,96], b [384], p là positional embedding:

`y = ((raw_patch / 127.5) - 1) @ W.T + b + p`

tương đương:

`y = raw_patch @ (W.T / 127.5) + (b - sum(W, axis=1)) + p`.

CPU chỉ xếp pixel thành patch. Phép chiếu học được được biểu diễn bằng MatMul,
không dùng Conv [4,8] và các phép normalize riêng ở đầu NPU graph của bản thử cũ.
Các trọng số/bias mới được suy ra đại số từ checkpoint gốc, không train lại
hay hiệu chỉnh theo nhãn; có sai số làm tròn float32 nhỏ.
Đây là thử nghiệm đường thực thi khác, không chứng minh chính xác operator gây lỗi.

## Cách chạy

Giải nén vào root repo, nơi có `probe_parseq_dxm1.py`, `.venv-linux`, demo_images,
image Docker `onekiwi/dxcom:2.4.1-cpu` và prefix run cũ.

```bash
cd "$HOME/OneKiwi/renesas-alpr"
bash run_dxm1_matmul_head_probe.sh
```

Script tạo thư mục mới theo timestamp, so sánh ONNX gốc/mới trên toàn bộ ảnh
calibration trước khi compile. Sau compile, đọc compilation summary để kiểm
tra node projection nằm trong partition NPU; không chấp nhận CPU-only projection.
Sau đó lấy dtype/shape thực tế từ DX-RT. Chỉ gửi dữ liệu nếu runtime có đúng
giao diện float32 raw-patch. Nếu compiler đổi sang uint8 hoặc shape khác, script
dừng và lưu `runtime_io.json`; không tự đoán scale/zero point.

Phép thử hiện tại chỉ dùng embedding-prefix, chưa chạy decoder hay ALPR.
Cosine/relative-L2 được đo so với ONNX prefix gốc. Kết quả ở
`results/dxm1_matmul_compare_<timestamp>/feature_comparison.json`.

## Kiểm chứng ở môi trường tạo gói

Đã chạy phép thay head trên ONNX thật do người dùng gửi, so sánh trên hai raw
input thật, ảnh toàn đen/trắng và ảnh ngẫu nhiên. Tất cả qua ngưỡng rtol=1e-3,
atol=1e-4; max absolute error lớn nhất 6.676e-6. Graph gốc không bị sửa.
Đã kiểm tra cú pháp Python 3.10/Bash và guard từ chối projection chạy trên CPU.
Chưa compile/chạy bản mới trên DX-M1 trong môi trường tạo gói. Cần lấy kết quả
trên máy thật; không có tuyên bố bản workaround này đã sửa thành công.

Nếu prefix mới khớp tốt trên NPU: thử thay cùng head trong encoder đầy đủ rồi
so sánh feature và text với PyTorch. Sau đó mới ghép vào ALPR và đo dữ liệu VN có nhãn.
