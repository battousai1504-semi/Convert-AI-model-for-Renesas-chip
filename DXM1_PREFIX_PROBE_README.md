# DX-M1: kiểm tra layout và prefix embedding

Thí nghiệm trước: trên cùng DXNN và cùng byte ảnh, Python/C++ cho kết quả bằng
nhau, chạy lặp ổn định, nhưng cosine so với ONNX thấp. Điều này khoanh vùng
khỏi lỗi riêng của Python API; chưa xác định lỗi formatter, compiler/codegen,
runtime chung, firmware hay mức sai số mô phỏng.

## Cách chạy

Giải nén vào root repo, nơi đã có `probe_parseq_dxm1.py`, `audit_dxm1_io.py`,
`.venv-linux`, `demo_images` và image Docker `onekiwi/dxcom:2.4.1-cpu`.

```bash
cd "$HOME/OneKiwi/renesas-alpr"
bash run_dxm1_prefix_probe.sh
```

Mặc định đọc kết quả audit đã biết:
`results/dxm1_io_audit_20261101_125834_387917`.
Nếu thư mục khác, truyền đối số:

```bash
bash run_dxm1_prefix_probe.sh results/THU_MUC_AUDIT_CUA_BAN
```

## Script làm gì?

1. Đọc các cặp `reference_onnx.npy` / `python_dxrt.npy` đã lưu (không dùng pickle).
   So sánh một danh sách cố định các giả thuyết thứ tự token/channel, hoán đổi
   hai trục patch, và channel-block 8/16/32/64. Không sửa các file đặc trưng.
   Các layout này là giả thuyết chẩn đoán, không phải khẳng định về layout DEEPX.
2. Nếu cùng một giả thuyết ngoài identity đạt cosine >=0.98 và tăng >=0.2 trên
   TẤT CẢ mẫu, dừng để xem kết quả. Đây là ngưỡng ưu tiên kiểm tra, không phải
   ngưỡng nghiệm thu, và không tự áp phép biến đổi vào pipeline.
3. Nếu chưa thấy giả thuyết chung như vậy: trích prefix sau positional Add của
   encoder gốc. Nhận diện qua initializer `pos_embed` [1,128,384] và dataflow,
   không đoán node theo số thứ tự. Dừng nếu vị trí cắt mơ hồ hoặc sai shape.
4. Trong container, kiểm tra wrapper chuẩn hóa bằng ONNX Runtime rồi compile
   riêng prefix: EMA, calibration 6 ảnh demo, opt_level=0, CPU, giống baseline.
5. Trên host, chạy lại Python và C++ để so với ONNX prefix tương ứng.

Prefix chứa patch projection, flatten/transpose và positional Add, trước các
khối Transformer. Nó không phải encoder hoàn chỉnh, không chạy với decoder.
Các model/run cũ giữ nguyên. Output nằm ở tên có timestamp:
`artifacts/dxm1_embedding_probe_<timestamp>_<pid>` và một thư mục `_source`.
Kết quả card nằm trong `results/dxm1_io_audit_<timestamp>`.

## Diễn giải có giới hạn

- Nếu prefix đã lệch lớn: ưu tiên phần đầu vào/layout hoặc phép toán/codegen của prefix.
- Nếu prefix khớp tốt nhưng encoder đầy đủ lệch: kiểm tra sâu hơn các khối Transformer.
- Cắt prefix có thể thay đổi fusion, partitioning và thống kê quantization. Đây
  là phép khoanh vùng bằng các graph biên dịch riêng; không chứng minh chính
  xác node nào sai trong binary encoder đầy đủ.
- Các phép này không đo độ chính xác biển số, không phải benchmark FPS, và
  không chứng minh card lỗi hay card hoàn toàn bình thường.

## Kiểm tra đã làm ở môi trường tạo gói

- Cú pháp Python 3.10 và Bash.
- Mô hình ONNX nhỏ: chọn đúng positional Add, không sửa model gốc; kiểm tra số
  học prefix so với mô hình có thêm phép phi tuyến ở sau; từ chối thiếu/mơ hồ điểm cắt.
- Phát hiện đúng dữ liệu thử bị đổi token/channel và không gắn cờ nhầm mẫu identity.

Chưa chạy các script mới trên PARSeq/DX-M1 thật trong môi trường tạo gói.
Kết quả trên card phải lấy từ máy thực tế.
