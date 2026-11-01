# Experiment — một giả thuyết, một thay đổi chính

## Câu hỏi

- Bottleneck/lỗi quan sát:
- Giả thuyết:
- Chỉ thay đổi:
- Tiêu chí quyết định; ai xác nhận yêu cầu nếu có:

## Điều kiện tái hiện

- Commit baseline / candidate:
- OS / kernel / hardware:
- Interpreter, dependencies, SDK/driver/firmware:
- Model URL/version/hash và cấu hình:
- Dataset/version/split, số mẫu, nguồn/nhóm:
- Calibration set ID; có tách test không:
- Lệnh baseline / candidate:
- Warm-up, số lần chạy, sampling, preview, thread, batch:
- Ranh giới timing, đồng bộ device, load model có tính không:

## Kết quả thực

| Chỉ số | Baseline | Candidate | Cách đo / mẫu số |
| --- | --- | --- | --- |
| Exact plate accuracy | Chưa đo | Chưa đo | Ground truth nào |
| CER | Chưa đo | Chưa đo | Số ký tự |
| Recall / precision nếu có | Chưa đo | Chưa đo | Ghép bbox ra sao |
| Latency median/p95 | Chưa đo | Chưa đo | Tầng nào |
| Throughput | Chưa đo | Chưa đo | Frame xử lý / giây đo |
| Lỗi / missed events | Chưa đo | Chưa đo | Cách đối chiếu |
| Bộ nhớ | Chưa đo | Chưa đo | Công cụ và điều kiện |

## Phân tích

- Kết quả có ủng hộ giả thuyết không:
- Case tốt hơn / xấu hơn:
- Có đổi đồng thời điều kiện khác không:
- Sai số/giới hạn số mẫu:
- Quyết định giữ / bỏ / cần thử thêm:
- Artifact/report cục bộ:
- Người chạy lại và kết quả:

Không gọi số đo model mẫu input giả là accuracy/FPS ALPR; không dùng cùng một tập để chọn mọi tham số rồi gọi đó là test độc lập.
