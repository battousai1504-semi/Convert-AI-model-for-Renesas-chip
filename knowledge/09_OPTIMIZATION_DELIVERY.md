# 09 — Tối ưu bằng số liệu và bàn giao có giới hạn rõ

## Tối ưu cho yêu cầu nào?

Ghi điều kiện sử dụng: ảnh/video/camera, độ phân giải, số luồng, thời gian chờ tối đa, điều kiện sáng và bộ biển cần hỗ trợ. Với nhóm hiện tại, ưu tiên VN và CSV; chưa có ngưỡng nghiệm thu chính thức.

Không cần bịa target rồi coi là yêu cầu công ty. Có thể đưa mục tiêu thử nghiệm nội bộ, ghi rõ chưa được xác nhận.

## Đo từng phần trước

Với luồng đồng bộ, tổng thời gian gần bằng decode + detect + crop/preprocess + OCR + tracking + ghi output + overhead. Nếu các bước chạy pipeline song song thì throughput không còn bằng nghịch đảo tổng đơn giản; còn queue, độ chờ và stage nghẽn.

Ghi:
- warm-up và số lần chạy;
- thời gian load model riêng;
- mean/median/p95, độ phân giải, số biển/frame;
- CPU threads, device, batch, preview bật/tắt, frame sampling;
- phiên bản, model hash, data ID;
- thời gian chuyển buffer CPU/GPU/NPU.

CUDA có tính bất đồng bộ; đo host quanh forward mà không đồng bộ tại ranh giới phù hợp có thể đo thời gian enqueue. Khi thêm synchronize cho profiling phải hiểu nó cũng thay đổi hành vi bất đồng bộ.

Throughput = số lượng hoàn tất / thời gian đo. Latency = thời gian cho một yêu cầu theo ranh giới đã định. p95 hữu ích cho đuôi chậm; số mẫu nhỏ thì ước lượng yếu.

## Luật Amdahl để tránh tối ưu nhầm chỗ

Nếu 60% tổng thời gian nằm ở encoder, encoder nhanh gấp 3 thì:

```math
\mathrm{Speedup}=1/(0.4+0.6/3)\approx1.67
```

Đây là ví dụ giả định, không phải số đo của PARSeq/DX-M1. TOPS chip không bao gồm mọi chi phí của ứng dụng; không đổi TOPS thành FPS OCR theo một hệ số cố định.

## Thử một thay đổi mỗi lần

| Thử nghiệm | Lợi ích có thể có | Chỉ số dễ bị ảnh hưởng |
| --- | --- | --- |
| Giảm input detector | Nhẹ compute | Recall biển nhỏ |
| Lấy mẫu thưa video | Ít inference | Mất xe nhanh, đứt track |
| OCR theo track/chọn crop tốt | Giảm OCR lặp | Chọn nhầm crop/chuỗi cũ |
| Batch nhiều crop | Tăng throughput | Chờ gom batch, RAM, latency |
| INT8 | Giảm compute/buffer tùy backend | Fidelity, accuracy |
| Chuyển encoder sang NPU | Giảm thời gian encoder | Chi phí truyền feature và decoder còn lại |
| Giảm preview/ghi ảnh | Giảm IO | Khác cấu hình sản phẩm cần bàn giao |

Đây là giả thuyết cần đo, không phải khẳng định mọi mục đều tốt hơn. Dùng [experiment template](templates/EXPERIMENT.md), giữ baseline và dữ liệu đánh giá cố định.

## Phát triển backend có hợp đồng

Với một adapter DX-RT tương lai, mô tả input/output, preprocessing ai chịu trách nhiệm, layout/dtype, quyền sở hữu buffer, device và lỗi. Không giấu fallback sang CPU trong khi log nói NPU.

Một frame lỗi khác frame không có biển. Runtime exception khác OCR đọc chuỗi rỗng. CSV cần thể hiện status/error phù hợp để downstream không hiểu sai.

## Bàn giao một mốc

- Lệnh setup/run từ clone sạch và dữ liệu mẫu có quyền sử dụng.
- Phiên bản, config, model hash và vị trí artifact.
- Output schema: đơn vị thời gian, chuẩn hóa text, ý nghĩa event.
- Kết quả fidelity/accuracy/performance đúng với môi trường đã chạy.
- Known issues và cách thu log.
- Một đồng đội chạy lại độc lập.

Branch chạy được trên máy người viết là một mốc phát triển, chưa tự thành sản phẩm thương mại. Các điều kiện lỗi chưa xử lý phải được ghi rõ để người dùng sau biết phạm vi.

## Hướng đi sâu sau khóa nền tảng

1. C++ và quản lý bộ nhớ để đọc runtime/API và tối ưu buffer.
2. Computer architecture: cache, memory bandwidth, DMA, PCIe và CPU scheduling.
3. Compiler: graph rewriting, constant folding, shape inference, partitioning và operator lowering.
4. ML cho OCR: dữ liệu/domain shift, fine-tuning, calibration confidence, distillation.
5. Hệ thống video: timestamps, queues/backpressure, nhiều camera, phục hồi lỗi.
6. Thiết kế benchmark và regression datasets cho mỗi camera/khách hàng.

Chọn một hướng sau khi đã hoàn thành một bản vá độc lập và một thí nghiệm tối ưu có số đo. Làm Lab 12.
