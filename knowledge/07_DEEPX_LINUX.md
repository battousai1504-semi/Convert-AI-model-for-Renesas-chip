# 07 — Tự chẩn đoán từ PCIe tới NPU

## Các tầng làm việc khác nhau

| Tầng | Vai trò | Bằng chứng phù hợp |
| --- | --- | --- |
| M.2 / PCIe | Kết nối card với host | lspci có vendor/device phù hợp |
| Kernel module | DMA/giao tiếp và điều khiển | modprobe thành công, module/device xuất hiện |
| Secure Boot / MOK | Kiểm tra chữ ký module | cert enroll + module có signer + kernel thực sự nạp |
| Firmware | Phần mềm điều khiển trên card | dxrt-cli báo version/trạng thái |
| DX-RT | Thư viện/API runtime | model inference hoàn tất trên thiết bị |
| Ứng dụng | Pre/postprocess, detector/OCR, CSV | Output đúng trên protocol end-to-end |

M.2 là hình thức/kết nối module, không đồng nghĩa mọi khe có cùng giao thức/nguồn/khoảng hở. Với laptop cần xác nhận phần cơ khí và tản nhiệt; dữ liệu boot đã kiểm tra chỉ trả lời được phần ổ đĩa/khởi động.

CPU host vẫn đọc video, xử lý ảnh, quản lý buffer, chạy phần chưa chuyển được và ghi output. NPU không thay thế hệ điều hành hoặc toàn bộ CPU.

## Những gì nhóm đã xác nhận

Ở base 068fc2e: DX-COM 2.4.1, DX-RT 3.4.2, build/cài driver cho kernel 7.0.0-38, MOK đã enroll và module có signer. Chưa gắn card để xác minh PCIe, module loading hoặc inference. [Bản ghi setup](../docs/DEEPX_UBUNTU_PREP.md) chứa SHA và lệnh đúng với máy Trực.

Không cập nhật firmware chỉ vì repository có bản mới. Cần xác nhận sản phẩm, phiên bản đang chạy, tương thích SDK và hướng dẫn phục hồi khi cần.

## Thứ tự kiểm tra sau khi gắn card

Các lệnh dưới đây chỉ đọc trạng thái, trừ modprobe là thao tác nạp driver. Chỉ nạp sau khi đã cài driver đúng kernel; dừng tại lỗi đầu tiên.

```bash
uname -r
lspci -nn -d 1ff4:
```

Không thấy PCIe: ưu tiên nguồn/khe/link/enumeration và BIOS/hardware, chưa phải lỗi preprocessing Python.

Có PCIe và đúng kernel đã cài driver:

```bash
sudo modprobe dx_dma
sudo modprobe dxrt_driver
dxrt-cli -s
```

Nếu lỗi, xem thông điệp kernel trước khi sửa:

```bash
modinfo -F filename dx_dma
modinfo -F signer dx_dma
modinfo -F vermagic dx_dma
sudo journalctl -k -b -n 120 --no-pager
```

Tập log này không cần in private key hoặc nội dung chứng thực.

## Giải nghĩa các case

| Hiện tượng | Giả thuyết cần phân biệt |
| --- | --- |
| Module not found | Chưa cài cho kernel đang chạy; depmod/path sai |
| Key was rejected by service | Khóa chưa được tin cậy, chữ ký sai/mất, hoặc chính sách khác; đọc log |
| Invalid module format | vermagic, ABI/symbol/kernel config không khớp; không kết luận chỉ từ tên lỗi |
| PCIe có, không có thiết bị runtime | module/probe/permission/firmware/runtime cần kiểm tra tiếp |
| Import dx_engine OK | Python tìm và import được package; chưa chứng minh card hoạt động |
| Mẫu MobileNet chạy, PARSeq không chạy | Model/config/operator/IO contract hoặc phiên bản khác; chia nhỏ để kiểm tra |
| Sau cập nhật kernel driver mất | Bản cài thủ công gắn kernel cũ; cần build/ký/cài cho kernel mới |

`modinfo signer` chỉ đọc metadata; kernel chấp nhận module khi nạp mới là phép thử thực tế. Bản cài manual không có bảo đảm tự build lại như DKMS đã cấu hình đúng.

## DMA và buffer: học để sửa adapter

DMA cho phép thiết bị trao đổi dữ liệu với bộ nhớ host qua cơ chế driver. Buffer cần đúng kích thước, layout, kiểu và vòng đời. Một view không contiguous hoặc buffer được giải phóng quá sớm có thể gây lỗi khác với model toán học.

Không ép mọi buffer thành uint8 mà không đọc API. Một mảng uint8 có thể là raw bytes chứa dữ liệu theo hợp đồng runtime, hoặc input quantized; phải xem model metadata/API, không suy từ dtype của container.

Ví dụ synchronous của SDK dùng buffer cấp phát và fill 0 cho smoke test. Output/FPS của nó không phải độ đúng OCR hay FPS cả pipeline.

## Bài thực hành

Đọc R15–R17 và làm Lab 9. Nếu chưa có card, phân loại từng log đã có thành “đã chứng minh/chưa chứng minh” và lập thứ tự lệnh dự kiến. Không tự điền PASS phần cứng.
