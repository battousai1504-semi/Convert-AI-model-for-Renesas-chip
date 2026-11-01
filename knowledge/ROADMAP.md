# Lộ trình học qua sản phẩm thực tế

Mốc thời gian dưới đây là gợi ý 8 tuần, không phải lịch bắt buộc. Có thể kéo dài một chặng nếu chưa đạt đầu ra. Không cần học xong toàn bộ mới được làm việc; ưu tiên kiến thức cho lỗi/nhiệm vụ đang gặp.

Mỗi buổi khoảng 60–90 phút: đọc ngắn, thực hành, rồi tự giải thích không nhìn tài liệu. Mỗi tuần dành một buổi chạy lại bài của đồng đội.

| Chặng | Đọc | Thực hành | Đầu ra có thể review |
| --- | --- | --- | --- |
| 1 | Bài 00–01, R01–R04 | Lab 1–2 | Sơ đồ pipeline, môi trường đúng, phân tích một traceback |
| 2 | Bài 02, R05–R06 | Lab 3 | Giải thích shape/dtype/device và tách được inference/training |
| 3 | Bài 03, R07–R09 | Lab 4 | Ảnh bbox/crop đúng, tính tay letterbox khớp code |
| 4 | Bài 04, R10–R11 | Lab 5 | Bản thuyết minh PARSeq và output encoder |
| 5 | Bài 05, R08 | Lab 6–7 | Báo cáo dữ liệu/metrics và một lỗi tracking có ví dụ |
| 6 | Bài 06, R12–R14 | Lab 8 | Export report, so sánh tensor và kế hoạch calibration |
| 7 | Bài 07, R15–R17 | Lab 9 | Nhật ký phần cứng từng tầng; hoặc phân tích log khi chưa có card |
| 8 | Bài 08–09 | Lab 10–12 | Một lỗi được vá và một tối ưu có baseline, số liệu, giới hạn |

## Điều kiện qua từng chặng

- Giải thích được ý nghĩa, không chỉ đọc lại tên lệnh.
- Chạy lại được từ trạng thái rõ ràng, ghi OS/commit/venv/input.
- Nói được điều gì kết quả chưa chứng minh.
- Đồng đội tái hiện được từ ghi chú.
- Nếu thay đổi code: có trường hợp lỗi trước và kiểm tra sau; chỉ đổi đúng phạm vi.

## Phân công gợi ý

| Mảng phụ trách sâu | Bằng chứng bàn giao |
| --- | --- |
| Detector + hình học | Hợp đồng tensor, bbox, ảnh crop trước/sau |
| PARSeq + export | Shape, preprocessing, fidelity report, operator cần xử lý |
| Video + đánh giá | Bộ nhãn, chính sách metrics, case tracking/event lỗi |
| Linux + DX-M1 | Matrix phiên bản, PCIe, chữ ký/module, trạng thái và inference log |

Không khóa vai trò vào một thành viên. Mỗi người cần tự chạy demo PC và giải thích được ranh giới CPU/NPU.

## Khi không có card

Vẫn làm được môi trường, dataset/nhãn, export, đọc graph, viết báo cáo compiler nếu đã cài và có cấu hình, phân tích log giả lập có ghi rõ. Không điền “PASS phần cứng” thay cho kết quả chưa chạy.

## Khi gặp một lỗi mới

Tạm dừng bài đang học để đi theo Bài 08. Sau khi sửa, ghi bài học vào knowledge base. Nếu nguyên nhân thuộc kiến thức chưa học, quay về đúng phần, không mở thêm nhiều framework chỉ để tìm lệnh chữa nhanh.

## Tự đánh giá năng lực

| Mức | Biểu hiện |
| --- | --- |
| Làm theo | Chạy lại lệnh và thu đúng output |
| Giải thích | Nói rõ input/output, giả định và ý nghĩa log |
| Chẩn đoán | Có giả thuyết, chọn phép thử phân biệt nguyên nhân |
| Sửa độc lập | Viết bản vá nhỏ, kiểm tra hồi quy và đồng đội tái hiện được |
| Thiết kế/tối ưu | Chọn trade-off theo số liệu và yêu cầu sản phẩm |

Mục tiêu trước mắt là đạt “sửa độc lập” trên pipeline của nhóm. Để tiến sâu hơn, mở rộng kiến thức về compiler, kiến trúc máy tính, C++ và tối ưu runtime sau khi đã nắm các ranh giới hiện tại.
