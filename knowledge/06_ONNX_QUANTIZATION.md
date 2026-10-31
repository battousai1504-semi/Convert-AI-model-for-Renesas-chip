# 06 — Chuyển model: giữ ý nghĩa trước khi tối ưu

## Các vai trò

- PyTorch: mô tả model và chạy reference.
- ONNX: graph, operator, weights và metadata.
- ONNX Runtime: thực thi graph trên execution provider được chọn.
- DX-COM: compiler tạo .dxnn theo phần cứng/SDK DEEPX.
- DX-RT: nạp và chạy .dxnn trên hệ thống có card phù hợp.

ONNX không phải driver; ORT CPU không phải DX-M1. Hợp lệ theo ONNX checker không bảo đảm compiler DEEPX hỗ trợ.

## Graph và phiên bản

Node là phép toán; edge/tensor nối các node. Input/output có tên, kiểu, shape. Opset là phiên bản tập operator; không phải tốc độ hay độ chính xác. IR version khác opset. Đổi số opset trong metadata không tự viết lại phép toán đúng nghĩa.

Đọc R12 để hiểu graph; R13 để dùng exporter đúng với PyTorch 2.2 đang ghim. Tài liệu PyTorch mới có API khác, không thay trực tiếp vào baseline.

## Rủi ro của tracing và control flow

Trace ghi lại đường thực thi trên input mẫu. Giá trị tensor chuyển thành bool/Python có thể bị coi là hằng; warning không luôn gây sai, nhưng yêu cầu kiểm tra giả định.

Nhóm đang export encoder có input shape cố định. Decoder có EOS/AR/refinement nên được giữ trong PyTorch. Không trace cả mô hình rồi kết luận tổng quát từ một ảnh chạy được.

Kiểm tra các ảnh có chuỗi ngắn/dài, ảnh khó và layout phù hợp. Test hợp đồng shape riêng; không dùng graph static cho kích thước khác rồi coi lỗi đó là bug runtime.

## Đối chiếu ba tầng

1. Tensor feature: cùng shape, dtype phù hợp, hữu hạn; so sai số tuyệt đối/tương đối.
2. Logits/đường giải mã: có thay đổi số học ảnh hưởng argmax/EOS không?
3. Chuỗi cuối: có bằng nhau không, và có đúng ground truth không?

Hai feature hơi khác có thể cho cùng chuỗi; một token có margin nhỏ có thể đổi nhãn chỉ vì sai số rất nhỏ. Không nới tolerance chỉ để báo cáo PASS; phải giải thích tác động bằng dữ liệu.

Exporter hiện ghi tolerance feature rtol 1e−3 / atol 1e−4 và logits rtol 1e−3 / atol 1e−3. Đó là ngưỡng kiểm tra hiện có, không phải quy tắc cho mọi model.

## Quantization ở mức trực giác

FP32 có nhiều mức biểu diễn; INT8 dùng ít bit hơn, cần ánh xạ theo scale s>0 và zero point z:

```math
q=\operatorname{clip}(\operatorname{round}(x/s)+z),\quad \tilde x=s(q-z)
```

Rounding tạo sai số; clipping làm mất các giá trị ngoài dải. Per-tensor/per-channel chọn mức scale khác nhau; symmetric/asymmetric khác cách dùng zero point. Đừng giả định tất cả biến thể đều được NPU/compiler hỗ trợ.

PTQ: lượng tử hóa sau training, thường dùng calibration để ước lượng thống kê. QAT: đưa hiệu ứng quantization vào training; chỉ cân nhắc khi có dữ liệu và nhu cầu.

R14 là tài liệu học khái niệm qua ONNX Runtime, không phải lệnh chuẩn bị model cho DX-COM. Quy trình quantization của hai công cụ không thể thay thế tùy ý.

## Calibration của encoder

Dùng ảnh dòng chữ giống phân bố triển khai, qua đúng preprocessing đã định nghĩa. Không đưa ảnh toàn cảnh vào calibration của encoder chỉ vì hệ thống cuối xử lý toàn cảnh. Cần phủ góc, kích thước chữ, độ sáng và các ký tự phù hợp.

Calibration thường không cần nhãn để ước lượng activation; đánh giá accuracy sau đó cần nhãn độc lập. Cách đóng gói dữ liệu/input layout phải theo phiên bản DX-COM và graph thực tế.

Không dùng test set để chọn calibration/threshold rồi vẫn báo đó là test độc lập.

## Khi compiler báo unsupported operator

Ghi node/operator/opset/shape, compiler version, config và log. Tạo minimal subgraph nếu cần. Kiểm tra tài liệu support của đúng phiên bản. Một số hướng xử lý cần thử: cách export tương đương, rewrite graph có đối chiếu, phân chia CPU/NPU, hoặc model khác.

Đổi LayerNorm/attention sang công thức khác phải chứng minh tương đương và kiểm tra độ đúng; không sửa graph chỉ để hết lỗi compile.

## Bằng chứng cần lưu

Dùng [experiment template](templates/EXPERIMENT.md): commit, model hash, input/output contract, preprocessing, opset, phiên bản compiler/runtime, config, calibration set ID, sai số feature, accuracy và thời gian.

Làm Lab 8. Mốc DX-COM của PARSeq vẫn chưa được xác minh tại base commit 068fc2e; bài này không cung cấp cấu hình compile giả định là đã chạy.
