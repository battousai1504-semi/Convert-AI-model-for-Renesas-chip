# 04 — PARSeq: ảnh đi thành chuỗi như thế nào?

## Attention ở mức cần dùng

Attention tạo trọng số để một vị trí chọn thông tin từ các vị trí khác. Query hỏi cần gì; key mô tả nội dung để so khớp; value là thông tin được tổng hợp.

```math
\operatorname{Attention}(Q,K,V)=\operatorname{softmax}(QK^T/\sqrt{d_k})V
```

Hãy kiểm tra shape của Q,K,V trước khi cố nhớ công thức. Multi-head tách nhiều không gian biểu diễn rồi kết hợp. Positional information giúp phân biệt vị trí; LayerNorm ổn định biểu diễn; residual connection truyền thông tin qua các khối.

Đọc R10 về scaled dot-product attention, multi-head, Transformer và ViT. Không cần tự viết tất cả để export, nhưng cần biết các phép MatMul, Softmax, LayerNorm đang làm gì.

## Cấu hình encoder của repo

[parseq.yaml](../configs/model/parseq.yaml) dùng patch 4×8, embedding 384, encoder 12 blocks. Với input 32×128:

```math
N=(32/4)(128/8)=128
```

Encoder biến ảnh thành memory `[B,128,384]` ở cấu hình này. Memory chưa là nhãn biển.

Với B=1, FP32, memory có 49,152 phần tử = 196,608 bytes ≈ 192 KiB. Input 1×3×32×128 là khoảng 48 KiB. Chia encoder/decoder sang hai thiết bị có thể phải truyền feature lớn hơn ảnh input; đo chi phí truyền thật, không suy luận NPU luôn nhanh hơn.

## Decoder và tokenizer

Decoder dùng memory ảnh và ngữ cảnh token để tạo logits `[B,L,V]`. L là số bước output, V là vocabulary. BOS mở đầu, EOS kết thúc, PAD cho padding; không phải ký tự in lên biển.

Chế độ AR dùng token trước để dự đoán token sau. Điểm dừng theo EOS và refinement tạo control flow cần giữ nguyên khi chuyển model. Chế độ NAR khác cách suy luận; không đổi decode_ar để dễ export rồi báo là cùng mô hình mà không đánh giá lại.

PARSeq học bằng permutation language modeling để dùng các thứ tự/ngữ cảnh khác nhau, hỗ trợ nhiều cách giải mã. Đây là phần cần đọc bài gốc R11 sau khi đã hiểu encoder/decoder. Không áp các con số accuracy của paper trực tiếp cho biển Việt Nam.

## Preprocessing cần giống nhau

[SceneTextDataModule.get_transform](../strhub/data/module.py) ở baseline:
RGB → resize bicubic theo img_size → ToTensor → Normalize(0.5,0.5).

Với giá trị pixel u trong [0,255], sau bước chuẩn hóa:

```math
x=2u/255-1
```

Nếu NPU compiler/runtime đã thực hiện một phần preprocessing, không áp lặp lại ở host. Phải xác nhận qua cấu hình/export report, không đoán dựa vào tên input.

## Export đang làm gì?

[export_parseq_encoder.py](../export_parseq_encoder.py) export bản copy encoder, tắt fused attention trên bản copy, giữ reference gốc để đối chiếu. Trong hybrid, ORT tạo memory rồi decoder PyTorch giải mã như trước.

Không export vòng lặp decoder hiện tại. Output .onnx hợp lệ chưa xác nhận DX-COM hỗ trợ tất cả operator.

Trên laptop Trực, encoder đối chiếu PASS 6/6 ảnh demo ngày 2026-10-31. Chuỗi “Chevrol” đồng ý giữa hai đường chạy là bằng chứng fidelity trên mẫu đó, không phải đọc đúng nhãn.

## Đọc code theo thứ tự

1. [read.py](../read.py): loading, transform, forward, decode.
2. [model.py](../strhub/models/parseq/model.py): encode, decode, forward.
3. [parseq.yaml](../configs/model/parseq.yaml): patch/width/depth/decoding.
4. Export script: reference versus hybrid và các ngưỡng đối chiếu.

## Tự kiểm tra

Vẽ lại đường đi ảnh → memory → logits → chuỗi. Ghi shape nào tĩnh, chiều nào phụ thuộc batch/giải mã; chỉ ra BOS/EOS xuất hiện ở bước nào.

Giải thích vì sao chỉ đổi hai token IDs trong tokenizer mà không đổi weights head có thể làm output sai. Làm Lab 5 và trả lời câu hỏi về kích thước buffer.
