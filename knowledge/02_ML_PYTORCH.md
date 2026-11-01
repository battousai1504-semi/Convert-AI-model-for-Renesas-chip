# 02 — ML và tensor trước khi học Transformer

## Model học gì?

Input x đi qua một hàm có tham số θ, sinh output fθ(x). Trong supervised learning, dữ liệu có nhãn y; loss đo mức sai, optimizer cập nhật θ để giảm loss trên train set.

- Training: tính forward, loss, gradient, cập nhật weights.
- Validation: chọn thiết lập/model bằng dữ liệu riêng.
- Test: đánh giá cuối với dữ liệu không dùng để chọn model.
- Inference: dùng weights đã có để dự đoán; không tự học từ các ảnh vừa đọc.

Pretrained nghĩa là đã học từ một bộ dữ liệu trước. Fine-tune là tiếp tục cập nhật weights trên dữ liệu phù hợp nhiệm vụ. Hai việc không giống sửa regex hoặc đổi preprocessing.

Data augmentation thay đổi mẫu train có kiểm soát; không tự động cải thiện mọi điều kiện. Augmentation không phù hợp có thể xóa nét ký tự quan trọng.

## Sáu câu hỏi cho một tensor

1. Shape: số chiều và kích thước từng chiều?
2. Dtype: uint8, float32 hay int8?
3. Layout: HWC, CHW, NCHW hay NTC?
4. Range: 0…255, 0…1 hay −1…1?
5. Device: CPU, CUDA hay buffer cho NPU?
6. Semantics: pixel, feature, logits hay xác suất?

Cùng shape không có nghĩa cùng nội dung. `reshape` không đổi vị trí kênh như `permute`. Ép float32 sang int8 bằng cast không phải lượng tử hóa đúng.

Ví dụ nhỏ để chạy bằng interpreter ứng dụng:

```python
import torch

image = torch.zeros(32, 128, 3, dtype=torch.uint8)
chw = image.permute(2, 0, 1).float() / 255
batch = chw.unsqueeze(0)
normalized = (batch - 0.5) / 0.5
print(normalized.shape, normalized.dtype, normalized.device)
print(normalized.min().item(), normalized.max().item())
```

Kỳ vọng: shape `[1,3,32,128]`, float32, CPU, min=max=−1 vì ảnh ban đầu toàn 0. Đây chỉ là minh họa tensor; preprocessing thực tế còn resize RGB theo repo.

## Logits, softmax và nhãn

Logits là điểm số trước chuẩn hóa. Với vector z, softmax chuyển thành phân bố:

```math
p_i = \frac{e^{z_i}}{\sum_j e^{z_j}}
```

Chọn argmax để lấy lớp/token có điểm lớn nhất. Softmax cao không bảo đảm đúng ngoài phân bố train. Với OCR, model dự đoán token theo vị trí; tokenizer mới biến token IDs thành chuỗi.

Cross-entropy thưởng cho xác suất của nhãn đúng. Không cần tự viết trainer PARSeq ngay; trước hết phải hiểu label mapping, padding và EOS.

## eval() và inference_mode()

`model.eval()` đổi hành vi các lớp như dropout/batch normalization khi có. `torch.inference_mode()` tắt theo dõi autograd cho tác vụ inference; không thay thế `eval()`. `read.py` dùng cả hai.

`model.to("cuda")` chuyển model; input phải ở device tương thích. Chạy CUDA còn cần driver, build PyTorch và GPU phù hợp; con số CUDA trong `nvidia-smi` không phải luôn là CUDA mà wheel PyTorch được build cùng.

## Toán cần học dần

| Kiến thức | Dùng để hiểu |
| --- | --- |
| Vector, matrix multiplication, dot product | linear layer và attention |
| Mean, variance, chuẩn hóa | normalization và thống kê |
| Xác suất có điều kiện, log | confidence và loss chuỗi |
| Đạo hàm, chain rule | backprop và fine-tuning |
| Complexity và bộ nhớ | chiều dài token, batch, latency |

Đọc R05/R06 và các mục toán của R10 theo nhu cầu; không trì hoãn mọi thực hành đến khi học hết toán.

## Tự kiểm tra

Giải thích vì sao:
- Đổi batch 1 thành 8 có thể tăng throughput nhưng tăng bộ nhớ/độ trễ chờ.
- CPU/GPU cùng trả sai một biển vẫn có thể hoàn toàn đồng ý.
- Không có gradient mà quên eval vẫn có thể cho hành vi inference khác.
- Bỏ một chiều bằng squeeze không chỉ là “làm tensor gọn hơn”.

Làm Lab 3; in và giải thích đủ metadata cho ba tensor liên tiếp.
