# DX-M1: triển khai encoder PARSeq và ALPR tự động

Mốc đã chạy trên card thật: **2026-11-01**. Đây là prototype có bằng chứng chạy lại,
chưa phải sản phẩm đã đánh giá đủ để giao khách hàng.

## 1. Kết quả đã đo

| Phép thử | Kết quả |
| --- | --- |
| PCIe / driver / runtime | DX-M1 M.2 Rev 1.0, 1ff4:0100, Gen3 x4; Secure Boot bật, nạp driver thành công |
| Phiên bản | DX-COM 2.4.1; DX-RT 3.4.2; RT driver 2.6.0; PCIe driver 2.5.0; FW card 2.7.0 |
| Full encoder INT8 | 1 NPU group, 0 CPU groups; input float32 [1,128,96], output float32 [1,128,384] |
| OCR trên 6 demo crops | Text giống PyTorch 6/6, không lỗi; feature cosine 0.971733–0.989557 |
| Ảnh toàn cảnh VN | CPU và DX-M1 đều đọc 30A33918; 1 detection / 1 event |
| Video VN | Mỗi backend: 30 frame lấy mẫu, 30 detections, 1 event; toàn bộ text 59V184625 |
| Video encoder calls | 61 = 30 frame × 2 dòng + 1 warmup |

Ảnh demo crop cũng dùng để calibration. Đồng thuận CPU/NPU không phải độ chính xác
so với ground truth trên tập test độc lập. 30 frame cùng biển không phải 30 mẫu độc lập.
Chưa công bố FPS toàn pipeline. Preview dùng frame lấy mẫu, timestamps theo FPS danh
nghĩa; số events không phải phép đếm xe duy nhất đã kiểm chứng.

## 2. Phân chia xử lý

Detector ONNX Runtime CPU tìm biển trong ảnh/frame. CPU crop/chỉnh phối cảnh/tách
một hoặc hai dòng, resize từng dòng RGB về 128×32 và pack thành 128 patch, mỗi patch
96 giá trị float32 thô [0,255]. DX-M1 chạy projection và các block encoder; memory
float32 trả về decoder PyTorch giữ nguyên autoregression/refinement. CPU gộp sự kiện,
vẽ preview và ghi detections.csv, frames.csv, events.csv, summary.json.

`--device cpu` trong `alpr_auto_dxm1.py` chọn **decoder CPU**. Encoder vẫn DX-M1.
Các lần chạy đã xác minh ở đây dùng decoder CPU. Baseline `alpr_auto.py` vẫn chạy PC.
Adapter kiểm tra hash model, checkpoint và runtime IO; lỗi NPU không tự đổi sang encoder CPU.

Đầu encoder RGB/Conv trước đó compile được nhưng text sai. Workaround hiện tại thay
patch Conv bằng MatMul tương đương, fold normalization vào weights/bias, pack patch
trên CPU. ONNX CPU đối chiếu trước compile. Đây chưa phải kết luận nguyên nhân gốc
của lỗi compiler/runtime/firmware. Giữ các script probe/audit để nhóm dò tiếp.

## 3. Môi trường của lần chạy

Laptop Acer PHN16-71 x86_64, Ubuntu 26.04.1, kernel 7.0.0-38-generic, RAM khoảng
30 GiB. Đây là cấu hình đã thử; không suy ra mọi kernel/Ubuntu đều được hỗ trợ.
Xem [chuẩn bị Ubuntu](DEEPX_UBUNTU_PREP.md) và [setup hai OS](SETUP_DUAL_BOOT.md).
Không chia sẻ venv Windows/Linux. Không đổi Python hệ thống.

Ứng dụng dùng `.venv-linux` Python 3.10. Cài DX-RT wheel vào môi trường đó sau khi
cài SDK; dùng --no-deps để giữ numpy 1.26.4 của baseline:

```bash
.venv-linux/bin/python -m pip install --no-deps \
  /usr/share/libdxrt-bin/python/dx_engine-3.4.2-cp310-cp310-manylinux_2_27_x86_64.whl
.venv-linux/bin/python -m pip check
sudo modprobe dx_dma
sudo modprobe dxrt_driver
dxrt-cli -s
```

Driver cài thủ công cho kernel hiện tại; đổi kernel cần build/ký/cài lại. Không đưa
MOK private key lên Git. Firmware trên card là 2.7.0; kho dx_fw khác phiên bản không
có nghĩa card đã được flash. Không có bước flash trong mốc này.

Compiler dùng Docker `onekiwi/dxcom:2.4.1-cpu` (DX-COM 2.4.1, Torch 2.3.1 CPU),
tách khỏi môi trường app. Dockerfile lưu tại docker/dxcom/Dockerfile:

```bash
sudo docker build -t onekiwi/dxcom:2.4.1-cpu -f docker/dxcom/Dockerfile .
```

Máy hiện tại có root chỉ 19 GiB. Docker data-root đã đặt ở
`/home/onekiwi-container-data/docker`, containerd root ở
`/home/onekiwi-container-data/containerd`; không chỉ chuyển một trong hai.
DX-COM yêu cầu filesystem root trong container còn hơn 8 GiB.
Trên máy đã có image chạy được không cần build lại.

## 4. Chạy lại từ source

Xuất full encoder ONNX với `export_parseq_encoder.py --input demo_images`; lưu đường
dẫn artifacts encoder vừa tạo. Compile raw-patch model và kiểm tra OCR:

```bash
.venv-linux/bin/python export_parseq_encoder.py --input demo_images
# Thay YOUR_EXPORT bằng tên thư mục vừa in ra, không lấy run embedding/RGB cũ.
ONEKIWI_ENCODER_ONNX=artifacts/YOUR_EXPORT/parseq_encoder.onnx \
  bash run_dxm1_full_encoder_probe.sh
```

Runner in `FULL encoder run: artifacts/...`; dùng đường dẫn đó cho `--dx-run` hoặc
tham số thứ hai của smoke runner. Trên máy Trực run đã qua test là
`artifacts/dxm1_full_encoder_mm_20261101_134233_21170`.

```bash
bash run_vn_dxm1_smoke.sh
# Chạy với model vừa compile trên máy khác:
bash run_vn_dxm1_smoke.sh inputs/your_source.mp4 artifacts/YOUR_FULL_RUN
# Chỉ chạy ALPR DX-M1, không chạy lại CPU baseline:
.venv-linux/bin/python alpr_auto_dxm1.py \
  --dx-run artifacts/YOUR_FULL_RUN --source inputs/your_source.mp4 \
  --device cpu --max-frames 30
```

`inputs/your_source.mp4` phải là file tồn tại. Smoke mặc định tự tải ảnh VN và kiểm
tra hash; với nguồn tùy chọn, phải kiểm tra có detections thực tế trước khi kết luận.
Các đường dẫn YOUR_* phải được thay bằng đường dẫn trên máy đang dùng.

## 5. Mẫu video đã thử và provenance

Ảnh: [3.jpg tại commit cố định](https://github.com/trungdinh22/License-Plate-Recognition/blob/0b08d1ad9d3da5b0569243a77f068138678f516b/test_image/3.jpg),
SHA256 `8345ddb7c0124ad6e823f5ea77544af44bd4fe7636f358035ad53c10d2c1afbd`,
nhãn kiểm tra trực quan đơn lẻ `30A33918`.

Video: [video1.mp4 tại commit cố định](https://github.com/ddkhoa2605/LicensePlateDetection/blob/f1c732e2270becc44df1de6529372f63ef251076/Video_Test/video1.mp4),
SHA256 `c94e8b3a90f2cfdcd294dd00c24fcc3d917c0b7adee5cdec9febd3331f99020d`.
Khung đầu có biển hai dòng `59-V1 / 846.25`; chưa gán nhãn toàn video.
Chỉ lưu recipe tải, không đưa ảnh/video/model vào Git.

```bash
mkdir -p inputs/vn_dxm1_smoke
curl -fL --retry 3 \
  'https://raw.githubusercontent.com/ddkhoa2605/LicensePlateDetection/f1c732e2270becc44df1de6529372f63ef251076/Video_Test/video1.mp4' \
  -o inputs/vn_dxm1_smoke/video1.mp4
printf '%s\n' \
  'c94e8b3a90f2cfdcd294dd00c24fcc3d917c0b7adee5cdec9febd3331f99020d  inputs/vn_dxm1_smoke/video1.mp4' \
  | sha256sum -c -
bash run_vn_dxm1_smoke.sh inputs/vn_dxm1_smoke/video1.mp4 artifacts/YOUR_FULL_RUN
```

## 6. Bằng chứng Git và tests

Trên máy Trực, collector mặc định đọc đúng các run đã thành công:

- artifacts/dxm1_full_encoder_mm_20261101_134233_21170
- results/dxm1_full_ocr_20261101_134253_446327
- results/vn_dxm1_smoke_20261101_135749_21703 (ảnh)
- results/vn_dxm1_smoke_20261101_140152_21831 (video)

```bash
python3 scripts/collect_dxm1_evidence.py
.venv-linux/bin/python -m unittest discover -s tests -p '*dxm1*.py' -v
```

Collector chỉ đọc artifacts/reports, kiểm tra hash, đối chiếu CSV/JSON và ghi
`docs/evidence/dxm1_20261101.json`. Không inference, reset hay flash. Nếu chạy mốc
mới, truyền --run/--ocr/--photo/--video/--output theo đường dẫn mới; không giả lập
report còn thiếu. File evidence cũ không bị ghi đè.

Tests kiểm tra contract/adapter/helpers bằng stub và ONNX; tests không thay thế
log NPU thật. Checkpoint chuẩn bị Git lưu riêng kết quả tests ở
`docs/evidence/dxm1_helper_tests_20261101.txt`.

Các helper được giữ ở root để tương thích imports hiện tại. Entry point chính là
alpr_auto_dxm1.py; probe_dxm1_embedding.py và audit_* phục vụ chẩn đoán, không phải
pipeline giao khách. Run RGB/embedding cũ không phải model full encoder đang dùng.

## 7. Nguồn chính và bước tiếp

- [PARSeq](https://github.com/baudm/parseq)
- [DEEPX suite](https://github.com/DEEPX-AI/dx-all-suite)
- [DX-RT source tại checkout đã dùng](https://github.com/DEEPX-AI/dx_rt/tree/ed4e8fdb4166e069176870f85a43b10dcbbf35eb)
- [DX-COM tại checkout đã dùng](https://github.com/DEEPX-AI/dx-compiler/tree/78a3ebf12ef5407d9dda759776cc74909a95094c)

Tiếp theo: tập VN có nhãn và tách holdout, phân loại lỗi detector/OCR/tracking, đo
thời gian từng bước và toàn pipeline. Sau đó mới tối ưu, chuyển thêm thành phần
nếu có lợi và tích hợp Renesas. Không suy ra chất lượng thương mại từ smoke test.
