# Chuẩn bị DEEPX DX-M1 trên Ubuntu

Cập nhật: 2026-11-01. Card đã được nhận qua PCIe, nạp driver dưới Secure Boot và chạy inference thành công. ALPR dùng encoder DX-M1 + decoder CPU đã qua smoke test ảnh/video VN. Xem [hướng dẫn triển khai và bằng chứng](DXM1_DEPLOYMENT.md); các bước dưới lưu lại cách chuẩn bị máy.

## 1. Cấu hình đã ghi nhận

| Mục | Giá trị |
| --- | --- |
| Host | Acer Predator Helios Neo 16 PHN16-71, x86_64 |
| OS / kernel | Ubuntu 26.04.1 LTS / 7.0.0-38-generic |
| RAM báo bởi free | Khoảng 30 GiB |
| Secure Boot | Enabled |
| Python ứng dụng | 3.10 trong môi trường riêng, không thay Python hệ thống 3.14 |
| PC OCR | PARSeq CPU chạy được; encoder ONNX khớp PyTorch 6/6 ảnh demo |
| Compiler | DX-COM 2.4.1, target M1 |
| Runtime | DX-RT / dx_engine 3.4.2, import API và pip check OK |
| Driver | RT driver 2.6.0; PCIe driver 2.5.0 theo log build |
| Driver đã cài | dx_dma và dxrt_driver trong /lib/modules/7.0.0-38-generic/updates/ |
| Chữ ký | MOK đã enroll; modinfo có signer, vermagic đúng kernel |
| Phần cứng | DX-M1 M.2 Rev 1.0, PCIe 1ff4:0100, Gen3 x4, LPDDR5 3.92 GiB, firmware thực tế 2.7.0 |

Đã nạp thành công dx_dma và dxrt_driver khi Secure Boot bật; dxrt-cli nhận Device 0: M1. Firmware trên card là 2.7.0; phiên bản kho dx_fw trong SDK không phải phiên bản firmware đã flash.

## 2. Thư mục và môi trường

```text
~/OneKiwi/
  renesas-alpr/                     # repo nhóm
    .venv-linux/                   # PARSeq, requirements PC
  dx-all-suite/                    # SDK riêng, ngoài repo nhóm
    dx-compiler/venv-dx-compiler/   # DX-COM
    dx-runtime/venv-dx-runtime/     # DX-RT Python API
  models/
    mobilenetv2_224x224.dxnn
  tmp/
```

Không dùng chung venv giữa Windows/Linux hoặc giữa PARSeq/compiler/runtime.
Runtime hiện cài NumPy 2.2.6 trong venv riêng; baseline PARSeq ghim NumPy 1.26.4.

## 3. Các phiên bản SDK đã clone

Không tự update submodule riêng lẻ trong lúc tái hiện setup. Các SHA dưới đây đến từ `git submodule status --recursive` trên máy Trực.

| Repository | Commit |
| --- | --- |
| dx-compiler | 78a3ebf12ef5407d9dda759776cc74909a95094c |
| dx-modelzoo | f620eae136b40b78b617b6acc7c19ba2d0855215 |
| dx-runtime | 28dde435c6260cee196c5a73d7ca7adde3f08ffd |
| dx-runtime/dx_app | 01b77271c3cf9f59c3440a953fe56d99adfbda12 |
| dx-runtime/dx_fw | bec4e795957bd5fce2ceb87f6147ff4269254440 |
| dx-runtime/dx_rt | ed4e8fdb4166e069176870f85a43b10dcbbf35eb |
| dx-runtime/dx_rt_npu_linux_driver | 7074748e7104f470b02f517583abba652b3f05fa |
| dx-runtime/dx_stream | dd42c6a91fd7ca0253737c8f89e45e29e71dd133 |

Có mã nguồn/ảnh firmware trong SDK không có nghĩa đã cập nhật firmware trên card. Chưa ghi nhận thao tác flash.

## 4. Kiểm tra phần mềm đã cài

Từ thư mục SDK:

```bash
cd "$HOME/OneKiwi/dx-all-suite"
./dx-compiler/venv-dx-compiler/bin/dxcom --version
dxrt-cli -h
./dx-runtime/venv-dx-runtime/bin/python -m pip check
./dx-runtime/venv-dx-runtime/bin/python -c "import dx_engine; print(dx_engine.__file__)"
modinfo -F signer dx_dma
modinfo -F vermagic dx_dma
modinfo -F signer dxrt_driver
modinfo -F vermagic dxrt_driver
```

Đã kiểm tra các lệnh trên bằng log Trực cung cấp. Chưa có kết quả `dxrt-cli -s` trên card.

Compiler được cài bằng `pip install dx-com==2.4.1` trong venv riêng.
Runtime được cài từ SDK đã clone:

```bash
./dx-runtime/install.sh --target=dx_rt --sanity-check=n
```

Tùy chọn này chỉ chọn runtime và bỏ kiểm tra phần cứng khi chưa có card; không chọn driver hoặc firmware. Không cần chạy lại trên máy đã cài thành công.

## 5. Driver và Secure Boot

Các bước đã thực hiện:

1. Build bằng `modules/build.sh -j 4` với headers của kernel đang chạy.
2. Kiểm tra `/var/lib/shim-signed/mok/MOK.der` đã enroll bằng `mokutil --test-key`.
3. Ký hai file `pci_deepx/dx_dma.ko`, `rt/dxrt_driver.ko` bằng `scripts/sign-file sha256`, dùng khóa riêng tương ứng trên máy.
4. Cài bằng `sudo ./build.sh -c install`, chạy `sudo depmod -a`.
5. Kiểm tra đường dẫn, signer và vermagic của module đã cài.

Đây là cài thủ công theo kernel, chưa phải setup DKMS tự build/ký lại.
Nếu kernel thay đổi, kiểm tra/build/ký/cài lại cho kernel mới.
Không trộn cài thủ công với gói driver DKMS mà chưa xử lý xung đột theo tài liệu hãng.

Không commit/copy khóa MOK riêng vào Git hoặc gửi nội dung khóa. Chứng chỉ enroll của máy khác không mặc nhiên được tin cậy trên máy này.

## 6. Model mẫu đã tải

Nguồn: manifest của dx_app tại SHA đã ghi ở trên.

- URL: https://sdk.deepx.ai/modelzoo/dxnn/2_4_0/mobilenetv2_224x224.dxnn
- File: `~/OneKiwi/models/mobilenetv2_224x224.dxnn`
- Dung lượng theo curl: khoảng 7.30 MiB.
- SHA-256 do máy Trực tính sau khi tải:
  `58116a5fb402c4ceebda4d59e0580b2dc31cc831c3da8a7042fe60edc9c61e35`

Hash này ghi nhận file đã tải, không phải checksum nhà cung cấp công bố độc lập.

Tải lại khi cần:

```bash
mkdir -p "$HOME/OneKiwi/models"
curl -fL --retry 3 \
  "https://sdk.deepx.ai/modelzoo/dxnn/2_4_0/mobilenetv2_224x224.dxnn" \
  -o "$HOME/OneKiwi/models/mobilenetv2_224x224.dxnn"
sha256sum "$HOME/OneKiwi/models/mobilenetv2_224x224.dxnn"
```

## 7. Sau khi gắn card

Tắt nguồn hoàn toàn và thao tác lắp theo hướng dẫn service của máy; kiểm tra khe M.2, khoảng hở và tản nhiệt phù hợp module. Chỉ khởi động sau khi lắp/cố định xong.

Kiểm tra kernel và PCIe:

```bash
uname -r
lspci -nn -d 1ff4:
```

Nếu kernel vẫn đúng bản đã cài driver và thiết bị DEEPX xuất hiện, thử nạp module:

```bash
sudo modprobe dx_dma
sudo modprobe dxrt_driver
dxrt-cli -s
```

Dừng tại lỗi đầu tiên và giữ log. Không diễn giải việc không có thiết bị khi chưa gắn card là lỗi cài runtime. Không flash firmware để thử sửa lỗi khi chưa xác định nguyên nhân/phiên bản phù hợp.

Nếu nạp module lỗi:

```bash
sudo journalctl -k -b -n 120 --no-pager
```

Nếu runtime nhận M1 và không báo lỗi tương thích, có thể thử ví dụ synchronous của chính SDK:

```bash
cd "$HOME/OneKiwi/dx-all-suite"
./dx-runtime/venv-dx-runtime/bin/python \
  dx-runtime/dx_rt/examples/python/run_sync_model.py \
  --model "$HOME/OneKiwi/models/mobilenetv2_224x224.dxnn" \
  --loops 10
```

**Đã chạy thành công trên card của nhóm:** MobileNetV2 10 vòng, trung bình 5.763 ms trong lần thử ngắn. Ví dụ dùng input giả điền 0; con số này không phải accuracy phân loại hay FPS pipeline ALPR.

Ghi lại: thiết bị PCIe, kernel, firmware trên card, phiên bản driver/runtime, model/hash, kết quả chạy và log lỗi nếu có.

## 8. Chuyển sang ALPR của nhóm

`alpr_auto.py` hiện dùng detector ONNX Runtime CPU và PARSeq PyTorch CPU/CUDA.
Cài SDK hoặc tải MobileNetV2 không tự chuyển ứng dụng sang NPU.

Encoder PARSeq đã compile và chạy trên DX-M1 qua phép thay đầu patch projection bằng MatMul tương đương. Dùng alpr_auto_dxm1.py cho encoder NPU + decoder PyTorch CPU; alpr_auto.py vẫn là baseline PC. Xem [DXM1_DEPLOYMENT.md](DXM1_DEPLOYMENT.md) để compile, kiểm tra fidelity và chạy ảnh/video. Detector vẫn ONNX Runtime CPU; chưa tích hợp Renesas.

## Tài liệu nguồn

- [DEEPX environment setup](https://github.com/DEEPX-AI/dx-all-suite/blob/main/docs/source/02_Setting_Up_Environment.md)
- [Compiler installer 2.4.1 checkout](https://github.com/DEEPX-AI/dx-compiler/blob/78a3ebf12ef5407d9dda759776cc74909a95094c/install.sh)
- [Driver instructions at tested source](https://github.com/DEEPX-AI/dx_rt_npu_linux_driver/blob/7074748e7104f470b02f517583abba652b3f05fa/README.md)
- [Model manifest](https://github.com/DEEPX-AI/dx_app/blob/01b77271c3cf9f59c3440a953fe56d99adfbda12/scripts/modelzoo_manifest.json)
- [Synchronous inference example](https://github.com/DEEPX-AI/dx_rt/blob/ed4e8fdb4166e069176870f85a43b10dcbbf35eb/examples/python/run_sync_model.py)
- [Linux module signing](https://docs.kernel.org/admin-guide/module-signing.html)
