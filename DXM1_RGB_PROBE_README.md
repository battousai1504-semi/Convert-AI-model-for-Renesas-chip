# PARSeq encoder DX-M1: explicit RGB input probe

Mục tiêu: so sánh OCR PyTorch với encoder chạy trên DX-M1 và decoder PyTorch
giữ nguyên autoregressive stopping/refinement. Đây là phép thử chuyển đổi,
chưa phải bản tích hợp `alpr_auto.py` hay kết quả đo độ chính xác biển số VN.

## Vì sao tạo thêm một bản ONNX?

Model trước đã compile và chạy được trên card, nhưng metadata đầu vào
`PRE_FORMATTER` có `scale: null`, `bias: null`. Hai trường null không cho phép
suy ra công thức biến pixel RGB thành đầu vào float [-1,1] của encoder gốc.

Script tạo bản ONNX mới: input float32 NCHW với giá trị pixel RGB [0,255],
thêm trực tiếp phép `((x/255)-0.5)/0.5` trước encoder. Calibration cũng dùng
pixel RGB thô [0,255]. Trước khi compile, ONNX Runtime so sánh bản mới với
encoder gốc nhận ảnh đã normalize, trên toàn bộ ảnh calibration.

Ở runtime chỉ resize bicubic + RGB, rồi đưa buffer uint8 NHWC vào DX-RT.
Script kiểm tra dtype, shape và số byte thực tế trước khi gửi buffer.
Không normalize thêm ở CPU, không áp scale/zero point phỏng đoán.
Nếu compiler hoặc card không đáp ứng contract này, phép so sánh có thể báo
lỗi/mismatch; cần xem report, không coi compile thành công là OCR chính xác.

## 1. Giải nén vào repo

Các file mới: `probe_parseq_dxm1.py`, tài liệu này và
`tests/test_dxm1_rgb_contract.py`. Không sửa `read.py`, `alpr_auto.py`, encoder
cũ hoặc các driver. Thực hiện các lệnh dưới đây từ thư mục repo.

```bash
cd "$HOME/OneKiwi/renesas-alpr"
```

## 2. Compile bằng container đã cài

Chọn một thư mục output **chưa tồn tại**. Nếu đã chạy rồi, đổi hậu tố `v1`.

```bash
sudo docker run --rm -i \
  --user "$(id -u):$(id -g)" \
  --mount "type=bind,source=$PWD,target=/work" \
  --workdir /work \
  --env PYTHONUNBUFFERED=1 \
  --env OMP_NUM_THREADS=4 \
  --env MKL_NUM_THREADS=4 \
  --env XDG_CACHE_HOME=/tmp/onekiwi-cache \
  --env MPLCONFIGDIR=/tmp/onekiwi-matplotlib \
  onekiwi/dxcom:2.4.1-cpu \
  python3 probe_parseq_dxm1.py compile \
    --encoder artifacts/encoder_20261031_225454_044655/parseq_encoder.onnx \
    --input demo_images \
    --output artifacts/dxm1_rgb_contract_v1
```

Kết quả: `artifacts/dxm1_rgb_contract_v1/contract.json`, bản encoder gốc được
đóng gói cùng để đối chiếu, ONNX RGB mới và `compiled/*.dxnn`.
`contract.json` lưu SHA256 của model, ảnh calibration và kiểm tra wrapper.
6 ảnh demo chỉ là smoke calibration, chưa đại diện cho biển số VN.

## 3. Thêm DX Python API vào đúng venv PARSeq trên host

Lệnh dưới dùng wheel DX-RT 3.4.2 Python 3.10 đã có trong Debian package.
`--no-deps` giữ phiên bản NumPy/PyTorch hiện tại của môi trường PARSeq;
`pip check` sẽ kiểm tra dependency sau khi cài.

```bash
(
    set -e
    .venv-linux/bin/python -m pip install --no-deps \
      /usr/share/libdxrt-bin/python/dx_engine-3.4.2-cp310-cp310-manylinux_2_27_x86_64.whl
    .venv-linux/bin/python -m pip check
    .venv-linux/bin/python -c "import dx_engine; print('DX API in PARSeq venv: OK')"
)
```

## 4. So sánh trên card thật

Sau khi compile thành công và dependency check sạch:

```bash
.venv-linux/bin/python probe_parseq_dxm1.py compare \
  --run artifacts/dxm1_rgb_contract_v1 \
  --input demo_images
```

Có thể chạy lại với `--input inputs` nếu đó là ảnh đã cắt sát MỘT dòng chữ.
Không dùng ảnh toàn cảnh/video ở bước này. Với biển hai dòng, dùng crop riêng
từng dòng để tách lỗi detector/cắt dòng khỏi lỗi encoder.

Các kiểm tra:

1. SHA256 các model khớp manifest.
2. Buffer thực tế đúng uint8 NHWC [1,32,128,3], output float32 [1,128,384].
3. Tiền xử lý ảnh khớp transform của repo.
4. Encoder ONNX gốc khớp checkpoint PyTorch trong ngưỡng export float32.
5. Chạy encoder thật trên DX-M1, kiểm tra shape và giá trị hữu hạn.
6. Đo sai khác feature INT8 bằng cosine, relative L2, MAE, max absolute error.
7. Chạy cùng decoder PyTorch cho feature gốc và feature NPU, đối chiếu text.

Kết quả ở `results/dxm1_compare_<timestamp>/comparison.csv` và `summary.json`.
Exit code 0: toàn bộ text khớp trên tập vừa chạy; exit code 2: có mismatch/error.
Không áp ngưỡng allclose float32 vào feature đã lượng tử hóa INT8. Thống kê
feature vẫn được giữ lại để xem mức sai lệch kể cả khi text khớp.
Text khớp PyTorch không đồng nghĩa đúng nhãn thực tế. `ground_truth_accuracy`
để null. Thời gian từng lời gọi DX-RT có cả overhead/transfer, chưa warmup hay
đo lặp đủ để gọi là benchmark hoặc FPS của ALPR toàn hệ thống.

## Kiểm tra helper không cần card

```bash
.venv-linux/bin/python -m unittest discover -s tests -p test_dxm1_rgb_contract.py -v
```

Test dùng ONNX nhỏ có Tanh và trọng số khác nhau giữa kênh, kiểm tra toàn dải
pixel 0..255 và ảnh đen/trắng, chống double wrapping, phát hiện sai shape/NaN
và feature đảo dấu. Chúng không thay thế so sánh PARSeq trên card thật.

## Cần làm tiếp sau smoke test

- Nếu text/feature lệch: giữ CSV/log, phân biệt input contract và lỗi lượng tử hóa;
  dùng nhiều crop VN đại diện hơn và xem tùy chọn tuning của DX-COM.
- Khi encoder ổn: thêm backend DX-M1 vào auto-ALPR, giữ detector/cắt dòng như baseline.
- Đánh giá ảnh/video có nhãn, thống kê bỏ sót/đọc sai/trùng sự kiện và latency cả pipeline.

## Nguồn API đã đối chiếu

- [DX-COM 2.4.1 Python compile API, DataLoader](https://github.com/DEEPX-AI/dx-compiler/blob/78a3ebf12ef5407d9dda759776cc74909a95094c/source/docs/02_06_Execution_of_DX-COM.md)
- [DX-COM preprocessing integration](https://github.com/DEEPX-AI/dx-compiler/blob/78a3ebf12ef5407d9dda759776cc74909a95094c/source/docs/02_05_JSON_File_Configuration.md)
- [DX-RT Python API implementation, pinned 3.4.2](https://github.com/DEEPX-AI/dx_rt/blob/ed4e8fdb4166e069176870f85a43b10dcbbf35eb/python_package/src/dx_engine/inference_engine.py)

Các test helper được chạy ở môi trường tạo package. Script chưa được chạy với
PARSeq/DX-M1 trong môi trường đó; kết quả trên card phải lấy từ máy thực tế.
