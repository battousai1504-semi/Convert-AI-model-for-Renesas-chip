# 01 — Môi trường, Git và Python để tự đọc lỗi

## Mục tiêu

Trước khi sửa một import, phải biết chương trình đang dùng Python nào, cài package ở đâu và chạy từ thư mục nào.

## Shell không phải Python

`$` hoặc `PS …>` là shell; `>>>` là Python REPL. Lệnh `git`, `cd`, `pip` chạy ở shell. Trong REPL, dùng `exit()` để thoát. Mở `python.exe` trong editor chỉ hiển thị dữ liệu nhị phân, không chọn interpreter cho project.

Đường dẫn tương đối tính từ working directory, không nhất thiết là thư mục chứa script. Trong lỗi `inputs/plate_01.png`, hãy kiểm tra file tồn tại và đúng đuôi trước khi sửa model.

```bash
pwd
git status --short
.venv-linux/bin/python -c "import sys; from pathlib import Path; print(sys.executable); print(Path.cwd()); print(Path('inputs/plate_01.png').resolve()); print(Path('inputs/plate_01.png').is_file())"
```

`is_file()` là False chứng minh đường dẫn đang dùng chưa trỏ tới file thường; chưa xác định file bị thiếu, sai tên hay ở thư mục khác.

## Python tối thiểu cần nắm

| Khái niệm | Gặp trong dự án |
| --- | --- |
| list / tuple / dict | danh sách bbox, kích thước, một row CSV |
| function / return | hợp đồng từng bước preprocessing |
| class và trạng thái | model, detector và tracker qua nhiều frame |
| import / module | strhub, torch, helper trong repo |
| exception | file lỗi, shape sai, runtime thất bại |
| pathlib và with | mở/đóng ảnh và file đúng cách |
| argparse | tham số CLI và giá trị mặc định |
| JSON / CSV | báo cáo cấu trúc; kiểu số khác chuỗi |

Đọc R01 theo các mục: control flow → data structures → modules → files → errors. Với class chưa cần học metaclass; hiểu constructor, attribute và method trước.

## Venv và dependency

Mỗi venv có interpreter và site-packages riêng. Dùng `python -m pip` từ đúng interpreter giúp tránh cài nhầm:

```bash
.venv-linux/bin/python -m pip check
.venv-linux/bin/python -c "import torch, numpy; print(torch.__version__); print(numpy.__version__)"
```

`pip check` phát hiện xung đột requirements được khai báo. Nó không kiểm tra GPU, ABI của mọi thư viện, driver hoặc output model. Có “No broken requirements” mà import vẫn lỗi là có thể.

Không đổi Python hệ thống để làm vừa một repo. Không chép venv Windows sang Linux. Khóa phiên bản phục vụ tái hiện; nâng cấp là một thay đổi cần thử trên nhánh riêng.

## Đọc traceback bằng một ví dụ

```text
File ".../read.py", line ..., in main
    image = Image.open(fname).convert("RGB")
FileNotFoundError: ... 'inputs/plate_01.png'
```

Đọc loại exception và thông điệp cuối trước; lần ngược về dòng trong code nhóm. Ở đây thao tác mở file lỗi trước khi inference, nên sửa CUDA hoặc train lại không giải quyết được.

Với `ModuleNotFoundError`: in `sys.executable` và kiểm tra package trong đúng venv.
Với `TypeError`: kiểm tra kiểu/tham số tại lời gọi.
Với `RuntimeError`: đọc đầy đủ thông điệp; đây là loại rộng, không đủ để kết luận nguyên nhân.

## Git có bốn nơi dễ nhầm

Working tree → staging/index → local commit → remote branch.

```bash
git branch --show-current
git status --short
git diff
git diff --cached
git log -3 --oneline
```

- `git add` chọn nội dung cho commit; chưa upload.
- `git commit` có thể thất bại vì chưa cấu hình author.
- `git push` thành công mới cập nhật remote; “Everything up-to-date” không chứng minh file chưa commit đã lên.
- Patch phụ thuộc nội dung nền. `git apply --check` kiểm tra có áp dụng được; không xác minh logic.
- Khi patch lỗi, xem base SHA, nội dung context, EOL và dòng cuối. Không dùng reset để làm lỗi biến mất.
- Git không lưu venv/driver đã cài trên hệ thống. Commit tài liệu tái hiện thay vì upload toàn bộ máy.

Đọc R02 chương 2–3 và R03 về shell; làm Lab 1–2. Đạt khi đồng đội tái hiện được một lỗi môi trường từ ghi chú của bạn.
