import numpy as np
import shutil

DATASET_PATH = "landmarks.npy"
BACKUP_PATH = "landmarks_backup.npy"

# Nhập tên label muốn xóa (ví dụ: "A")

label_to_remove = "1_fist"

# Tạo bản sao lưu trước khi xóa
shutil.copy(DATASET_PATH, BACKUP_PATH)
print(f"✅ Đã sao lưu file cũ thành: {BACKUP_PATH}")

# Đọc dữ liệu
data = np.load(DATASET_PATH, allow_pickle=True)
print(f"📦 Số mẫu ban đầu: {len(data)}")

# Lọc bỏ các mẫu có label cần xóa
new_data = [item for item in data if item[0] != label_to_remove]
removed = len(data) - len(new_data)
print(f"🗑️ Đã xóa {removed} mẫu có label '{label_to_remove}'")

# Lưu lại file mới
np.save(DATASET_PATH, np.array(new_data, dtype=object))
print(f"💾 Đã lưu lại file mới: {DATASET_PATH}")
print(f"📦 Số mẫu còn lại: {len(new_data)}")
