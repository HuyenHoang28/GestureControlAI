# Hand Gesture Recognition Project
# Dự án Nhận diện Cử chỉ Tay

---

## 1. Giới thiệu / Introduction

Dự án sử dụng AI để nhận diện sáu cử chỉ tay cơ bản  từ webcam theo thời gian thực.  
 
This project uses AI to recognize six basic hand gestures from a webcam in real-time.  

---

## 2. Yêu cầu / Requirements

- Python 3.9  
- Thư viện Python:
  + OpenCV
  + Mediapipe
  + scikit-learn
  + numpy
  + joblib
  + ipywidgets

Cài đặt thư viện / Install required libaries: pip install -r requirements.txt

## 3. Hướng dẫn sử dụng / Instructions

 - Bước 1: Cài các thư viện (có thể dùng lệnh trên trong terminal).
 - Bước 2: Mở file Project_report_G15.ipynb
 - Bước 3: Đến mục 3. Tiến độ giữa kỳ (W8) / 4. Cập nhật kết quả cuối kỳ (W15), chạy (execute) cell code mục "Chương trình" trước và đợi cell chạy xong.
 - Bước 4: Nếu muốn thu thập mẫu data cho các label cử chỉ(có sẵn, hoặc viết thêm vào trong label list), bấm chạy cell có comment "COLLECT DATA MODE" ở đầu, chọn label cử chỉ lấy mẫu, sau đó chọn "Bắt đầu collect" và thực hiện lấy mẫu:
   + SPACE: chụp ảnh lấy mẫu
   + q: thoát quá trình lấy mẫu
 - Bước 5: Sau khi lấy mẫu các label xong, chạy cell có comment "TRAIN MODE" để huấn luyện mô hình. Kết quả huấn luyện sẽ hiển thị ngay bên dưới cell
 - Bước 6: Sau khi huấn luyện model, chạy cell có comment "PREDICT MODE" để chạy mô hình đã huấn luyện, kiểm tra xem mô hình đã được huấn luyện đúng chưa. Ấn q để thoát.