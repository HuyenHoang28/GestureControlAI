Hand Gesture Recognition (MediaPipe + scikit-learn)
====================================================

This project captures hand landmarks using MediaPipe and trains a lightweight scikit-learn classifier (RandomForest) to recognize gestures in real-time.

Quick steps:
1. Collect data for each gesture:
   python src/collect_dataset.py --label gesture_0 --out data/landmarks.csv
   (Repeat for gesture_1 ... gesture_5 and append to same CSV)

2. Train model:
   python src/train_model.py --in data/landmarks.csv --out models/gesture_clf.joblib

3. Run real-time detection:
   python src/detect_gesture.py --model models/gesture_clf.joblib

Controls:
- collect_dataset: press 's' to save sample, 'q' to quit
- detect_gesture: press 'q' to quit

Requirements:
- Python 3.9 recommended
- Install dependencies: pip install -r requirements.txt


STT	Tên label (dùng trong code)	Mô tả cử chỉ	Gợi ý khi thu mẫu
1️⃣	thumbs_up	👍 Ngón cái giơ lên (like)	Tay hướng về camera, ngón cái lên rõ
2️⃣	thumbs_down	👎 Ngón cái chỉ xuống	Giống like nhưng xoay ngược tay
3️⃣	fist	✊ Nắm tay lại	Giữ bàn tay nắm chặt, không duỗi ngón
4️⃣	open_hand	🖐️ Bàn tay mở, các ngón duỗi	Tay mở thẳng, lòng bàn tay hướng vào camera
5️⃣	peace	✌️ Giơ 2 ngón (index + middle)	Hai ngón tách nhau, hướng về camera
6️⃣	okay	👌 Ngón cái và ngón trỏ chạm nhau thành vòng tròn	Giữ ổn định, camera thấy rõ hình tròn