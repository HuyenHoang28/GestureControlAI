import cv2

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Không mở được camera. Hãy kiểm tra quyền truy cập hoặc chọn camera khác (sửa 0 → 1).")
else:
    print("✅ Camera hoạt động! Nhấn 'q' để thoát.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Test Camera", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()
