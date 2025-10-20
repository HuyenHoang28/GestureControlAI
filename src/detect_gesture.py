"""
detect_gesture.py
Real-time gesture detection using MediaPipe landmarks + trained scikit-learn model.
Usage:
    python src/detect_gesture.py --model models/gesture_clf.joblib
Press:
    q : quit
"""
import cv2
import mediapipe as mp
import numpy as np
import joblib
import argparse
import time

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="models/gesture_clf.joblib", help="Trained model file (joblib)")
args = parser.parse_args()

data = joblib.load(args.model)
clf = data["model"]
le = data["label_encoder"]

cap = cv2.VideoCapture(0)
with mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                    min_detection_confidence=0.6, min_tracking_confidence=0.5) as hands:
    prev_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(img_rgb)
        label_text = "No hand"
        conf = 0.0
        if res.multi_hand_landmarks:
            lm = res.multi_hand_landmarks[0].landmark
            coords = []
            for p in lm:
                coords += [p.x, p.y, p.z]
            # preprocess like training
            pts = np.array(coords).reshape(-1,3)
            wrist = pts[0]
            pts = pts - wrist
            scale = np.max(np.linalg.norm(pts, axis=1)) + 1e-6
            pts = pts / scale
            feat = pts.flatten().reshape(1,-1)
            pred = clf.predict(feat)[0]
            proba = clf.predict_proba(feat).max()
            label_text = le.inverse_transform([pred])[0]
            conf = proba
            mp_drawing.draw_landmarks(frame, res.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)
        # FPS
        cur_time = time.time()
        fps = 1.0 / (cur_time - prev_time + 1e-6)
        prev_time = cur_time
        cv2.putText(frame, f"{label_text} ({conf:.2f})", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.putText(frame, f"FPS: {int(fps)}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
        cv2.imshow("Gesture Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()