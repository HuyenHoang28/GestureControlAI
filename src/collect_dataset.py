"""
collect_dataset.py
Capture hand landmarks using MediaPipe and save samples labelled by gesture.
Usage:
    python src/collect_dataset.py --label "thumbs_up" --out data/landmarks.csv
Press:
    s : save current sample (when hand detected)
    q : quit
"""
import cv2
import mediapipe as mp
import argparse
import csv
import os
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

parser = argparse.ArgumentParser()
parser.add_argument("--label", required=True, help="Label name for this gesture")
parser.add_argument("--out", default="data/landmarks.csv", help="Output CSV file")
args = parser.parse_args()

out_path = args.out
os.makedirs(os.path.dirname(out_path), exist_ok=True)

cap = cv2.VideoCapture(0)
with mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                    min_detection_confidence=0.6, min_tracking_confidence=0.5) as hands, \
     open(out_path, "a", newline="") as f:
    writer = csv.writer(f)
    print("Press 's' to save a sample, 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(img_rgb)
        if res.multi_hand_landmarks:
            lm = res.multi_hand_landmarks[0].landmark
            mp_drawing.draw_landmarks(frame, res.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)
            coords = []
            for p in lm:
                coords += [p.x, p.y, p.z]
            # not auto-save: wait for 's'
        cv2.putText(frame, f"Label: {args.label}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.imshow("Collect Dataset", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            if res.multi_hand_landmarks:
                row = coords + [args.label]
                writer.writerow(row)
                print("Saved sample.")
            else:
                print("No hand detected.")
        elif key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
