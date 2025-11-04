# -*- coding: utf-8 -*-
import cv2
import mediapipe as mp
import numpy as np
import argparse
import os
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# === VIETNAMESE TEXT SUPPORT (using Pillow) ===
from PIL import ImageFont, ImageDraw, Image

def draw_text_vi(img, text, pos, font_path="Roboto-Regular.ttf", font_size=32, color=(0,255,0)):
    """Draw Vietnamese text on an OpenCV image using Pillow."""
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype(font_path, font_size)
    except OSError:
        font = ImageFont.load_default()
    draw.text(pos, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
# ==============================================

DATASET_PATH = "landmarks.npy"     
MODEL_PATH   = "knn_model.joblib"
SCALER_PATH  = "scaler.joblib"

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_hands_connections = mp.solutions.hands.HAND_CONNECTIONS

def extract_hand_landmarks(results):
    if not results.multi_hand_landmarks:
        return None

    all_coords = []
    for hand in results.multi_hand_landmarks:
        coords = np.array([[lm.x, lm.y, lm.z] for lm in hand.landmark])
        base = coords[0].copy()
        coords -= base
        max_dist = np.max(np.linalg.norm(coords, axis=1))
        if max_dist > 1e-6:
            coords /= max_dist
        all_coords.append(coords.flatten())

    # if only one hand, add zero vector for the other
    if len(all_coords) == 1:
        all_coords.append(np.zeros_like(all_coords[0]))

    feature = np.concatenate(all_coords)
    return feature  

def load_dataset():
    if os.path.exists(DATASET_PATH):
        return list(np.load(DATASET_PATH, allow_pickle=True))
    else:
        return []

def save_dataset(dataset):
    np.save(DATASET_PATH, np.array(dataset, dtype=object))
    print(f"✅ Saved dataset: {DATASET_PATH} (samples={len(dataset)})")

def collect(label):
    cap = cv2.VideoCapture(0)
    dataset = load_dataset()
    print("📷 Camera opened. Press SPACE to capture a sample for label:", label)
    print("Press q to quit.")
    with mp_hands.Hands(static_image_mode=False,
                        max_num_hands=2,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5) as hands:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Cannot read camera frame.")
                break
            img = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands_connections)
                    label_h = handedness.classification[0].label  # Left / Right
                    coords = hand_landmarks.landmark[0]
                    h, w, _ = img.shape
                    x, y = int(coords.x * w), int(coords.y * h)
                    cv2.putText(img, label_h, (x - 20, y - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            # --- Vietnamese text drawn with Pillow ---
            img = draw_text_vi(img, f"Label: {label}", (10, 20))
            img = draw_text_vi(img, "Space: save sample  |  q: quit", (10, 60), font_size=26, color=(255,255,0))
            # ------------------------------------------

            cv2.imshow("Collect gestures", img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == 32:  # space
                landmark_vec = extract_hand_landmarks(results)
                if landmark_vec is not None:
                    dataset.append((label, landmark_vec))
                    print(f"Saved sample #{len(dataset)} for label '{label}'")
                    save_dataset(dataset)
                else:
                    print("No hand detected. Try again.")

    cap.release()
    cv2.destroyAllWindows()

def train():
    dataset = load_dataset()
    if len(dataset) == 0:
        print("No data found. Use collect mode to gather samples first.")
        return
    labels = [d[0] for d in dataset]
    X = np.stack([d[1] for d in dataset], axis=0)
    y = np.array(labels)

    unique_labels = sorted(list(set(labels)))
    label2idx = {lab:i for i,lab in enumerate(unique_labels)}
    y_idx = np.array([label2idx[l] for l in y])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    Xtr, Xte, ytr, yte = train_test_split(X_scaled, y_idx, test_size=0.2, random_state=42, stratify=y_idx)

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(Xtr, ytr)
    acc = knn.score(Xte, yte)
    print(f"Trained KNN. Test accuracy: {acc*100:.2f}%")

    joblib.dump({
        "model": knn,
        "label2idx": label2idx,
        "idx2label": {v:k for k,v in label2idx.items()}
    }, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"Saved model to {MODEL_PATH} and scaler to {SCALER_PATH}")

def predict():
    if not os.path.exists(MODEL_PATH):
        print("Model not found. Train first.")
        return
    data = joblib.load(MODEL_PATH)
    knn = data["model"]
    idx2label = data["idx2label"]
    scaler = joblib.load(SCALER_PATH)

    cap = cv2.VideoCapture(0)
    with mp_hands.Hands(static_image_mode=False,
                        max_num_hands=2,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5) as hands:
        print("Starting real-time prediction. Press q to quit.")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            img = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            label_text = "No hand"
            prob_text = ""
            if results.multi_hand_landmarks:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands_connections)

                vec = extract_hand_landmarks(results)
                if vec is not None:
                    vec_scaled = scaler.transform(vec.reshape(1, -1))
                    pred_idx = knn.predict(vec_scaled)[0]
                    if hasattr(knn, "predict_proba"):
                        probs = knn.predict_proba(vec_scaled)[0]
                        conf = probs[pred_idx]
                        prob_text = f"{conf*100:.1f}%"
                    label_text = idx2label[int(pred_idx)]

            # --- Draw Vietnamese text with Pillow ---
            img = draw_text_vi(img, f"Gesture: {label_text}", (10, 20), color=(0,255,0))
            if prob_text:
                img = draw_text_vi(img, f"Confidence: {prob_text}", (10, 60), color=(0,200,255))
            # ----------------------------------------

            cv2.imshow("Hand Gesture Recognition", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description="Hand gesture recognition pipeline with MediaPipe + KNN")
    sub = parser.add_subparsers(dest="mode", required=True)

    p_collect = sub.add_parser("collect")
    p_collect.add_argument("--label", required=True, help="label name for the gesture (e.g. fist, palm)")

    sub.add_parser("train")
    sub.add_parser("predict")

    args = parser.parse_args()
    if args.mode == "collect":
        collect(args.label)
    elif args.mode == "train":
        train()
    elif args.mode == "predict":
        predict()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
