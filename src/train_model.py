"""
train_model.py
Train a classifier (RandomForest) on collected landmarks CSV and save a model (joblib).
Usage:
    python src/train_model.py --in data/landmarks.csv --out models/gesture_clf.joblib
"""
import pandas as pd
import numpy as np
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os

parser = argparse.ArgumentParser()
parser.add_argument("--in", dest="infile", default="data/landmarks.csv", help="Input CSV")
parser.add_argument("--out", dest="outfile", default="models/gesture_clf.joblib", help="Output model file")
args = parser.parse_args()

df = pd.read_csv(args.infile, header=None)
print(f"Loaded {len(df)} samples from {args.infile}")
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Simple preprocessing: center by wrist (landmark 0)
def preprocess(X):
    Xp = []
    for row in X:
        pts = row.reshape(-1,3)
        wrist = pts[0]
        pts = pts - wrist
        # scale normalization by max distance
        scale = np.max(np.linalg.norm(pts, axis=1)) + 1e-6
        pts = pts / scale
        Xp.append(pts.flatten())
    return np.array(Xp)

Xp = preprocess(X)
le = LabelEncoder()
y_enc = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(Xp, y_enc, test_size=0.2, stratify=y_enc, random_state=42)

clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred, target_names=le.classes_))
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))

os.makedirs(os.path.dirname(args.outfile), exist_ok=True)
joblib.dump({"model": clf, "label_encoder": le}, args.outfile)
print(f"Saved model to {args.outfile}")
