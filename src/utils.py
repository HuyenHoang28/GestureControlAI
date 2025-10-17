"""
utils.py
Utility functions (preprocessing helpers).
"""
import numpy as np

def preprocess_row(row):
    """Given flattened row length 63 -> center by wrist and scale normalize."""
    pts = np.array(row).reshape(-1,3)
    wrist = pts[0]
    pts = pts - wrist
    scale = np.max(np.linalg.norm(pts, axis=1)) + 1e-6
    pts = pts / scale
    return pts.flatten()
