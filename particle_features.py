"""Image-only features for learned particle filtering; no label or position inputs.

All channels are calculated from a fixed, local patch at the original candidate
center. Absolute coordinates, filename, point ID, review reason, and user actions
are deliberately unavailable to the classifier.
"""
from __future__ import annotations
import cv2
import numpy as np

FEATURE_VERSION = 'local-core-features-1'


def extract_features(image: np.ndarray, points: list[dict]) -> tuple[np.ndarray, list[str]]:
    b, g, r = cv2.GaussianBlur(image.astype(np.float32), (0, 0), 1.0).transpose(2, 0, 1)
    channels = np.stack([g, np.minimum(r, g) - b, g - r, b], axis=-1)
    yy, xx = np.mgrid[-32:33, -32:33]
    radius = np.hypot(xx, yy)
    rings = [(0, 3), (3, 6), (6, 9), (9, 12), (12, 16), (16, 22), (22, 32)]
    names, rows = [], []
    angles = np.linspace(0, 2 * np.pi, 32, endpoint=False)
    rad = np.arange(2, 25, dtype=np.float32)
    map_x = (32 + np.cos(angles[:, None]) * rad).astype(np.float32)
    map_y = (32 + np.sin(angles[:, None]) * rad).astype(np.float32)
    for point in points:
        # getRectSubPix supports one/three channels; remap works for four.
        px = (xx + float(point['x'])).astype(np.float32)
        py = (yy + float(point['y'])).astype(np.float32)
        patch = cv2.remap(channels, px, py, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        features = {}
        for ci, channel_name in enumerate(('green', 'chroma', 'g_minus_r', 'blue')):
            arr = patch[:, :, ci]
            centers = []
            for lo, hi in rings:
                values = arr[(radius >= lo) & (radius < hi)]
                for q in (10, 50, 90):
                    features[f'{channel_name}_r{lo}_{hi}_q{q}'] = float(np.percentile(values, q))
                features[f'{channel_name}_r{lo}_{hi}_std'] = float(values.std())
                centers.append(float(values.mean()))
            for i in range(1, len(centers)):
                features[f'{channel_name}_center_minus_r{i}'] = centers[0] - centers[i]
            if ci in (0, 1):
                polar = cv2.remap(arr, map_x, map_y, cv2.INTER_LINEAR)
                trough = polar[:, 4:18].min(axis=1)
                ring_depth = centers[0] - trough
                for q in (0, 10, 25, 50, 75, 90, 100):
                    features[f'{channel_name}_trough_depth_q{q}'] = float(np.percentile(ring_depth, q))
                for threshold in (5, 15, 30, 60):
                    features[f'{channel_name}_dark_coverage_{threshold}'] = float(np.mean(ring_depth > threshold))
                trough_radius = np.argmin(polar[:, 4:18], axis=1) + 6
                features[f'{channel_name}_trough_radius_mean'] = float(trough_radius.mean())
                features[f'{channel_name}_trough_radius_std'] = float(trough_radius.std())
                for extent in (7, 11, 16):
                    box = radius < extent
                    weight = np.maximum(arr - np.percentile(arr[box], 25), 0) * box
                    norm = max(float(weight.sum()), 1e-5)
                    cx, cy = float((weight * xx).sum() / norm), float((weight * yy).sum() / norm)
                    moments = np.array([[(weight * (xx - cx)**2).sum(), (weight * (xx - cx) * (yy - cy)).sum()],
                                        [(weight * (xx - cx) * (yy - cy)).sum(), (weight * (yy - cy)**2).sum()]]) / norm
                    eig = np.linalg.eigvalsh(moments)
                    features[f'{channel_name}_center_offset_{extent}'] = float(np.hypot(cx, cy))
                    features[f'{channel_name}_eigen_ratio_{extent}'] = float(eig[0] / max(eig[1], 1e-5))
        if not names:
            names = list(features)
        rows.append([features[name] for name in names])
    if not points:
        _, names = extract_features(np.full((70, 70, 3), 100, np.uint8), [{'x': 35, 'y': 35}])
    return np.asarray(rows, dtype=np.float32).reshape(len(points), len(names)), names
