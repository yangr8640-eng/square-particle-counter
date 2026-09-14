"""Image-only v4 descriptors: frozen radial features plus spatial shape context.

Coordinates only select pixels. IDs, model scores, status, human actions and
filename are never classifier inputs. No learned preprocessing spans images.
"""
from __future__ import annotations
import cv2
import numpy as np
from particle_features import extract_features as extract_radial_features

FEATURE_VERSION = 'local-radial-spatial-features-4.0'
_YY, _XX = np.mgrid[-16:17, -16:17].astype(np.float32)
_RADIUS = np.hypot(_XX, _YY)
_ANGLES = np.arange(32, dtype=np.float32) * (2 * np.pi / 32)
_R = np.arange(3, 16, dtype=np.float32)
_POLAR_X = (16 + np.cos(_ANGLES[:, None]) * _R).astype(np.float32)
_POLAR_Y = (16 + np.sin(_ANGLES[:, None]) * _R).astype(np.float32)


def spatial_feature_names():
    names = [f'v4_{channel}_grid_{y}_{x}' for channel in ('green', 'chroma')
             for y in range(9) for x in range(9)]
    names += [f'v4_hog_{y}_{x}_{b}' for y in range(4) for x in range(4) for b in range(8)]
    names += [f'v4_{channel}_{stat}' for channel in ('green', 'chroma') for stat in
              ('contrast_scale', 'trough_mean', 'trough_std', 'trough_harmonic2',
               'trough_harmonic4', 'radius_harmonic4', 'center_offset')]
    return names


def extract_spatial_features(image, points):
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError('Expected a nonempty uint8 BGR image')
    height, width = image.shape[:2]
    if min(height, width) < 1:
        raise ValueError('Expected a nonempty image')
    names = spatial_feature_names()
    if not points:
        return np.empty((0, len(names)), dtype=np.float32), names
    blurred = cv2.GaussianBlur(image.astype(np.float32), (0, 0), .8)
    b, g, r = blurred.transpose(2, 0, 1)
    channels = np.stack([g, np.minimum(r, g) - b], axis=-1)
    rows = []
    for point in points:
        x, y = float(point['x']), float(point['y'])
        if not (np.isfinite(x) and np.isfinite(y) and 0 <= x < width and 0 <= y < height):
            raise ValueError('Candidate center lies outside image')
        patch = cv2.remap(channels, _XX + x, _YY + y, cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT_101)
        row, normalized, extra = [], [], []
        for ci in range(2):
            p = patch[:, :, ci]
            median = np.median(p)
            scale = max(4., float(np.percentile(p, 90) - np.percentile(p, 10)))
            z = np.clip((p - median) / scale, -2, 2)
            row.extend(cv2.resize(z, (9, 9), interpolation=cv2.INTER_AREA).ravel())
            normalized.append(z)
            polar = cv2.remap(z, _POLAR_X, _POLAR_Y, cv2.INTER_LINEAR)
            trough = polar.min(axis=1)
            radii = np.argmin(polar, axis=1).astype(np.float32) + 3
            depth = float(z[_RADIUS < 3].mean()) - trough
            centered = depth - depth.mean()
            fourier = np.fft.rfft(centered) / len(centered)
            radial_fourier = np.fft.rfft(radii - radii.mean()) / len(radii)
            weight = np.maximum(z - np.percentile(z[_RADIUS < 10], 25), 0) * (_RADIUS < 10)
            total = max(1e-6, float(weight.sum()))
            offset = np.hypot(float((weight * _XX).sum()) / total,
                              float((weight * _YY).sum()) / total)
            extra.extend([scale, depth.mean(), depth.std(), abs(fourier[2]),
                          abs(fourier[4]), abs(radial_fourier[4]), offset])
        z = normalized[0][:32, :32]
        gx = cv2.Sobel(z, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(z, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = np.hypot(gx, gy)
        angle = (np.arctan2(gy, gx) % np.pi) * (8 / np.pi)
        for yy in range(0, 32, 8):
            for xx in range(0, 32, 8):
                m = magnitude[yy:yy + 8, xx:xx + 8].ravel()
                a = angle[yy:yy + 8, xx:xx + 8].ravel()
                floor = np.floor(a).astype(int)
                frac = a - floor
                hist = (np.bincount(floor % 8, m * (1 - frac), minlength=8)
                        + np.bincount((floor + 1) % 8, m * frac, minlength=8))
                row.extend(hist / max(1e-6, float(np.linalg.norm(hist))))
        row.extend(extra)
        rows.append(row)
    matrix = np.asarray(rows, dtype=np.float32)
    if matrix.shape != (len(points), len(names)) or not np.isfinite(matrix).all():
        raise ValueError('Invalid spatial feature matrix')
    return matrix, names


def extract_features(image, points, spatial=True):
    radial, names = extract_radial_features(image, points)
    if not spatial:
        return radial, names
    shape, shape_names = extract_spatial_features(image, points)
    return np.hstack([radial, shape]), names + shape_names
