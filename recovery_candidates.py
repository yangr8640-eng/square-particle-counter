"""Optional proposals for dim cores missed by the bright-core detector.

This is a proposal branch, not a validated extra count. Returned points must stay
in review: absent labels in a correction export do not prove background. No
reference image names, point coordinates, or corrected labels are used here.
"""
from __future__ import annotations

from dataclasses import replace
import math

import cv2
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree


def propose_dim_cores(image, existing_points=(), cfg=None):
    """Return new review-point dictionaries in original image coordinates.

    `image` is a uint8 BGR ndarray; `existing_points` is the original detector's
    point list, including rejected old candidates so proposals cannot resurrect
    deleted detections. `cfg` is particle_counter.Config, or None for defaults.
    Pixels and existing coordinates are the only inference inputs.

    This deliberately retains the original shape constraints, switches the
    radial evidence to min(R,G)-B, and only proposes G < 180 dim interiors. It
    does not attempt to identify particles with no coherent stained interior.
    """
    # Lazy import lets the main detector optionally import this proposal module.
    from particle_counter import Config, core_candidates, red_exclusion, suppress_duplicates

    cfg = Config() if cfg is None else cfg
    if cfg.scale <= 0:
        raise ValueError("scale must be positive")
    original_h, original_w = image.shape[:2]
    if cfg.scale != 1:
        image = cv2.resize(image, None, fx=1 / cfg.scale, fy=1 / cfg.scale,
                           interpolation=cv2.INTER_AREA)
    h, w = image.shape[:2]
    b, g, r = image.astype(np.float32).transpose(2, 0, 1)
    sm = cv2.GaussianBlur(g, (0, 0), cfg.smooth_sigma)
    chroma = cv2.GaussianBlur(np.minimum(r, g) - b, (0, 0), cfg.smooth_sigma)
    red, _ = red_exclusion(image, cfg)
    halo = cfg.tile_padding
    padded = [np.pad(a, halo, mode="reflect") for a in (sm, chroma, red)]
    old_xy = [(p["x"] * w / original_w, p["y"] * h / original_h)
              for p in existing_points]
    tree = cKDTree(old_xy) if old_xy else None
    theta = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    radii = np.arange(6, 20)
    dx = np.cos(theta[:, None]) * radii
    dy = np.sin(theta[:, None]) * radii
    rows = []
    relaxed = replace(cfg, min_ring=-100)
    for y0 in range(0, h, cfg.tile_size):
        y1 = min(h, y0 + cfg.tile_size)
        for x0 in range(0, w, cfg.tile_size):
            x1 = min(w, x0 + cfg.tile_size)
            sl = np.s_[y0:y1 + 2 * halo, x0:x1 + 2 * halo]
            for p in core_candidates(*(a[sl] for a in padded), relaxed):
                if p["source"] != "chroma" or p["chroma"] < 10 or p["green"] >= 180:
                    continue
                p["x"] += x0 - halo
                p["y"] += y0 - halo
                x, y = p["x"], p["y"]
                if not (x0 <= x < x1 and y0 <= y < y1):
                    continue
                if tree is not None and tree.query([x, y])[0] < 13:
                    continue
                center = float(cv2.getRectSubPix(chroma, (5, 5), (x, y)).mean())
                trough = ndi.map_coordinates(padded[1], [y + halo + dy, x + halo + dx],
                                               order=1, mode="nearest").min(axis=1)
                p["chroma_ring"] = float(np.quantile(center - trough, .2))
                if p["chroma_ring"] < 20:
                    continue
                p["rank"] = p["chroma_ring"] + .2 * p["edge"]
                rows.append(p)
    rows = suppress_duplicates(rows, max(17, cfg.nms_distance))
    numeric_ids = [p.get("id", 0) for p in existing_points
                   if isinstance(p.get("id", 0), (int, float))]
    first_id = int(max(numeric_ids, default=0)) + 1
    result = []
    for p in sorted(rows, key=lambda p: (round(p["y"] / 35), p["x"])):
        result.append(dict(
            id=first_id + len(result),
            x=round(p["x"] * original_w / w, 2),
            y=round(p["y"] * original_h / h, 2),
            radius=round(max(6, math.sqrt(p["area"] / math.pi)) * cfg.scale, 2),
            status="review", removed=True, suggestion=True, reason="低亮度补漏候选；仅色差暗边成立，需要人工复核",
            source="chroma_recovery", origin="automatic",
            metrics={k: round(float(p[k]), 3) for k in
                     ("area", "edge", "ring", "chroma", "green", "extent", "chroma_ring")},
        ))
    return result
