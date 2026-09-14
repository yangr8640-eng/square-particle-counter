#!/usr/bin/env python3
"""Count individual stained square cores, with reviewable, original-pixel outputs.

A deterministic, training-free model. It detects bounded bright regions at many
thresholds in G and min(R,G)-B, verifies surrounding dark edges, and suppresses
multiple threshold observations of the same core. No per-image count is stored.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import time
from urllib.parse import quote

import cv2
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage.feature import peak_local_max

MODEL_VERSION = "square-core-1.0"


@dataclass(frozen=True)
class Config:
    # Dimensions below refer to images at the supplied 2448 x 2048 resolution.
    min_area: int = 30
    max_area: int = 800
    min_side: int = 7
    max_side: int = 43
    max_aspect: float = 2.0
    min_extent: float = 0.50
    min_eigen_ratio: float = 0.22
    min_edge: float = 5.0
    min_ring: float = 6.0
    min_green: float = 65.0
    min_chroma: float = -10.0
    nms_distance: float = 11.0
    red_margin: float = 35.0
    red_min_value: float = 130.0
    red_min_area: int = 5
    red_dilation: int = 3
    tile_size: int = 640
    tile_padding: int = 64
    smooth_sigma: float = 1.5
    threshold_step: int = 8
    agreement_distance: float = 9.0
    review_chroma: float = 20.0
    review_ring: float = 20.0
    review_edge: float = 12.0
    review_aspect: float = 1.65
    # This is a review heuristic, not a calibrated probability.
    scale: float = 1.0


def load_image(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode image: {path}")
    return image


def red_exclusion(image: np.ndarray, cfg: Config):
    b, g, r = image.astype(np.float32).transpose(2, 0, 1)
    raw = ((r > g + cfg.red_margin) & (r > b + cfg.red_margin)
           & (r > cfg.red_min_value)).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(raw, 8)
    valid = np.zeros(n, dtype=bool)
    valid[1:] = stats[1:, cv2.CC_STAT_AREA] >= cfg.red_min_area
    mask = valid[labels]
    # Reflect closes rings cut by the image boundary. Zero padding leaves
    # the yellow center connected to the exterior and cannot fill it.
    padding = max(32, cfg.max_side)
    mask = np.pad(mask, padding, mode="reflect")
    mask = ndi.binary_fill_holes(ndi.binary_closing(mask, iterations=2))
    mask = ndi.binary_dilation(mask, iterations=cfg.red_dilation) if cfg.red_dilation else mask
    mask = mask[padding:-padding, padding:-padding]
    n, labels, stats, centers = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    excluded = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        excluded.append(dict(x=float(centers[i, 0]), y=float(centers[i, 1]),
                             radius=float(max(w, h) / 2), area=int(area),
                             reason="red_halo_region"))
    return mask, excluded


def peak_agreement(sm, chroma, red, cfg):
    contrast = sm - cv2.GaussianBlur(sm, (0, 0), 10)
    response = chroma - cv2.GaussianBlur(chroma, (0, 0), 10) + .4 * contrast
    response = cv2.GaussianBlur(response, (0, 0), 1.2)
    mask = (chroma > 12) & (sm > cfg.min_green) & ~red
    coords = peak_local_max(response, min_distance=10, threshold_abs=8,
                            exclude_border=False, labels=mask.astype(np.uint8))
    return cKDTree(coords[:, ::-1]) if len(coords) else None


def core_candidates(sm, chroma, red, cfg):
    """Return region candidates within one padded tile, in tile coordinates."""
    mser = cv2.MSER_create(3, cfg.min_area, cfg.max_area, .45, .2, 200, 1.01, .003, 5)
    rows = []
    theta = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    radii = np.arange(6, 18)
    dy, dx = np.sin(theta[:, None]) * radii, np.cos(theta[:, None]) * radii
    tree = peak_agreement(sm, chroma, red, cfg)
    for source, channel in [("green", sm), ("chroma", chroma + 100)]:
        regions, boxes = mser.detectRegions(np.clip(channel, 0, 255).astype(np.uint8))
        regions, boxes = list(regions), list(boxes)
        # Explicit high threshold components recover flat/saturated cores that
        # an extremal-region stability detector alone can omit.
        start = 65 if source == "green" else 95
        for level in range(start, 256, cfg.threshold_step):
            n, labels, stats, _ = cv2.connectedComponentsWithStats((channel > level).astype(np.uint8), 8)
            for label in range(1, n):
                bx, by, w, h, area = stats[label]
                if not (cfg.min_area <= area <= cfg.max_area and min(w, h) >= cfg.min_side
                        and max(w, h) <= cfg.max_side and max(w, h) / min(w, h) <= cfg.max_aspect
                        and area / (w * h) >= cfg.min_extent):
                    continue
                ys, xs = np.where(labels[by:by + h, bx:bx + w] == label)
                regions.append(np.column_stack((xs + bx, ys + by)))
                boxes.append((bx, by, w, h))
        for coords, box in zip(regions, boxes):
            bx, by, w, h = map(int, box)
            area = len(coords)
            if (min(w, h) < cfg.min_side or max(w, h) > cfg.max_side
                or max(w, h) / min(w, h) > cfg.max_aspect
                or area / (w * h) < cfg.min_extent):
                continue
            x, y = map(float, coords.mean(axis=0))
            ix, iy = round(x), round(y)
            if red[iy, ix] or sm[iy, ix] < cfg.min_green:
                continue
            eig = np.linalg.eigvalsh(np.cov(coords.T))
            if eig[0] / max(eig[1], 1e-6) < cfg.min_eigen_ratio:
                continue
            pad = 5
            if bx < pad or by < pad or bx + w + pad > channel.shape[1] or by + h + pad > channel.shape[0]:
                continue
            region = np.zeros((h + 2 * pad, w + 2 * pad), np.uint8)
            region[coords[:, 1] - by + pad, coords[:, 0] - bx + pad] = 1
            ring_mask = (cv2.dilate(region, np.ones((7, 7), np.uint8)) > 0) & (region == 0)
            crop = channel[by - pad:by + h + pad, bx - pad:bx + w + pad]
            edge = float(channel[coords[:, 1], coords[:, 0]].mean() - np.median(crop[ring_mask]))
            if edge < cfg.min_edge:
                continue
            trough = ndi.map_coordinates(sm, [y + dy, x + dx], order=1, mode="nearest").min(axis=1)
            center = float(cv2.getRectSubPix(sm, (5, 5), (x, y)).mean())
            dark_ring = float(np.quantile(center - trough, .2))
            if dark_ring < cfg.min_ring:
                continue
            color = float(chroma[iy, ix])
            if color < cfg.min_chroma:
                continue
            agree = tree is not None and tree.query([x, y])[0] <= cfg.agreement_distance
            rank = .2 * edge + dark_ring + .04 * max(color, 0)
            rows.append(dict(x=x, y=y, area=area, width=w, height=h, edge=edge,
                             ring=dark_ring, chroma=color, green=center, rank=rank,
                             source=source, agrees=bool(agree), extent=area / (w * h)))
    return rows


def suppress_duplicates(rows, distance):
    # Spatial bins make suppression linear for dense images; threshold versions
    # of a core are ranked by closed dark boundary strength.
    bins, picked = {}, []
    for row in sorted(rows, key=lambda p: p["rank"], reverse=True):
        key = (int(row["x"] // distance), int(row["y"] // distance))
        neighbors = [q for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                     for q in bins.get((key[0] + dx, key[1] + dy), [])]
        if any((row["x"] - q["x"]) ** 2 + (row["y"] - q["y"]) ** 2 < distance ** 2
               for q in neighbors):
            continue
        bins.setdefault(key, []).append(row)
        picked.append(row)
    return picked


def detect(image: np.ndarray, cfg: Config = Config()):
    original_h, original_w = image.shape[:2]
    if cfg.scale <= 0:
        raise ValueError("scale must be positive")
    if cfg.scale != 1:
        image = cv2.resize(image, None, fx=1 / cfg.scale, fy=1 / cfg.scale, interpolation=cv2.INTER_AREA)
    h, w = image.shape[:2]
    red, excluded = red_exclusion(image, cfg)
    # Reflect padding makes edge objects detectable. Only original-image
    # centers survive, and incomplete edge objects are always marked review.
    halo = cfg.tile_padding
    b, g, r = image.astype(np.float32).transpose(2, 0, 1)
    sm = cv2.GaussianBlur(g, (0, 0), cfg.smooth_sigma)
    chroma = cv2.GaussianBlur(np.minimum(r, g) - b, (0, 0), cfg.smooth_sigma)
    padded_sm = np.pad(sm, halo, mode="reflect")
    padded_c = np.pad(chroma, halo, mode="reflect")
    padded_red = np.pad(red, halo, mode="reflect")
    rows = []
    for y0 in range(0, h, cfg.tile_size):
        y1 = min(h, y0 + cfg.tile_size)
        for x0 in range(0, w, cfg.tile_size):
            x1 = min(w, x0 + cfg.tile_size)
            sl = np.s_[y0:y1 + 2 * halo, x0:x1 + 2 * halo]
            candidates = core_candidates(padded_sm[sl], padded_c[sl], padded_red[sl], cfg)
            # Suppress first, then keep centers inside a disjoint tile interior.
            # The 64px halo is greater than maximum core size + ring radius.
            for p in suppress_duplicates(candidates, cfg.nms_distance):
                p["x"] += x0 - halo
                p["y"] += y0 - halo
                if x0 <= p["x"] < x1 and y0 <= p["y"] < y1:
                    rows.append(p)
    rows = suppress_duplicates(rows, cfg.nms_distance)
    red_distance = ndi.distance_transform_edt(~red) if np.any(red) else None
    points = []
    for p in sorted(rows, key=lambda p: (round(p["y"] / 35), p["x"])):
        x, y = p["x"], p["y"]
        ix, iy = min(w - 1, round(x)), min(h - 1, round(y))
        if red[iy, ix]:
            continue
        reasons = []
        if not p["agrees"]:
            reasons.append("两种检测依据不一致")
        if p["chroma"] < cfg.review_chroma:
            reasons.append("染色较淡或亮缝")
        if p["ring"] < cfg.review_ring or p["edge"] < cfg.review_edge:
            reasons.append("暗边较弱")
        if max(p["width"], p["height"]) / min(p["width"], p["height"]) > cfg.review_aspect:
            reasons.append("形状偏长")
        if p["area"] < 55 or p["area"] > 500:
            reasons.append("尺寸异常")
        if min(x, y, w - 1 - x, h - 1 - y) < max(p["width"], p["height"]) / 2 + 2:
            reasons.append("图像边缘不完整")
        if red_distance is not None and red_distance[iy, ix] < 18:
            reasons.append("邻近红色光晕")
        metrics = {k: round(float(p[k]), 3) for k in ("area", "edge", "ring", "chroma", "green", "extent")}
        points.append(dict(id=len(points) + 1, x=round(x * original_w / w, 2),
                           y=round(y * original_h / h, 2),
                           radius=round(max(6, math.sqrt(p["area"] / math.pi)) * cfg.scale, 2),
                           status="review" if reasons else "accepted", reason="；".join(reasons),
                           source=p["source"], metrics=metrics))
    for p in excluded:
        p["x"] = round(p["x"] * original_w / w, 2)
        p["y"] = round(p["y"] * original_h / h, 2)
        p["radius"] = round(p["radius"] * cfg.scale, 2)
    return points, excluded, red


def write_image(path: Path, image: np.ndarray):
    ok, data = cv2.imencode(path.suffix, image)
    if not ok:
        raise OSError(f"Cannot encode {path}")
    data.tofile(path)


def annotate(image, points, excluded):
    result = image.copy()
    for p in excluded:
        x, y = round(p["x"]), round(p["y"])
        cv2.circle(result, (x, y), max(5, round(p["radius"])), (65, 65, 245), 2, cv2.LINE_AA)
    for p in points:
        x, y = round(p["x"]), round(p["y"])
        color = (0, 175, 255) if p["status"] == "review" else (30, 210, 40)
        cv2.circle(result, (x, y), 5, color, 1, cv2.LINE_AA)
        pos = (x + 5, y - 4)
        cv2.putText(result, str(p["id"]), pos, cv2.FONT_HERSHEY_SIMPLEX, .30, (25, 25, 25), 2, cv2.LINE_AA)
        cv2.putText(result, str(p["id"]), pos, cv2.FONT_HERSHEY_SIMPLEX, .30, color, 1, cv2.LINE_AA)
    return result


def run(args):
    source, output = Path(args.input).resolve(), Path(args.output).resolve()
    if source == output:
        raise ValueError("Output must be separate from original input directory")
    if not source.exists():
        raise ValueError(f"Input does not exist: {source}")
    files = sorted(source.glob("*.png")) + sorted(source.glob("*.PNG")) if source.is_dir() else [source]
    files = sorted(set(files), key=lambda p: p.name.casefold())
    if not files:
        raise ValueError(f"No PNG images in {source}")
    values = json.loads(Path(args.config).read_text()) if args.config else {}
    if args.scale is not None:
        values["scale"] = args.scale
    cfg = Config(**values)
    if cfg.tile_size < 64 or cfg.tile_padding < 48 or cfg.threshold_step < 1 or cfg.nms_distance <= 0:
        raise ValueError("Invalid tile, threshold or suppression settings")
    output.mkdir(parents=True, exist_ok=True)
    (output / "annotated").mkdir(exist_ok=True)
    (output / "red_masks").mkdir(exist_ok=True)
    import os
    records = []
    for path in files:
        start = time.perf_counter()
        image = load_image(path)
        points, excluded, red = detect(image, cfg)
        record = dict(name=path.name, width=image.shape[1], height=image.shape[0],
                      src=quote(os.path.relpath(path, output), safe="/"),
                      sha256=hashlib.sha256(path.read_bytes()).hexdigest(), points=points,
                      excluded=excluded, count=len(points), review_count=sum(p["status"] == "review" for p in points),
                      red_region_count=len(excluded), elapsed_seconds=round(time.perf_counter() - start, 2))
        records.append(record)
        (output / f"{path.stem}.json").write_text(json.dumps(record, ensure_ascii=False, indent=2))
        write_image(output / "annotated" / f"{path.stem}_counted.png", annotate(image, points, excluded))
        if red.shape != image.shape[:2]:
            red = cv2.resize(red.astype(np.uint8), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        write_image(output / "red_masks" / f"{path.stem}_red.png", red.astype(np.uint8) * 255)
        print(f"{path.name}: {record['count']} candidates, {record['review_count']} review, "
              f"{len(excluded)} red regions; {record['elapsed_seconds']:.1f}s", flush=True)
    data = dict(model_version=MODEL_VERSION, config=asdict(cfg),
                method="多阈值封闭亮芯 + MSER 稳定区域 + 黄绿色差/局部峰交叉检查 + 暗边形状约束 + 红光晕排除。自动数包含橙色待复核点，并非人工确认真值。",
                images=records)
    (output / "detections.json").write_text(json.dumps(data, ensure_ascii=False, indent=2))
    with (output / "counts.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "automatic_count_including_review", "standard_candidates", "review_candidates", "red_regions_not_particles"])
        for r in records:
            writer.writerow([r["name"], r["count"], r["count"] - r["review_count"], r["review_count"], r["red_region_count"]])
        writer.writerow(["TOTAL", sum(r["count"] for r in records), sum(r["count"] - r["review_count"] for r in records),
                         sum(r["review_count"] for r in records), sum(r["red_region_count"] for r in records)])
    template = Path(__file__).with_name("review_template.html").read_text()
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    (output / "review.html").write_text(template.replace("__PARTICLE_DATA_JSON__", payload))
    print(f"Review: {output / 'review.html'}")
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=".", help="PNG file or directory (non-recursive)")
    parser.add_argument("--output", default="results", help="Output directory separate from inputs")
    parser.add_argument("--config", help="Optional JSON overrides for Config")
    parser.add_argument("--scale", type=float, help="Particle size relative to this data, e.g. 2 for double size")
    args = parser.parse_args()
    try:
        run(args)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
