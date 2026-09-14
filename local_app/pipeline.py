"""Replaceable single-image inference adapter; no uploads or HTTP logic here."""
from __future__ import annotations
from dataclasses import asdict
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def default_model_path():
    """Prefer explicit/bundled artifacts; never select an experimental model."""
    configured = os.environ.get('PARTICLE_COUNTER_MODEL')
    if configured:
        return Path(configured).expanduser().resolve()
    bundled = Path(__file__).with_name('model') / 'particle_filter.joblib'
    if bundled.is_file():
        return bundled
    development = PROJECT_ROOT / 'models' / 'particle_filter_v3' / 'particle_filter.joblib'
    if not development.is_file():
        raise FileNotFoundError('未找到可用模型，请用 --model 指定模型文件。')
    return development


@lru_cache(maxsize=2)
def _load_filter(model_path, artifact_mtime, metadata_mtime):
    metadata = json.loads(Path(model_path).with_suffix('.json').read_text(encoding='utf-8'))
    if str(metadata.get('model_version', '')).startswith('square-core-filter-4'):
        from particle_filter_v4 import ParticleFilterV4
        return ParticleFilterV4(model_path)
    from particle_filter import ParticleFilter
    return ParticleFilter(model_path)


def prepare_model(model_path: Path):
    """Load/verify the artifact, feature code and required inference/review code."""
    from particle_counter import Config, detect, load_image, annotate, write_image
    from recovery_candidates import propose_dim_cores
    from count_particles_v2 import make_review_template

    path = Path(model_path).resolve()
    model = _load_filter(str(path), path.stat().st_mtime_ns,
                         path.with_suffix('.json').stat().st_mtime_ns)
    # The offline review template is a required bundled runtime resource.
    make_review_template()
    return model


def run_one_image(image_path: Path, output_dir: Path, *, model_path: Path) -> dict:
    """Write annotated/<stem>_counted.png and return the original-pixel record.

    The worker calls this serially. A replacement must preserve point provenance
    and keep unconfirmed model suggestions removed from the initial count.
    """
    from particle_counter import Config, detect, load_image, annotate, write_image
    from recovery_candidates import propose_dim_cores

    started = time.perf_counter()
    model_path = Path(model_path).resolve()
    model = _load_filter(str(model_path), model_path.stat().st_mtime_ns,
                         model_path.with_suffix('.json').stat().st_mtime_ns)
    cfg = Config()
    image = load_image(image_path)
    base_points, excluded, _ = detect(image, cfg)
    points = model.apply(image, base_points)
    proposals = propose_dim_cores(image, base_points, cfg)
    for point in proposals:
        point.update(removed=True, suggestion=True, model_decision='dim_proposal', origin='automatic')
        point['reason'] = '暗颗粒补漏建议（尚未计入）；' + point.get('reason', '')
    points.extend(proposals)
    active = [point for point in points if not point.get('removed', False)]
    digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
    annotated = Path(output_dir) / 'annotated' / (image_path.stem + '_counted.png')
    annotated.parent.mkdir(parents=True, exist_ok=True)
    write_image(annotated, annotate(image, active, excluded))
    return dict(width=int(image.shape[1]), height=int(image.shape[0]), points=points,
                excluded=excluded, count=len(active), baseline_count=len(base_points),
                review_count=sum(point['status'] == 'review' for point in active),
                model_rejected_count=sum(bool(point.get('model_rejected')) for point in points),
                dim_proposal_count=len(proposals), red_region_count=len(excluded),
                used_in_model_training=digest in model.metadata.get('source_image_hashes', {}).values(),
                elapsed_seconds=round(time.perf_counter() - started, 2),
                model_metadata=dict(model_version=model.metadata.get('model_version', '本地模型'),
                    model_sha256=model.metadata['artifact_sha256'], thresholds=model.thresholds,
                    config=asdict(cfg), method='原视觉候选检测与人工标注训练的本地筛选模型。橙点已计入；灰色建议尚未计入，模型通过不等于人工确认。'))
