"""Portable CPU inference for v4; original greens and gray suggestions protected."""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import cv2
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from particle_features_v4 import FEATURE_VERSION, extract_features


def apply_decisions(points, scores, thresholds):
    if len(points) != len(scores):
        raise ValueError('Candidate and score lengths differ')
    result = []
    for old, score in zip(points, scores):
        if not np.isfinite(score) or not 0 <= score <= 1:
            raise ValueError('Invalid model score')
        p = copy.deepcopy(old)
        # Reviewed inputs are immutable human decisions, never model labels.
        if old.get('origin') == 'manual' or old.get('human_decision') or old.get('confirmed'):
            result.append(p)
            continue
        p['model_score'] = round(float(score), 6)
        if old.get('suggestion') or old.get('removed'):
            result.append(p)
            continue
        baseline = old.get('baseline_status', old.get('status', 'review'))
        p['origin'], p['removed'], p['baseline_status'] = 'automatic', False, baseline
        priority = thresholds.get('review_priority_below', thresholds['reject_below'])
        p['review_priority'] = bool(score < priority)
        p.pop('model_rejected', None)
        if baseline == 'accepted':
            p['status'] = 'review' if score < priority else 'accepted'
            p['model_decision'] = 'review_original_green' if score < priority else 'preserve_original_green'
            if score < priority:
                p['reason'] = '模型与原检测不一致，需要复核；' + old.get('reason', '')
        elif score < thresholds['reject_below']:
            p.update(removed=True, model_rejected=True, status='review', model_decision='reject')
            p['reason'] = '模型自动排除（可恢复）：疑似干扰中心；' + old.get('reason', '')
        elif score >= thresholds['accept_at_least']:
            p.update(status='accepted', model_decision='accept')
            p['reason'] = '模型通过；' + old.get('reason', '')
        else:
            p.update(status='review', model_decision='review')
            p['reason'] = '模型仍不确定；' + old.get('reason', '')
        result.append(p)
    return result


class ParticleFilterV4:
    def __init__(self, model_path):
        self.path = Path(model_path)
        self.metadata = json.loads(self.path.with_suffix('.json').read_text(encoding='utf-8'))
        if hashlib.sha256(self.path.read_bytes()).hexdigest() != self.metadata['artifact_sha256']:
            raise ValueError('Trained artifact hash differs from metadata')
        for filename, digest in self.metadata['feature_source_hashes'].items():
            if hashlib.sha256(Path(__file__).with_name(filename).read_bytes()).hexdigest() != digest:
                raise ValueError('Model feature implementation differs; retrain required')
        artifact = joblib.load(self.path)
        if artifact['feature_version'] != FEATURE_VERSION:
            raise ValueError('Model feature version is incompatible')
        self.model = artifact['classifier']
        self.names = artifact['feature_names']
        self.thresholds = artifact['thresholds']
        self.spatial = artifact['spatial_features']

    def score(self, image, base_points, scale=1.0):
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError('scale must be positive and finite')
        if not base_points:
            return np.empty(0, np.float64)
        if scale != 1.0:
            height, width = image.shape[:2]
            image = cv2.resize(image, None, fx=1 / scale, fy=1 / scale, interpolation=cv2.INTER_AREA)
            ih, iw = image.shape[:2]
            base_points = [dict(p, x=p['x'] * iw / width, y=p['y'] * ih / height) for p in base_points]
        features, names = extract_features(image, base_points, spatial=self.spatial)
        if names != self.names:
            raise ValueError('Model feature schema differs from current implementation')
        with threadpool_limits(limits=4):
            return self.model.predict_proba(features)[:, 1]

    def apply(self, image, base_points, scale=1.0):
        return apply_decisions(base_points, self.score(image, base_points, scale), self.thresholds)
