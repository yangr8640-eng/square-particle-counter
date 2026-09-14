"""Inference wrapper for the reviewed-label classifier.

Only local image features enter inference. User correction JSON is never loaded
here; model metadata names/hashes are provenance, not per-image overrides.
"""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import cv2
from threadpoolctl import threadpool_limits
from particle_features import extract_features, FEATURE_VERSION


class ParticleFilter:
    def __init__(self, model_path):
        self.path = Path(model_path)
        metadata_path = self.path.with_suffix('.json')
        self.metadata = json.loads(metadata_path.read_text())
        digest = hashlib.sha256(self.path.read_bytes()).hexdigest()
        if digest != self.metadata['artifact_sha256']:
            raise ValueError('Trained artifact hash differs from metadata')
        feature_hash = hashlib.sha256(Path(__file__).with_name('particle_features.py').read_bytes()).hexdigest()
        if self.metadata.get('feature_source_sha256') != feature_hash:
            raise ValueError('Model feature source differs from current implementation; retrain required')
        artifact = joblib.load(self.path)
        if artifact['feature_version'] != FEATURE_VERSION:
            raise ValueError('Model feature version is incompatible')
        self.model = artifact['classifier']
        self.names = artifact['feature_names']
        self.thresholds = artifact['thresholds']

    def score(self, image, points, scale=1.0):
        if not points:
            return np.empty(0, np.float64)
        if scale != 1.0:
            h, w = image.shape[:2]
            image = cv2.resize(image, None, fx=1 / scale, fy=1 / scale, interpolation=cv2.INTER_AREA)
            ih, iw = image.shape[:2]
            points = [dict(p, x=p['x'] * iw / w, y=p['y'] * ih / h) for p in points]
        X, names = extract_features(image, points)
        if names != self.names:
            raise ValueError('Model feature schema differs from current implementation')
        with threadpool_limits(limits=4):
            return self.model.predict_proba(X)[:, 1]

    def apply(self, image, points, scale=1.0):
        scores = self.score(image, points, scale)
        return apply_decisions(points, scores, self.thresholds)


def apply_decisions(points, scores, thresholds):
    """Keep unconfirmed baseline-green candidates outside learned hard rejection.

    The training labels predominantly cover orange candidates. A low-scoring
    original green point is therefore flagged for review, not silently deleted.
    Removed model candidates remain available in the review UI for restoration.
    """
    if len(points) != len(scores):
        raise ValueError('Candidate and score lengths differ')
    result = []
    for old, score in zip(points, scores):
        p = copy.deepcopy(old)
        p['origin'] = 'automatic'
        p['removed'] = False
        p['baseline_status'] = old['status']
        p['model_score'] = round(float(score), 6)
        if not np.isfinite(score) or not 0 <= score <= 1:
            raise ValueError('Invalid model score')
        priority = thresholds.get('review_priority_below', thresholds['reject_below'])
        p['review_priority'] = bool(score < priority)
        if old['status'] == 'accepted':
            if score < priority:
                p['status'] = 'review'
                p['reason'] = '模型与原检测不一致，需要复核；' + old.get('reason', '')
                p['model_decision'] = 'review_original_green'
            else:
                p['model_decision'] = 'preserve_original_green'
        elif score < thresholds['reject_below']:
            p['removed'] = True
            p['model_rejected'] = True
            p['model_decision'] = 'reject'
            p['reason'] = '模型自动排除（可恢复）：疑似光晕/亮缝；' + old.get('reason', '')
        elif score >= thresholds['accept_at_least']:
            p['status'] = 'accepted'
            p['model_decision'] = 'accept'
            p['reason'] = '模型通过；' + old.get('reason', '')
        else:
            p['status'] = 'review'
            p['model_decision'] = 'review'
            p['reason'] = '模型仍不确定；' + old.get('reason', '')
        # A learned acceptance is not a human confirmation; no confirmed=True.
        result.append(p)
    return result
