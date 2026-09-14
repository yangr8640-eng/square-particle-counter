#!/usr/bin/env python3
"""Count green particles using the reviewed-label filter and optional dim proposals."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import csv
import hashlib
import json
import os
from pathlib import Path
import time
from urllib.parse import quote
from particle_counter import Config, detect, load_image, annotate, write_image
import math
from particle_filter import ParticleFilter
from recovery_candidates import propose_dim_cores


IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}


def resolve_input_files(source, input_list=None):
    """Validate a flat image directory or an exact, ordered UTF-8 selection.

    Only inspect paths; image decoding and model loading happen after the whole
    selection is validated, so invalid lists cannot produce a partial batch.
    """
    source = Path(source).resolve()
    if not source.exists():
        raise ValueError(f'Input does not exist: {source}')
    if input_list is not None:
        if not source.is_dir():
            raise ValueError('--input-list requires --input to be a directory')
        try:
            names = Path(input_list).read_text(encoding='utf-8-sig').splitlines()
        except (OSError, UnicodeError) as exc:
            raise ValueError(f'Cannot read UTF-8 input list: {input_list}: {exc}') from exc
        files = []
        for line_number, name in enumerate(names, 1):
            name = name.strip()
            if not name:
                continue
            relative = Path(name)
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError(f'Input list line {line_number} must stay inside --input: {name}')
            files.append(source / relative)
    elif source.is_dir():
        files = sorted(p for p in source.iterdir()
                       if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    else:
        files = [source]

    if not files:
        raise ValueError('No supported images found (PNG/JPG/JPEG/TIF/TIFF/BMP)')
    seen_files, seen_stems = set(), set()
    for path in files:
        if not path.is_file():
            raise ValueError(f'Input image is missing or is not a file: {path}')
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f'Unsupported image or incomplete download: {path.name}')
        if source.is_dir() and not path.resolve().is_relative_to(source):
            raise ValueError(f'Input image resolves outside --input: {path}')
        stat = path.stat()
        identity = (stat.st_dev, stat.st_ino)
        if identity in seen_files:
            raise ValueError(f'Duplicate input image: {path}')
        seen_files.add(identity)
        # Per-image JSON and annotated images are named by stem. Case folding
        # also prevents overwrites on the usual macOS/Windows filesystems.
        stem = path.stem.casefold()
        if stem in seen_stems:
            raise ValueError(f'Input images have conflicting output stems: {path.stem}')
        seen_stems.add(stem)
    return files


def make_review_template(*, has_report=False):
    template = Path(__file__).with_name('review_template.html').read_text()
    before = 'removed:false, origin:"automatic"'
    if template.count(before) != 1:
        raise ValueError('Review baseline template changed; check removed-point preservation')
    template = template.replace(before, 'removed:Boolean(p.removed), origin:p.origin==="manual"?"manual":"automatic"')
    # Record actual human actions separately from model initialization so a
    # later export cannot silently turn predictions into training labels.
    template = template.replace('p.removed=!p.removed;changed(',
        'p.removed=!p.removed;p.human_decision=p.removed?"delete":"keep";changed(')
    template = template.replace('p.status="accepted";p.confirmed=true;',
        'p.status="accepted";p.confirmed=true;p.human_decision="keep";')
    template = template.replace('const ordered=states[current].map((p,i)=>({p,i}));',
        'const ordered=states[current].map((p,i)=>({p,i})).sort((a,b)=>(a.p.model_score??1)-(b.p.model_score??1)||a.i-b.i);')
    template = template.replace('原始自动数', '模型初始计数')
    template = template.replace('已删除', '未计入')
    template = template.replace('恢复本图', '重置本图')
    template = template.replace('全部恢复', '重置全部')
    template = template.replace('逐粒核对自动识别结果，排除干扰、补充漏检，并导出最终计数。',
        '优化模型结果：橙点按疑似误检优先复核，灰色补漏建议尚未计入。')
    template = template.replace('橙点是绿色目标中的待复核子集，默认计入；红圈仅为干扰候选，默认不计入。如红圈处确为目标，请用“增加漏检”补点。',
        '橙点已计入。灰色标记包含模型排除项与暗颗粒补漏建议，均未计入；悬停查看原因，点击“删除 / 恢复”可纳入。模型通过不等于人工确认。')
    template = template.replace('<span class="badge">离线 · 原图坐标</span>',
        '<span class="badge">模型预测 · 请复核</span>')
    report_link = '<a href="report.html">查看优化报告</a>' if has_report else ''
    template = template.replace('<div class="topline">',
        '<p style="margin-bottom:14px">本页是优化模型重新推理的结果。用于训练的图片不能作为独立测试；你的人工修正文件和原核对页仍单独保留。'
        + report_link + '</p><div class="topline">')
    return template


def write_bundle(output, records, metadata):
    output.mkdir(parents=True, exist_ok=True)
    data = dict(metadata, images=records)
    (output / 'detections.json').write_text(json.dumps(data, ensure_ascii=False, indent=2))
    for im in records:
        (output / (Path(im['name']).stem + '.json')).write_text(json.dumps(im, ensure_ascii=False, indent=2))
    payload = json.dumps(data, ensure_ascii=False, separators=(',', ':')).replace('<', '\\u003c')
    template = make_review_template(has_report=(output / 'report.html').is_file())
    (output / 'review.html').write_text(template.replace('__PARTICLE_DATA_JSON__', payload))
    with (output / 'counts.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image', 'baseline_count', 'v2_count_including_review', 'review_included', 'model_rejected_not_counted', 'dim_proposals_not_counted'])
        for im in records:
            writer.writerow([im['name'], im['baseline_count'], im['count'], im['review_count'], im['model_rejected_count'], im['dim_proposal_count']])
        writer.writerow(['TOTAL'] + [sum(im[k] for im in records) for k in ['baseline_count', 'count', 'review_count', 'model_rejected_count', 'dim_proposal_count']])
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', default='.')
    parser.add_argument('--input-list', help='UTF-8 file with one image filename per line, relative to --input')
    parser.add_argument('--output', default='results_v2')
    parser.add_argument('--model', default=str(Path(__file__).with_name('models') / 'particle_filter_v2' / 'particle_filter.joblib'))
    parser.add_argument('--scale', type=float, default=1.)
    parser.add_argument('--no-recovery', action='store_true', help='Skip dim-core suggestions')
    args = parser.parse_args()
    if not math.isfinite(args.scale) or args.scale <= 0:
        parser.error("scale must be finite and positive")
    source, output = Path(args.input).resolve(), Path(args.output).resolve()
    # Existing v1 corrections stay at their original URL and data version.
    old_output = Path(__file__).with_name('results').resolve()
    if output == old_output or output == source:
        parser.error('Choose a separate output directory; original results are preserved')
    try:
        files = resolve_input_files(source, args.input_list)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    filter_model = ParticleFilter(args.model)
    cfg = Config(scale=args.scale)
    records = []
    output.mkdir(parents=True, exist_ok=True)
    (output / 'annotated').mkdir(exist_ok=True)
    for path in files:
        start = time.perf_counter()
        image = load_image(path)
        base_points, excluded, _ = detect(image, cfg)
        points = filter_model.apply(image, base_points, args.scale)
        proposals = [] if args.no_recovery else propose_dim_cores(image, base_points, cfg)
        for p in proposals:
            p.update(removed=True, suggestion=True, model_decision='dim_proposal')
            p['reason'] = '暗颗粒补漏建议（尚未计入）；' + p['reason']
        points.extend(proposals)
        active = [p for p in points if not p['removed']]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        known = digest in filter_model.metadata['source_image_hashes'].values()
        record = dict(name=path.name, src=quote(os.path.relpath(path, output), safe='/'),
            width=image.shape[1], height=image.shape[0], sha256=digest,
            points=points, excluded=excluded, count=len(active), baseline_count=len(base_points),
            review_count=sum(p['status']=='review' for p in active),
            model_rejected_count=sum(p.get('model_rejected',False) for p in points),
            dim_proposal_count=len(proposals), red_region_count=len(excluded),
            used_in_model_training=known, elapsed_seconds=round(time.perf_counter()-start,2))
        records.append(record)
        write_image(output / 'annotated' / (path.stem+'_counted.png'), annotate(image, active, excluded))
        print(path.name, 'v1',len(base_points),'v2',len(active),'review',record['review_count'],
              'dim_suggestions',len(proposals),f'{record["elapsed_seconds"]}s',flush=True)
    metadata=dict(model_version=filter_model.metadata.get('model_version', 'square-core-filter-2.0'), model_sha256=filter_model.metadata['artifact_sha256'],
        config=asdict(cfg), thresholds=filter_model.thresholds,
        method='原视觉候选检测 + 使用人工修正训练的梯度提升树筛选。灰色模型排除项和暗颗粒补漏建议均未计入；橙点已计入。训练图片的本页结果是拟合输出，泛化证据见按原图分组验证报告。')
    write_bundle(output, records, metadata)
    print('Review:',output/'review.html')


if __name__=='__main__':
    main()
