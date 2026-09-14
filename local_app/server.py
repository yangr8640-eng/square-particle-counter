"""Flask API with bounded uploads, serial background inference and private jobs."""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
import threading
from urllib.parse import urlsplit
import uuid
import warnings
import zipfile

from flask import Flask, jsonify, request, send_file, send_from_directory
from PIL import Image, ImageOps, UnidentifiedImageError
from werkzeug.exceptions import RequestEntityTooLarge

from .pipeline import default_model_path, run_one_image

DEFAULT_LIMITS = dict(max_files=50, max_file_bytes=25 * 1024 * 1024,
                      max_total_bytes=250 * 1024 * 1024, max_pixels=25_000_000,
                      max_pending_jobs=3)
FORMATS = {'.png': 'PNG', '.jpg': 'JPEG', '.jpeg': 'JPEG', '.tif': 'TIFF',
           '.tiff': 'TIFF', '.bmp': 'BMP'}
JOB_ID = re.compile(r'^[0-9a-f]{32}$')
APP_VERSION = '1.0.0'


class UploadError(ValueError):
    pass


def _app_review_template():
    """Use display names in this app's page without changing stored identities."""
    from count_particles_v2 import make_review_template
    template = make_review_template()
    anchor = 'const images = Array.isArray(DATA.images) ? DATA.images : [];'
    replacements = [
        (anchor, anchor + '\n  const displayName = im => String(im.original_name || im.name);'
         + '\n  ' + r'''const spreadsheetName = im => {const name=displayName(im);return /^\s*[=+\-@]/.test(name)?"'"+name:name;};''', 1),
        ('row.setAttribute("aria-label","切换至 "+im.name)',
         'row.setAttribute("aria-label","切换至 "+displayName(im))', 1),
        ('[im.name,...keys.map(k=>c[k])]', '[displayName(im),...keys.map(k=>c[k])]', 1),
        ('option.textContent=im.name;', 'option.textContent=displayName(im);', 1),
        ('rows.push([im.name,...keys.map((k,j)=>', 'rows.push([spreadsheetName(im),...keys.map((k,j)=>', 1),
        ('images[current].name', 'displayName(images[current])', 8),
    ]
    for before, after, expected in replacements:
        if template.count(before) != expected:
            raise ValueError('Local review display-name template changed; check identity preservation')
        template = template.replace(before, after)
    return template


def _validate_filename(filename):
    if not filename or len(filename) > 255 or any(ord(char) < 32 for char in filename):
        raise UploadError('图片文件名无效，请重命名后上传。')
    if '/' in filename or '\\' in filename or filename in {'.', '..'} or ':' in filename:
        raise UploadError('文件名不能包含路径，请直接选择图片文件。')
    suffix = Path(filename).suffix.lower()
    if suffix not in FORMATS:
        raise UploadError('暂不支持此文件类型，请上传 PNG、JPG、TIFF 或 BMP 图片。')
    return suffix


def _validate_image(path, suffix, limits):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(path) as image:
                if image.format != FORMATS[suffix]:
                    raise UploadError('图片内容与文件后缀不一致，请重新导出图片后上传。')
                width, height = image.size
                if width < 1 or height < 1 or width * height > limits['max_pixels']:
                    raise UploadError(f'图片像素过大，单张最多支持 {limits["max_pixels"]:,} 像素。')
                if getattr(image, 'n_frames', 1) != 1:
                    raise UploadError('暂不支持动图或多页图片，请导出单张图片后上传。')
                image.verify()
            with Image.open(path) as image:
                image.load()  # Also validate JPEG's compressed pixel stream.
        return width, height
    except UploadError:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise UploadError('图片像素过大，请缩小图片后上传。') from None
    except (OSError, ValueError, SyntaxError, UnidentifiedImageError):
        raise UploadError('无法读取这张图片，文件可能损坏，请重新导出后上传。') from None


class JobManager:
    def __init__(self, data_dir=None, *, pipeline=run_one_image, model_path=None, limits=None):
        self._temporary = tempfile.TemporaryDirectory(prefix='particle-counter-') if data_dir is None else None
        self.root = Path(self._temporary.name if self._temporary else data_dir).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.pipeline = pipeline
        self.model_path = Path(model_path).resolve() if model_path is not None else default_model_path()
        self.limits = dict(DEFAULT_LIMITS, **(limits or {}))
        self.jobs = {}
        self.lock = threading.RLock()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='particle-worker')
        self.futures = {}
        self.closed = False
        try:
            metadata = json.loads(self.model_path.with_suffix('.json').read_text(encoding='utf-8'))
            self.model_name = metadata.get('model_version', '本地模型')
        except (OSError, ValueError):
            self.model_name = '本地模型'

    def close(self):
        with self.lock:
            if self.closed:
                return
            self.closed = True
        self.executor.shutdown(wait=True, cancel_futures=True)
        if self._temporary:
            self._temporary.cleanup()

    def reserve(self):
        with self.lock:
            pending = sum(job['status'] in {'uploading', 'queued', 'running'} for job in self.jobs.values())
            if self.closed or pending >= self.limits['max_pending_jobs']:
                raise UploadError('当前排队任务较多，请等待已有任务完成后再上传。')
            job_id = uuid.uuid4().hex
            directory = self.root / job_id
            (directory / 'bundle' / 'images').mkdir(parents=True)
            self.jobs[job_id] = dict(job_id=job_id, status='uploading', progress=dict(completed=0, total=0, current_name=None),
                                     images=[], error=None, review_url=None, json_url=None, csv_url=None, zip_url=None,
                                     created_at=datetime.now(timezone.utc).isoformat(), _directory=directory,
                                     _allowed=set(), _records=[])
            return job_id

    def discard(self, job_id):
        with self.lock:
            job = self.jobs.pop(job_id, None)
        if job:
            shutil.rmtree(job['_directory'], ignore_errors=True)

    def receive(self, files):
        if not files:
            raise UploadError('请先选择至少一张图片。')
        if len(files) > self.limits['max_files']:
            raise UploadError(f'一次最多上传 {self.limits["max_files"]} 张图片，请分批处理。')
        job_id = self.reserve()
        try:
            job = self.jobs[job_id]
            total_bytes = 0
            for index, upload in enumerate(files, 1):
                suffix = _validate_filename(upload.filename)
                name = f'{job_id[:12]}_image_{index:03d}{suffix}'
                path = job['_directory'] / 'bundle' / 'images' / name
                size = 0
                with path.open('xb') as output:
                    while chunk := upload.stream.read(1024 * 1024):
                        size += len(chunk)
                        total_bytes += len(chunk)
                        if size > self.limits['max_file_bytes']:
                            raise UploadError(f'单张图片不能超过 {self.limits["max_file_bytes"] // (1024 * 1024)} MB。')
                        if total_bytes > self.limits['max_total_bytes']:
                            raise UploadError('本批图片总大小超过限制，请分批上传。')
                        output.write(chunk)
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                item_id = f'image_{index:03d}'
                item = dict(id=item_id, name=upload.filename, export_name=name, status='queued',
                    count=None, review_count=None, preview_url=None, annotated_url=None,
                    original_url=f'/api/jobs/{job_id}/files/images/{name}', error=None,
                    width=None, height=None, sha256=digest)
                try:
                    item['width'], item['height'] = _validate_image(path, suffix, self.limits)
                    job['_allowed'].add('images/' + name)
                except UploadError as exc:
                    item.update(status='failed', error=str(exc), original_url=None)
                    path.unlink()
                job['images'].append(item)
            if not any(item['status'] == 'queued' for item in job['images']):
                raise UploadError(job['images'][0]['error'])
            with self.lock:
                job['progress']['total'] = len(files)
                job['status'] = 'queued'
                self.futures[job_id] = self.executor.submit(self._run, job_id)
            return job_id
        except BaseException:
            self.discard(job_id)
            raise

    def snapshot(self, job_id):
        with self.lock:
            job = self.jobs.get(job_id)
            return copy.deepcopy({key: value for key, value in job.items() if not key.startswith('_')}) if job else None

    def _run(self, job_id):
        with self.lock:
            job = self.jobs[job_id]
            job['status'] = 'running'
        bundle = job['_directory'] / 'bundle'
        metadata = {}
        for item in job['images']:
            if item['status'] == 'failed':
                with self.lock:
                    job['progress']['completed'] += 1
                continue
            with self.lock:
                item['status'] = 'running'
                job['progress']['current_name'] = item['name']
            try:
                source = bundle / 'images' / item['export_name']
                record = self.pipeline(source, bundle, model_path=self.model_path)
                current_metadata = record.pop('model_metadata', {})
                if metadata and current_metadata != metadata:
                    raise ValueError('Model metadata changed during one upload batch')
                metadata = current_metadata or metadata
                annotated_relative = 'annotated/' + source.stem + '_counted.png'
                annotated = bundle / annotated_relative
                with Image.open(annotated) as annotated_image:
                    if annotated_image.size != (record['width'], record['height']):
                        raise ValueError('Annotation is not original resolution')
                with Image.open(source) as original_image:
                    preview = ImageOps.exif_transpose(original_image).convert('RGB')
                    if preview.size != (record['width'], record['height']):
                        raise ValueError('Prediction coordinates do not match the uploaded image')
                    display_relative = 'images/' + item['export_name']
                    # Offline browser review must work for TIFF/BMP too. Keep
                    # source bytes and their identity unchanged; only the visual
                    # source uses an oriented, full-resolution RGB PNG copy.
                    if source.suffix.lower() not in {'.png', '.jpg', '.jpeg'}:
                        display_relative = 'display_images/' + source.stem + '.png'
                        (bundle / 'display_images').mkdir(exist_ok=True)
                        preview.save(bundle / display_relative, 'PNG')
                    preview.thumbnail((1600, 1600))
                    thumbnail_relative = 'thumbnails/' + source.stem + '.jpg'
                    (bundle / 'thumbnails').mkdir(exist_ok=True)
                    preview.save(bundle / thumbnail_relative, 'JPEG', quality=90)
                record.update(name=item['export_name'], original_name=item['name'],
                              src=display_relative, sha256=item['sha256'])
                # Count using the unchanged point provenance, not a prediction label.
                active = [point for point in record['points'] if not point.get('removed', False)]
                record['count'] = len(active)
                record['review_count'] = sum(point.get('status') == 'review' for point in active)
                json_relative = 'per_image/' + source.stem + '.json'
                (bundle / 'per_image').mkdir(exist_ok=True)
                (bundle / json_relative).write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
                with self.lock:
                    job['_records'].append(record)
                    job['_allowed'].update({annotated_relative, thumbnail_relative, json_relative, display_relative})
                    item.update(status='completed', count=record['count'], review_count=record['review_count'],
                                model_rejected_count=int(record.get('model_rejected_count', 0)),
                                dim_proposal_count=int(record.get('dim_proposal_count', 0)),
                                preview_url=f'/api/jobs/{job_id}/files/{thumbnail_relative}',
                                display_url=f'/api/jobs/{job_id}/files/{display_relative}',
                                annotated_url=f'/api/jobs/{job_id}/files/{annotated_relative}',
                                json_url=f'/api/jobs/{job_id}/files/{json_relative}')
            except Exception:
                logging.getLogger(__name__).exception('Image processing failed: job=%s image=%s', job_id, item['export_name'])
                with self.lock:
                    item.update(status='failed', error='这张图片处理失败，请尝试重新导出图片；其余图片会继续处理。')
            finally:
                with self.lock:
                    job['progress']['completed'] += 1
        try:
            self._write_bundle(job, metadata)
            with self.lock:
                failed = sum(item['status'] == 'failed' for item in job['images'])
                job['failed_images'] = failed
                job['status'] = 'completed' if job['_records'] else 'failed'
                job['error'] = f'{failed} 张图片处理失败，其余结果可下载。' if failed and job['_records'] else ('图片处理失败，请检查图片或本地模型。' if failed else None)
                job['progress']['current_name'] = None
        except Exception:
            logging.getLogger(__name__).exception('Bundle generation failed: job=%s', job_id)
            with self.lock:
                job['status'] = 'failed'
                job['error'] = '生成下载文件失败，请确认本地磁盘空间足够后重试。'
                job['progress']['current_name'] = None

    def _write_bundle(self, job, metadata):
        bundle = job['_directory'] / 'bundle'
        records = job['_records']
        failures = [dict(name=item['name'], export_name=item['export_name'], error=item['error'])
                    for item in job['images'] if item['status'] == 'failed']
        data = dict(metadata, schemaVersion=1, coordinateSystem='original-image-pixels',
                    reviewPointsIncludedInCount=True, job_id=job['job_id'], created_at=job['created_at'],
                    images=records, failed_images=failures)
        (bundle / 'detections.json').write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
        with (bundle / 'counts.csv').open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['export_name', 'original_name', 'count_including_review', 'review_included', 'model_rejected_not_counted', 'dim_proposals_not_counted'])
            for record in records:
                # Original display names can be spreadsheet formulas; export
                # them as text while leaving JSON names and source bytes intact.
                name = record['original_name']
                if name.lstrip().startswith(('=', '+', '-', '@')):
                    name = "'" + name
                writer.writerow([record['name'], name, record['count'], record['review_count'],
                                 record.get('model_rejected_count', 0), record.get('dim_proposal_count', 0)])
        payload = json.dumps(data, ensure_ascii=False, separators=(',', ':')).replace('<', '\\u003c')
        template = _app_review_template()
        (bundle / 'review.html').write_text(template.replace('__PARTICLE_DATA_JSON__', payload), encoding='utf-8')
        manifest = dict(job_id=job['job_id'], uploaded_files=[dict(original_name=item['name'], name=item['export_name'],
            sha256=item['sha256'], included_in_bundle='images/' + item['export_name'] in job['_allowed']) for item in job['images']])
        (bundle / 'upload_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
        with self.lock:
            job['_allowed'].update({'detections.json', 'counts.csv', 'review.html', 'upload_manifest.json'})
            allowed = sorted(job['_allowed'])
        archive = job['_directory'] / 'results.zip'
        with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as zipped:
            for relative in allowed:
                source = bundle / relative
                if source.is_file() and not source.is_symlink():
                    zipped.write(source, relative)
        prefix = f'/api/jobs/{job["job_id"]}'
        with self.lock:
            job.update(review_url=prefix + '/files/review.html', json_url=prefix + '/files/detections.json',
                       csv_url=prefix + '/files/counts.csv', zip_url=prefix + '/download.zip')


def create_app(data_dir=None, *, pipeline=run_one_image, model_path=None, limits=None, static_dir=None):
    app = Flask(__name__, static_folder=None)
    manager = JobManager(data_dir, pipeline=pipeline, model_path=model_path, limits=limits)
    app.extensions['particle_jobs'] = manager
    app.config['MAX_CONTENT_LENGTH'] = manager.limits['max_total_bytes'] + 1024 * 1024
    app.config['MAX_FORM_PARTS'] = manager.limits['max_files'] + 10
    app.config['MAX_FORM_MEMORY_SIZE'] = 1024 * 1024
    static_root = Path(static_dir) if static_dir is not None else Path(__file__).with_name('static')

    def error(message, status):
        return jsonify(error=message), status

    @app.before_request
    def protect_local_server():
        try:
            parsed = urlsplit('http://' + request.host)
            if (parsed.hostname not in {'localhost', '127.0.0.1'} or parsed.username or parsed.password
                    or parsed.path or parsed.query or parsed.fragment):
                return error('仅允许通过本机地址打开程序。', 403)
            _ = parsed.port
        except ValueError:
            return error('本机地址无效。', 403)
        if request.headers.get('Sec-Fetch-Site') == 'cross-site':
            return error('已拒绝来自其他网站的请求，请在本地程序页面操作。', 403)
        if request.method not in {'GET', 'HEAD', 'OPTIONS'}:
            origin = request.headers.get('Origin')
            if origin != request.host_url.rstrip('/'):
                return error('已拒绝跨来源操作，请从本地程序页面上传。', 403)

    @app.after_request
    def local_headers(response):
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['Referrer-Policy'] = 'no-referrer'
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['Cross-Origin-Resource-Policy'] = 'same-origin'
        response.headers['Cache-Control'] = 'no-store'
        response.headers['Content-Security-Policy'] = "default-src 'self'; img-src 'self' data: blob:; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; connect-src 'self'; frame-ancestors 'none'; object-src 'none'; base-uri 'none'"
        return response

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(_error):
        return error('上传内容过大或文件过多，请缩小图片或分批上传。', 413)

    @app.errorhandler(404)
    def not_found(_error):
        return error('未找到该任务或文件，请返回本地程序重试。', 404)

    @app.get('/health')
    def health():
        return jsonify(status='ok', offline=True, app_version=APP_VERSION,
                       model=dict(name=manager.model_name, available=manager.model_path.is_file()))

    @app.get('/api/config')
    def config():
        return jsonify(model_name=manager.model_name, limits=manager.limits, offline=True,
                       app_version=APP_VERSION, supported_extensions=list(FORMATS))

    @app.post('/api/jobs')
    def upload():
        try:
            files = request.files.getlist('files')
            if any(key != 'files' for key in request.files):
                raise UploadError('上传字段无效，请在本地程序重新选择图片。')
            job_id = manager.receive(files)
            return jsonify(job_id=job_id, status='queued', status_url=f'/api/jobs/{job_id}'), 202
        except UploadError as exc:
            return error(str(exc), 400)

    def get_job(job_id):
        return manager.snapshot(job_id) if JOB_ID.fullmatch(job_id) else None

    @app.get('/api/jobs/<job_id>')
    def status(job_id):
        snapshot = get_job(job_id)
        return jsonify(snapshot) if snapshot is not None else not_found(None)

    @app.get('/api/jobs/<job_id>/files/<path:relative>')
    def output_file(job_id, relative):
        if not JOB_ID.fullmatch(job_id) or '\\' in relative or '..' in PurePosixPath(relative).parts:
            return not_found(None)
        with manager.lock:
            job = manager.jobs.get(job_id)
            if not job or relative not in job['_allowed']:
                return not_found(None)
            root = job['_directory'] / 'bundle'
            path = root / relative
            if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(root):
                return not_found(None)
        return send_file(path, as_attachment=request.args.get('download') == '1', download_name=path.name)

    @app.get('/api/jobs/<job_id>/download.zip')
    def download(job_id):
        if not JOB_ID.fullmatch(job_id):
            return not_found(None)
        with manager.lock:
            job = manager.jobs.get(job_id)
            if not job or not job.get('zip_url'):
                return not_found(None)
            archive = job['_directory'] / 'results.zip'
        return send_file(archive, as_attachment=True, download_name=f'particle-results-{job_id[:8]}.zip', mimetype='application/zip')

    @app.get('/')
    def index():
        return send_from_directory(static_root, 'index.html')

    @app.get('/static/<path:name>')
    def static_file(name):
        return send_from_directory(static_root, name)

    return app
