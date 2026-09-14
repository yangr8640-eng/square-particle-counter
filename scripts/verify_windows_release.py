"""Verify this repository's Windows release using only runner Python's stdlib.

The actual model, native libraries and HTTP app run under the ZIP's Python.
No photographs, tokens, installed Python packages or external test services are
used. --archive/--verify-only support local archive checks without Windows.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import signal
import stat
import struct
import subprocess
import sys
import tempfile
import time
from urllib.parse import quote, urljoin, urlsplit
from urllib.request import ProxyHandler, Request, build_opener, urlopen
import uuid
import zipfile
import zlib

ASSET = 'ParticleCounter-Windows-x64-v4.zip'
PACKAGE_NAME = 'ParticleCounter-Windows-x64-v4'
MAX_ARCHIVE_BYTES = 512 * 1024 * 1024
MAX_UNPACKED_BYTES = 2 * 1024 * 1024 * 1024
REPO_ROOT = Path(__file__).resolve().parent.parent


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_source(repository, tag):
    require(bool(re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', repository)), 'Invalid repository; expected owner/repo')
    require(all(part not in {'.', '..'} for part in repository.split('/')), 'Invalid repository components')
    require(bool(re.fullmatch(r'v?\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?', tag)), 'Invalid release tag; expected v1.0.0 or another semantic version')
    return f'https://github.com/{repository}/releases/download/{quote(tag, safe="")}/{ASSET}'


def expected_checksum(path):
    hashes = []
    for line in Path(path).read_text(encoding='utf-8-sig').splitlines():
        match = re.fullmatch(r'([0-9a-fA-F]{64})\s+\*?(.+)', line.strip())
        if match and match.group(2) == ASSET:
            hashes.append(match.group(1).lower())
    require(len(hashes) == 1, f'Expected exactly one {ASSET} entry in SHA256SUMS.txt')
    return hashes[0]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def download_release(url, target):
    deadline = time.monotonic() + 180
    total = 0
    with urlopen(Request(url, headers={'User-Agent': 'particle-counter-release-check'}), timeout=30) as response, target.open('xb') as output:
        require(urlsplit(response.url).scheme == 'https', 'Release download must remain HTTPS')
        while chunk := response.read(1024 * 1024):
            total += len(chunk)
            require(total <= MAX_ARCHIVE_BYTES, 'Release archive exceeds download size limit')
            require(time.monotonic() < deadline, 'Release download exceeded three minutes')
            output.write(chunk)


def safe_relative(name):
    require(isinstance(name, str) and name and '\\' not in name and ':' not in name,
            'Unsafe archive or manifest path')
    path = PurePosixPath(name)
    require(not path.is_absolute() and '..' not in path.parts and '.' not in name.split('/'),
            'Archive or manifest path escapes its package')
    for part in path.parts:
        require(part and part == part.rstrip(' .') and not any(ord(char) < 32 for char in part),
                'Ambiguous Windows archive filename')
        require(not re.fullmatch(r'(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?', part, re.I),
                'Reserved Windows archive filename')
    return path


def extract_safely(archive, destination):
    destination.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(archive) as zipped:
        entries = zipped.infolist()
        require(len(entries) <= 30_000, 'Too many archive entries')
        require(sum(entry.file_size for entry in entries) <= MAX_UNPACKED_BYTES, 'Unpacked archive exceeds size limit')
        seen = set()
        for entry in entries:
            relative = safe_relative(entry.filename)
            key = str(relative).casefold()
            require(key not in seen, 'Duplicate or case-colliding Windows archive path')
            seen.add(key)
            mode = entry.external_attr >> 16
            require(not stat.S_ISLNK(mode), 'Archive symlinks are not allowed')
            require(not (mode & 0o170000) or stat.S_ISREG(mode) or stat.S_ISDIR(mode), 'Archive special files are not allowed')
            require(not entry.flag_bits & 1, 'Encrypted archive entries are not allowed')
            path = destination.joinpath(*relative.parts)
            require(path.resolve().is_relative_to(destination.resolve()), 'Unsafe archive destination')
        # No archive entry is extracted until all names have passed validation.
        for entry in entries:
            path = destination.joinpath(*safe_relative(entry.filename).parts)
            if entry.is_dir():
                path.mkdir(parents=True, exist_ok=True)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                with zipped.open(entry) as source, path.open('xb') as output:
                    shutil.copyfileobj(source, output, 1024 * 1024)
    root = destination / PACKAGE_NAME
    require(root.is_dir() and set(destination.iterdir()) == {root}, 'Unexpected release package root')
    return root


def verify_manifest(root):
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    require(manifest.get('target') == 'windows-x64', 'Release manifest is not for Windows x64')
    require(manifest.get('model_version') == 'square-core-filter-4.0', 'Expected the v4 release model')
    require(manifest.get('contains_runtime') is True, 'Release is missing its runtime declaration')
    require(manifest.get('contains_training_images') is False, 'Release must not contain training images')
    files = manifest.get('files')
    require(isinstance(files, dict) and files, 'Release manifest has no file hashes')
    required = {'runtime/python.exe', 'program/start.py', 'program/local_app/model/particle_filter.joblib',
                'program/local_app/model/particle_filter.json', 'program/local_app/static/index.html'}
    require(required.issubset(files), 'Required application/runtime/model files are missing from manifest')
    for name, digest in files.items():
        path = root.joinpath(*safe_relative(name).parts)
        require(bool(re.fullmatch(r'[0-9a-f]{64}', digest)), f'Invalid manifest digest: {name}')
        require(path.is_file() and not path.is_symlink(), f'Missing manifest file: {name}')
        require(sha256(path) == digest, f'Manifest hash mismatch: {name}')
    actual = {path.relative_to(root).as_posix() for path in root.rglob('*') if path.is_file()}
    require(actual == set(files) | {'manifest.json'}, 'Release contains unlisted or missing files')
    model = root / 'program/local_app/model/particle_filter.joblib'
    require(sha256(model) == manifest.get('model_sha256'), 'Manifest model hash mismatch')
    return manifest


def synthetic_png():
    """Two stained square cores, created directly as RGB PNG without libraries."""
    size = 160
    pixels = bytearray([245] * (size * size * 3))
    for cx, cy, core in [(45, 80, (195, 220, 160)), (105, 80, (218, 225, 205))]:
        for radius, color in [(11, (65, 75, 70)), (7, core)]:
            for y in range(cy - radius, cy + radius + 1):
                for x in range(cx - radius, cx + radius + 1):
                    offset = (y * size + x) * 3
                    pixels[offset:offset + 3] = bytes(color)
    raw = b''.join(b'\0' + pixels[y * size * 3:(y + 1) * size * 3] for y in range(size))
    def chunk(kind, data):
        return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data) & 0xffffffff)
    return b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', size, size, 8, 2, 0, 0, 0)) + chunk(b'IDAT', zlib.compress(raw)) + chunk(b'IEND', b'')


def png_size(payload):
    require(payload[:8] == b'\x89PNG\r\n\x1a\n' and payload[12:16] == b'IHDR', 'Expected a PNG output')
    return struct.unpack('>II', payload[16:24])


SCORE_PROBE = r'''
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import numpy as np
from local_app.pipeline import prepare_model
from particle_counter import Config, detect, load_image
image = load_image(Path(sys.argv[2]))
points, excluded, _ = detect(image, Config())
assert len(points) >= 1, 'Synthetic cores did not enter the classifier'
model = prepare_model(Path(sys.argv[1]) / 'local_app/model/particle_filter.joblib')
original = model.model.predict_proba
calls = []
def counted(features):
    calls.append(len(features))
    return original(features)
model.model.predict_proba = counted
scores = model.score(image, points)
assert len(calls) == 1 and calls[0] == len(points), 'Classifier prediction was bypassed'
assert np.isfinite(scores).all() and ((0 <= scores) & (scores <= 1)).all()
print(json.dumps({'model_version': model.metadata['model_version'], 'candidate_count': len(points),
                  'classifier_calls': len(calls), 'scores': [float(x) for x in scores],
                  'python': str(Path(sys.executable).resolve())}))
'''


def probe_model(executable, program, image_path):
    result = subprocess.run([str(executable), '-I', '-B', '-X', 'utf8', '-c', SCORE_PROBE, str(program), str(image_path)],
                            cwd=program.parent, text=True, encoding='utf-8', errors='replace',
                            capture_output=True, timeout=90, check=False)
    require(result.returncode == 0, 'Bundled model/feature probe failed: ' + (result.stderr or result.stdout)[-3000:])
    probe = json.loads(result.stdout.strip().splitlines()[-1])
    require(Path(probe['python']).resolve() == executable.resolve(), 'Model probe used a different Python runtime')
    require(probe['model_version'] == 'square-core-filter-4.0', 'Model probe did not use v4')
    return {key: value for key, value in probe.items() if key != 'python'}


def request_local(url, *, payload=None, content_type=None):
    parsed = urlsplit(url)
    require(parsed.scheme == 'http' and parsed.hostname == '127.0.0.1' and parsed.port,
            'Application URL is not loopback HTTP')
    headers = {}
    if payload is not None:
        headers = {'Origin': f'http://127.0.0.1:{parsed.port}', 'Content-Type': content_type or 'application/octet-stream'}
    # Never send synthetic local uploads through any runner HTTP proxy.
    opener = build_opener(ProxyHandler({}))
    with opener.open(Request(url, data=payload, headers=headers), timeout=30) as response:
        body = response.read(32 * 1024 * 1024 + 1)
        require(len(body) <= 32 * 1024 * 1024, 'Unexpectedly large synthetic response')
        return response.status, body


def local_link(base, link):
    require(isinstance(link, str) and link.startswith('/') and not link.startswith('//'), 'Invalid application result link')
    result = urljoin(base, link)
    require(urlsplit(result).netloc == urlsplit(base).netloc, 'Application result link changed origin')
    return result


def multipart(png):
    boundary = 'particlecheck' + uuid.uuid4().hex
    data = bytearray()
    for name, content in [('synthetic 检查.png', png), ('broken.png', b'not an image: generated failure fixture')]:
        data.extend((f'--{boundary}\r\nContent-Disposition: form-data; name="files"; filename="{name}"\r\n'
                     'Content-Type: image/png\r\n\r\n').encode('utf-8'))
        data.extend(content + b'\r\n')
    data.extend(f'--{boundary}--\r\n'.encode('ascii'))
    return bytes(data), 'multipart/form-data; boundary=' + boundary


def assert_record(record, png):
    require(record['sha256'] == hashlib.sha256(png).hexdigest(), 'Uploaded source hash changed')
    require(record['original_name'] == 'synthetic 检查.png', 'Original display name changed')
    require(bool(re.fullmatch(r'[0-9a-f]{12}_image_001\.png', record['name'])), 'Export name is not batch-specific')
    require((record['width'], record['height']) == (160, 160), 'Original image dimensions changed')
    points = record['points']
    require(isinstance(points, list) and points and record['baseline_count'] >= 1, 'Synthetic inference produced no candidates')
    active = [point for point in points if not point.get('removed', False)]
    require(record['count'] == len(active), 'Result count disagrees with active points')
    require(record['review_count'] == sum(point['status'] == 'review' for point in active), 'Review count disagrees with active points')
    require(len({point['id'] for point in points}) == len(points), 'Duplicate output point IDs')
    for point in points:
        require(all(isinstance(point[axis], (int, float)) and math.isfinite(point[axis]) and 0 <= point[axis] < 160 for axis in ('x', 'y')),
                'Point coordinates are not valid original-image coordinates')
        require(not point.get('confirmed') and point.get('origin') != 'manual' and not point.get('human_decision'),
                'Automatic output incorrectly claims a human label')
        if point.get('suggestion') or point.get('model_rejected'):
            require(point.get('removed') is True, 'Unconfirmed gray suggestion entered the initial count')


def verify_http(base, png):
    code, body = request_local(base + 'health')
    health = json.loads(body)
    require(code == 200 and health.get('status') == 'ok' and health.get('offline') is True, 'Application health failed')
    require(health.get('model', {}).get('name') == 'square-core-filter-4.0', 'Application health did not identify v4')
    require(request_local(base)[0] == 200, 'Upload UI did not load')
    body, content_type = multipart(png)
    code, body = request_local(base + 'api/jobs', payload=body, content_type=content_type)
    require(code == 202, 'Synthetic batch was not queued')
    queued = json.loads(body)
    status_url = local_link(base, queued['status_url'])
    deadline = time.monotonic() + 120
    while True:
        _, body = request_local(status_url)
        job = json.loads(body)
        if job['status'] in {'completed', 'failed'}:
            break
        require(time.monotonic() < deadline, 'Synthetic job did not finish within two minutes')
        time.sleep(.2)
    require(job['status'] == 'completed' and job.get('failed_images') == 1, 'Mixed batch did not preserve its successful image')
    require(job['progress']['completed'] == job['progress']['total'] == 2, 'Batch progress is inconsistent')
    success = [item for item in job['images'] if item['status'] == 'completed']
    failed = [item for item in job['images'] if item['status'] == 'failed']
    require(len(success) == len(failed) == 1 and failed[0].get('error'), 'Missing per-image failure details')
    _, body = request_local(local_link(base, job['json_url']))
    data = json.loads(body)
    require(data['coordinateSystem'] == 'original-image-pixels' and len(data['images']) == 1, 'Invalid full JSON result')
    record = data['images'][0]
    assert_record(record, png)
    require(success[0]['count'] == record['count'], 'Job count and complete JSON disagree')
    _, annotated = request_local(local_link(base, success[0]['annotated_url']))
    require(png_size(annotated) == (160, 160), 'Annotated PNG is not original resolution')
    _, csv_payload = request_local(local_link(base, job['csv_url']))
    require(record['name'] in csv_payload.decode('utf-8-sig'), 'CSV lost the image identity')
    _, archive = request_local(local_link(base, job['zip_url']))
    with zipfile.ZipFile(io.BytesIO(archive)) as zipped:
        names = zipped.namelist()
        for name in names:
            safe_relative(name)
        require(len(names) == len(set(names)) and 'review.html' in names, 'Offline ZIP is invalid')
        source = str(safe_relative(record['src']))
        require(source in names and zipped.read(source) == png, 'Offline review image path/source is invalid')
        require(zipped.read('images/' + record['name']) == png, 'Offline ZIP changed the original upload')
        offline_data = json.loads(zipped.read('detections.json'))
        require(offline_data['images'][0] == record, 'Offline/full JSON records differ')
        review = zipped.read('review.html').decode('utf-8')
        match = re.search(r'const DATA = (.*?);\n', review)
        require(match is not None, 'Offline review JSON is missing')
        embedded = json.loads(match.group(1))
        require(embedded['images'][0]['src'] == source, 'Offline review uses a different image source')
        require(not urlsplit(source).scheme and not source.startswith('/'), 'Offline review depends on an external URL')
        require(not any(name.startswith(('program/', 'runtime/')) for name in names), 'Result ZIP contains application files')
    return dict(status=job['status'], successful_images=1, failed_images=1, count=record['count'],
                point_records=len(record['points']), review_count=record['review_count'],
                source_hash_verified=True, offline_zip_verified=True, no_human_labels=True)


def stop_process(process):
    if process.poll() is not None:
        return
    if os.name == 'nt':
        try:
            process.send_signal(signal.CTRL_BREAK_EVENT)
            process.wait(timeout=3)
            return
        except (OSError, subprocess.TimeoutExpired):
            pass
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def exercise_runtime(root, work, executable=None):
    executable = Path(executable) if executable is not None else root / 'runtime/python.exe'
    program = root / 'program'
    png = synthetic_png()
    image_path = work / '合成 检查.png'
    image_path.write_bytes(png)
    probe = probe_model(executable, program, image_path)
    logfile = work / 'server.log'
    command = [str(executable), '-I', '-B', '-X', 'utf8', str(program / 'start.py'),
               '--no-browser', '--port', '0', '--data-dir', str(work / '任务 数据')]
    flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
    with logfile.open('wb') as log:
        process = subprocess.Popen(command, cwd=work, stdout=log, stderr=subprocess.STDOUT, creationflags=flags)
        try:
            deadline = time.monotonic() + 90
            while True:
                output = logfile.read_text(encoding='utf-8', errors='replace')
                require(process.poll() is None, 'Application exited during startup: ' + output[-3000:])
                match = re.search(r'http://127\.0\.0\.1:(\d{1,5})/', output)
                if match:
                    base = match.group(0)
                    break
                require(time.monotonic() < deadline, 'Application did not print a local URL: ' + output[-3000:])
                time.sleep(.2)
            result = verify_http(base, png)
        except Exception as exc:
            raise RuntimeError(str(exc) + '\nApplication log: ' + logfile.read_text(encoding='utf-8', errors='replace')[-3000:]) from exc
        finally:
            stop_process(process)
    return dict(model_probe=probe, application=result, server_stopped=process.poll() is not None)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', default=os.environ.get('GITHUB_REPOSITORY', ''))
    parser.add_argument('--tag', default='v1.0.0')
    parser.add_argument('--checksums', type=Path, default=REPO_ROOT / 'SHA256SUMS.txt')
    parser.add_argument('--archive', type=Path, help='Verify an existing local Windows ZIP instead of downloading it')
    parser.add_argument('--verify-only', action='store_true', help='Verify extraction/manifest only; does not claim Windows execution')
    args = parser.parse_args(argv)
    try:
        url = validate_source(args.repository, args.tag)
        digest = expected_checksum(args.checksums)
        if not args.verify_only:
            require(os.name == 'nt', 'Windows runtime execution requires Windows; use --verify-only for local archive checks')
        with tempfile.TemporaryDirectory(prefix='PC 测试 ') as temporary:
            work = Path(temporary).resolve()
            archive = args.archive.resolve() if args.archive else work / ASSET
            if args.archive is None:
                download_release(url, archive)
            require(archive.stat().st_size <= MAX_ARCHIVE_BYTES, 'Archive exceeds the size limit')
            require(sha256(archive) == digest, 'Release ZIP hash does not match the repository SHA256SUMS.txt')
            root = extract_safely(archive, work / '接收端 解压目录')
            manifest = verify_manifest(root)
            result = dict(status='passed', repository=args.repository, tag=args.tag, asset=ASSET,
                          archive_sha256=digest, manifest_files=len(manifest['files']),
                          model_version=manifest['model_version'], windows_runtime_executed=not args.verify_only)
            if not args.verify_only:
                result.update(exercise_runtime(root, work))
            print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
        return 0
    except Exception as exc:
        print(json.dumps(dict(status='failed', error=str(exc)), ensure_ascii=False), flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
