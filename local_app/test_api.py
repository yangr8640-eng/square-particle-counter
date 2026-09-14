"""Disposable API fixtures only: generated pixels and a fake inference adapter."""
from __future__ import annotations
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import tempfile
import threading
import unittest
from unittest import mock
from types import SimpleNamespace
from urllib.parse import urljoin
import zipfile

from flask.testing import FlaskClient
from PIL import Image, ImageDraw
from werkzeug.datastructures import MultiDict
from local_app.server import create_app


class BufferedClient(FlaskClient):
    def open(self, *args, **kwargs):
        kwargs.setdefault('buffered', True)
        return super().open(*args, **kwargs)


def image_bytes(color='navy', size=(24, 18), format='PNG'):
    stream = io.BytesIO()
    Image.new('RGB', size, color).save(stream, format=format)
    return stream.getvalue()


def fake_pipeline(image_path, output_dir, *, model_path):
    with Image.open(image_path) as source:
        image = source.convert('RGB')
    ImageDraw.Draw(image).rectangle((2, 2, 5, 5), fill='red')
    annotation = output_dir / 'annotated' / (image_path.stem + '_counted.png')
    annotation.parent.mkdir(parents=True, exist_ok=True)
    image.save(annotation)
    points = [dict(id=1, x=3, y=3, origin='automatic', status='review', removed=False),
              dict(id=2, x=9, y=9, origin='automatic', status='review', removed=True,
                   suggestion=True, model_decision='dim_proposal'),
              dict(id=3, x=15, y=12, origin='manual', status='accepted', removed=False,
                   confirmed=True, human_decision='keep')]
    return dict(width=image.width, height=image.height, points=points, excluded=[], count=999,
                baseline_count=2, review_count=999, model_rejected_count=0, dim_proposal_count=1,
                model_metadata=dict(model_version='fixture', model_sha256='fixture-hash',
                                    thresholds={'reject_below': 0}, method='合成数据测试'))


class LocalApiTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='particle-api-tests-')
        self.root = Path(self.temporary.name)
        self.model = self.root / 'fixture.joblib'
        self.model.write_bytes(b'fake model: never deserialized')
        self.model.with_suffix('.json').write_text(json.dumps(dict(model_version='fixture')))
        self.static = self.root / 'static'
        self.static.mkdir()
        (self.static / 'index.html').write_text('<h1>Fixture upload UI</h1>')
        self.app = create_app(self.root / 'jobs', pipeline=fake_pipeline, model_path=self.model, static_dir=self.static)
        self.app.test_client_class = BufferedClient
        self.manager = self.app.extensions['particle_jobs']
        self.client = self.app.test_client()
        self.addCleanup(self.temporary.cleanup)
        self.addCleanup(self.manager.close)

    def post(self, files, client=None, **kwargs):
        form = MultiDict([('files', (io.BytesIO(content), name)) for name, content in files])
        return (client or self.client).post('/api/jobs', data=form, headers={'Origin': 'http://localhost'}, **kwargs)

    def wait(self, job_id):
        self.manager.futures[job_id].result(timeout=10)
        return self.client.get('/api/jobs/' + job_id).get_json()

    def test_batch_exports_original_resolution_provenance_and_offline_zip(self):
        first, second = image_bytes(), image_bytes('green')
        response = self.post([('中文同名.png', first), ('中文同名.png', second)])
        self.assertEqual(response.status_code, 202)
        job = self.wait(response.get_json()['job_id'])
        self.assertEqual(job['status'], 'completed')
        self.assertEqual(job['progress'], {'completed': 2, 'total': 2, 'current_name': None})
        self.assertEqual([item['count'] for item in job['images']], [2, 2])
        self.assertEqual([item['review_count'] for item in job['images']], [1, 1])
        self.assertEqual([item['model_rejected_count'] for item in job['images']], [0, 0])
        self.assertEqual([item['dim_proposal_count'] for item in job['images']], [1, 1])
        data = self.client.get(job['json_url']).get_json()
        self.assertEqual(data['coordinateSystem'], 'original-image-pixels')
        image_names = [job['job_id'][:12] + '_image_001.png', job['job_id'][:12] + '_image_002.png']
        self.assertEqual([record['name'] for record in data['images']], image_names)
        self.assertEqual(data['images'][0]['sha256'], hashlib.sha256(first).hexdigest())
        self.assertEqual(data['images'][1]['sha256'], hashlib.sha256(second).hexdigest())
        points = data['images'][0]['points']
        self.assertNotIn('confirmed', points[0])
        self.assertTrue(points[1]['removed'])
        self.assertTrue(points[1]['suggestion'])
        self.assertEqual(points[2]['human_decision'], 'keep')
        self.assertEqual(points[2]['origin'], 'manual')
        annotated = Image.open(io.BytesIO(self.client.get(job['images'][0]['annotated_url']).data))
        self.assertEqual(annotated.size, (24, 18))
        self.assertEqual(self.client.get(job['images'][0]['original_url']).data, first)
        self.assertIn('image_001.png', self.client.get(job['csv_url']).data.decode('utf-8-sig'))
        review = self.client.get(job['review_url']).data.decode()
        self.assertIn('removed:Boolean(p.removed)', review)
        self.assertIn('human_decision', review)
        self.assertIn('"src":"images/' + image_names[0] + '"', review)
        self.assertNotIn('__PARTICLE_DATA_JSON__', review)
        (self.root / 'jobs' / 'private-history.png').write_bytes(image_bytes('yellow'))
        archive_response = self.client.get(job['zip_url'])
        self.assertIn('attachment;', archive_response.headers['Content-Disposition'])
        with zipfile.ZipFile(io.BytesIO(archive_response.data)) as archive:
            names = archive.namelist()
            self.assertTrue({'review.html', 'detections.json', 'counts.csv', *('images/' + name for name in image_names)}.issubset(names))
            self.assertFalse(any('private-history' in name or name.startswith('/') or '..' in name for name in names))
            self.assertEqual(archive.read('images/' + image_names[0]), first)

    def test_async_queue_and_progress_do_not_block_status_requests(self):
        started, release = threading.Event(), threading.Event()
        def blocked(*args, **kwargs):
            started.set()
            if not release.wait(5):
                raise RuntimeError('Fixture release was not signaled')
            return fake_pipeline(*args, **kwargs)
        self.manager.pipeline = blocked
        try:
            first = self.post([('one.png', image_bytes())]).get_json()['job_id']
            self.assertTrue(started.wait(2))
            second = self.post([('two.png', image_bytes())]).get_json()['job_id']
            state = self.client.get('/api/jobs/' + first).get_json()
            self.assertEqual(state['status'], 'running')
            self.assertEqual(state['progress']['current_name'], 'one.png')
            self.assertEqual(self.client.get('/api/jobs/' + second).get_json()['status'], 'queued')
        finally:
            release.set()
        self.assertEqual(self.wait(first)['status'], 'completed')
        self.assertEqual(self.wait(second)['status'], 'completed')

    def test_oriented_tiff_has_full_resolution_offline_display_without_changing_source(self):
        pixels = Image.new('RGB', (24, 18), 'navy')
        pixels.putpixel((0, 0), (255, 0, 0))
        stream = io.BytesIO()
        pixels.save(stream, format='TIFF', tiffinfo={274: 6})
        original = stream.getvalue()
        response = self.post([('显微图片.tiff', original)])
        self.assertEqual(response.status_code, 202)
        job = self.wait(response.get_json()['job_id'])
        self.assertEqual(job['status'], 'completed')
        record = self.client.get(job['json_url']).get_json()['images'][0]
        exported_stem = job['job_id'][:12] + '_image_001'
        self.assertEqual(record['name'], exported_stem + '.tiff')
        self.assertEqual(record['sha256'], hashlib.sha256(original).hexdigest())
        self.assertEqual(record['src'], 'display_images/' + exported_stem + '.png')
        self.assertEqual((record['width'], record['height']), (18, 24))
        self.assertEqual(self.client.get(job['images'][0]['original_url']).data, original)
        per_image = self.client.get(job['images'][0]['json_url']).get_json()
        self.assertEqual(per_image['src'], record['src'])
        display_url = urljoin(job['review_url'], record['src'])
        self.assertEqual(display_url, job['images'][0]['display_url'])
        with Image.open(io.BytesIO(self.client.get(display_url).data)) as display:
            self.assertEqual(display.format, 'PNG')
            self.assertEqual(display.mode, 'RGB')
            self.assertEqual(display.size, (18, 24))
            self.assertEqual(display.getpixel((17, 0)), (255, 0, 0))
            self.assertNotIn(274, display.getexif())
        with zipfile.ZipFile(io.BytesIO(self.client.get(job['zip_url']).data)) as archive:
            self.assertEqual(archive.read('images/' + record['name']), original)
            offline = archive.read('review.html').decode()
            embedded = json.loads(re.search(r'const DATA = (.*?);\n', offline).group(1))
            self.assertEqual(embedded['images'][0]['src'], record['src'])
            with Image.open(io.BytesIO(archive.read(record['src']))) as display:
                self.assertEqual(display.size, (record['width'], record['height']))
                self.assertEqual(display.format, 'PNG')

    def test_corrupt_image_is_reported_while_valid_image_completes(self):
        response = self.post([('broken.png', b'not a PNG'), ('valid.png', image_bytes())])
        self.assertEqual(response.status_code, 202)
        job = self.wait(response.get_json()['job_id'])
        self.assertEqual(job['status'], 'completed')
        self.assertEqual(job['failed_images'], 1)
        self.assertEqual(job['progress']['completed'], 2)
        self.assertEqual(job['images'][0]['status'], 'failed')
        self.assertIn('无法读取', job['images'][0]['error'])
        self.assertIsNone(job['images'][0]['original_url'])
        self.assertEqual(len(self.client.get(job['json_url']).get_json()['images']), 1)

    def test_completed_image_exposes_automatic_rejections_and_suggestions_separately(self):
        def rejected_candidate(*args, **kwargs):
            record = fake_pipeline(*args, **kwargs)
            record['points'].append(dict(id=4, x=20, y=12, status='review', origin='automatic',
                                         removed=True, model_rejected=True, model_decision='reject'))
            record['model_rejected_count'] = 1
            return record
        self.manager.pipeline = rejected_candidate
        job_id = self.post([('one.png', image_bytes())]).get_json()['job_id']
        item = self.wait(job_id)['images'][0]
        self.assertEqual(item['count'], 2)
        self.assertEqual(item['model_rejected_count'], 1)
        self.assertEqual(item['dim_proposal_count'], 1)

    def test_pipeline_failure_does_not_drop_other_results_or_expose_exception(self):
        def fail_one(image_path, output_dir, **kwargs):
            if image_path.stem.endswith('_image_001'):
                raise RuntimeError('/private/path/with-secret failed')
            return fake_pipeline(image_path, output_dir, **kwargs)
        self.manager.pipeline = fail_one
        with self.assertLogs('local_app.server', level='ERROR'):
            response = self.post([('one.png', image_bytes()), ('two.png', image_bytes())])
            job = self.wait(response.get_json()['job_id'])
        self.assertEqual(job['status'], 'completed')
        self.assertEqual(job['failed_images'], 1)
        self.assertNotIn('/private/path', json.dumps(job))
        self.assertEqual(job['images'][1]['status'], 'completed')

    def test_invalid_and_corrupt_uploads_leave_no_task_files(self):
        cases = [('bad.exe', image_bytes()), ('../escape.png', image_bytes()),
                 ('C:\\escape.png', image_bytes()), ('bad.png', b'broken'),
                 ('wrong.jpg', image_bytes())]
        for name, content in cases:
            with self.subTest(name=name):
                response = self.post([(name, content)])
                self.assertEqual(response.status_code, 400)
                self.assertIsInstance(response.get_json()['error'], str)
                self.assertEqual(self.manager.jobs, {})
                self.assertEqual(list(self.manager.root.iterdir()), [])

    def test_file_size_total_pixels_and_batch_limits(self):
        self.manager.limits['max_files'] = 1
        self.assertEqual(self.post([('a.png', image_bytes()), ('b.png', image_bytes())]).status_code, 400)
        self.manager.limits['max_files'] = 50
        self.manager.limits['max_file_bytes'] = 10
        self.assertEqual(self.post([('a.png', image_bytes())]).status_code, 400)
        self.manager.limits['max_file_bytes'] = 1_000_000
        self.manager.limits['max_total_bytes'] = 10
        self.assertEqual(self.post([('a.png', image_bytes())]).status_code, 400)
        self.manager.limits['max_total_bytes'] = 1_000_000
        self.manager.limits['max_pixels'] = 100
        self.assertEqual(self.post([('a.png', image_bytes())]).status_code, 400)
        self.assertEqual(list(self.manager.root.iterdir()), [])

    def test_request_limit_returns_friendly_json(self):
        self.app.config['MAX_CONTENT_LENGTH'] = 10
        response = self.post([('a.png', image_bytes())])
        self.assertEqual(response.status_code, 413)
        self.assertIn('上传', response.get_json()['error'])

    def test_cross_origin_and_dns_rebinding_requests_are_rejected(self):
        for host in ['example.com', '127.0.0.1.evil.test', 'localhost.evil.test', 'localhost@evil.test']:
            with self.subTest(host=host):
                self.assertEqual(self.client.get('/health', headers={'Host': host}).status_code, 403)
        for origin in ['https://evil.test', 'null', 'http://localhost:9999']:
            self.assertEqual(self.client.post('/api/jobs', headers={'Origin': origin}).status_code, 403)
        self.assertEqual(self.client.post('/api/jobs').status_code, 403)
        self.assertEqual(self.client.get('/health', headers={'Sec-Fetch-Site': 'cross-site'}).status_code, 403)
        self.assertEqual(self.client.get('/health', headers={'Host': '127.0.0.1'}).status_code, 200)

    def test_only_job_allowlisted_files_and_dedicated_static_are_served(self):
        job_id = self.post([('one.png', image_bytes())]).get_json()['job_id']
        job = self.wait(job_id)
        directory = self.manager.jobs[job_id]['_directory']
        (directory / 'bundle' / 'private.txt').write_text('not a downloadable product')
        for suffix in ['../fixture.joblib', '%2e%2e/fixture.joblib', 'private.txt', 'images/../../private.txt']:
            self.assertEqual(self.client.get(f'/api/jobs/{job_id}/files/{suffix}').status_code, 404)
        for path in ['/api/jobs', '/models/particle_filter_v3/particle_filter.joblib', '/static/../fixture.joblib', '/api/jobs/not-a-uuid']:
            self.assertIn(self.client.get(path).status_code, [404, 405])
        self.assertIn(b'Fixture upload UI', self.client.get('/').data)
        self.assertEqual(self.client.get(job['review_url']).headers['X-Frame-Options'], 'DENY')

    def test_json_script_escaping_and_csv_names_are_safe_without_losing_original_name(self):
        names = ['=1+2.png', '<script>.png']
        job_id = self.post([(name, image_bytes()) for name in names]).get_json()['job_id']
        job = self.wait(job_id)
        csv_text = self.client.get(job['csv_url']).data.decode('utf-8-sig')
        self.assertIn("'=1+2.png", csv_text)
        data = self.client.get(job['json_url']).get_json()
        self.assertEqual([record['original_name'] for record in data['images']], names)
        self.assertNotIn('"original_name":"<script>.png"', self.client.get(job['review_url']).data.decode())

    def test_batch_identity_is_unique_and_review_uses_safe_display_names(self):
        template_path = Path(__file__).resolve().parent.parent / 'review_template.html'
        template_hash = hashlib.sha256(template_path.read_bytes()).hexdigest()
        names = ['中文原图.png', '=1+2.png', '  @sum.png']
        first_id = self.post([(name, image_bytes()) for name in names]).get_json()['job_id']
        first = self.wait(first_id)
        second_id = self.post([(names[0], image_bytes('green'))]).get_json()['job_id']
        second = self.wait(second_id)
        records = self.client.get(first['json_url']).get_json()['images']
        second_record = self.client.get(second['json_url']).get_json()['images'][0]
        self.assertNotEqual(records[0]['name'], second_record['name'])
        self.assertEqual(records[0]['original_name'], second_record['original_name'])
        for record in records + [second_record]:
            self.assertRegex(record['name'], r'^[0-9a-f]{12}_image_\d{3}\.png$')
            self.assertEqual(Path(record['name']).name, record['name'])
            self.assertNotIn(str(self.root), record['name'])
        review = self.client.get(first['review_url']).data.decode()
        self.assertIn('option.textContent=displayName(im);', review)
        self.assertIn('[displayName(im),...keys.map(k=>c[k])]', review)
        self.assertIn('rows.push([spreadsheetName(im),...keys.map((k,j)=>', review)
        self.assertIn('return delta?[[im.name,delta]]:[]', review)
        self.assertIn('stored.changes,im.name', review)
        self.assertIn('images:images.map((im,i)=>({...im,', review)
        helpers = '\n'.join(re.findall(r'^  const (?:displayName|spreadsheetName) = .*$', review, re.M))
        program = helpers + '\nconsole.log(JSON.stringify(' + json.dumps(records, ensure_ascii=False) + '.map(im=>({display:displayName(im),csv:spreadsheetName(im),name:im.name}))));'
        result = subprocess.run(['node', '-e', program], check=True, text=True, capture_output=True)
        displayed = json.loads(result.stdout)
        self.assertEqual([item['display'] for item in displayed], names)
        self.assertEqual([item['csv'] for item in displayed], [names[0], "'" + names[1], "'" + names[2]])
        self.assertEqual([item['name'] for item in displayed], [record['name'] for record in records])
        script = re.search(r'<script>(.*?)</script>', review, re.S).group(1)
        subprocess.run(['node', '--check'], input=script, check=True, text=True, capture_output=True)
        self.assertEqual(hashlib.sha256(template_path.read_bytes()).hexdigest(), template_hash)

    def test_health_config_and_missing_job_are_useful_without_model_inference(self):
        health = self.client.get('/health').get_json()
        self.assertEqual(health['status'], 'ok')
        self.assertEqual(health['model'], {'name': 'fixture', 'available': True})
        self.assertTrue(health['offline'])
        config = self.client.get('/api/config').get_json()
        self.assertIn('.tiff', config['supported_extensions'])
        self.assertEqual(config['model_name'], 'fixture')
        self.assertEqual(self.client.get('/api/jobs/' + 'a' * 32).status_code, 404)


class LauncherModelValidationTests(unittest.TestCase):
    def test_model_validation_precedes_opening_port_and_browser(self):
        from local_app import __main__ as launcher
        manager = SimpleNamespace(model_path=Path('/synthetic/model.joblib'), close=mock.Mock())
        app = SimpleNamespace(extensions={'particle_jobs': manager}, config={'MAX_CONTENT_LENGTH': 1000})
        server = SimpleNamespace(effective_port=43210, run=mock.Mock(side_effect=KeyboardInterrupt), close=mock.Mock())
        events = []
        with mock.patch.object(launcher, 'create_app', return_value=app), \
                mock.patch.object(launcher, 'prepare_model', side_effect=lambda path: events.append(('validated', path))), \
                mock.patch.object(launcher, 'create_server', side_effect=lambda *args, **kwargs: (events.append(('server', kwargs['host'])) or server)), \
                mock.patch('sys.argv', ['local_app', '--no-browser']), \
                mock.patch('sys.stdout', new_callable=io.StringIO):
            launcher.main()
        self.assertEqual(events, [('validated', manager.model_path), ('server', '127.0.0.1')])
        server.close.assert_called_once()
        manager.close.assert_called_once()

    def test_invalid_model_and_missing_dependency_fail_before_server_with_chinese_error(self):
        from local_app import __main__ as launcher
        for failure in [ValueError('feature hash mismatch'), ModuleNotFoundError('missing library')]:
            manager = SimpleNamespace(model_path=Path('/synthetic/model.joblib'), close=mock.Mock())
            app = SimpleNamespace(extensions={'particle_jobs': manager}, config={})
            with self.subTest(failure=failure), \
                    mock.patch.object(launcher, 'create_app', return_value=app), \
                    mock.patch.object(launcher, 'prepare_model', side_effect=failure), \
                    mock.patch.object(launcher, 'create_server') as server, \
                    mock.patch('sys.argv', ['local_app', '--no-browser']), \
                    mock.patch('sys.stderr', new_callable=io.StringIO) as stderr, \
                    self.assertRaises(SystemExit) as raised:
                launcher.main()
            self.assertEqual(raised.exception.code, 1)
            self.assertIn('无法启动：本地模型或运行依赖未通过检查', stderr.getvalue())
            server.assert_not_called()
            manager.close.assert_called_once()


if __name__ == '__main__':
    unittest.main()
