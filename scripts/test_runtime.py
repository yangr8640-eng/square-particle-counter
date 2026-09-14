"""Standard-library checks for release verification; no models or user images."""
import hashlib
import json
from pathlib import Path
import stat
import struct
import tempfile
import unittest
import zipfile
import zlib

from scripts import verify_windows_release as verify


class ReleaseVerifierTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='release-check-fixture-')
        self.root = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)

    def fixture_archive(self):
        files = {'runtime/python.exe': b'fake executable; never run', 'program/start.py': b'# fixture',
                 'program/local_app/model/particle_filter.joblib': b'fake model; never loaded',
                 'program/local_app/model/particle_filter.json': b'{}',
                 'program/local_app/static/index.html': b'<html>fixture</html>'}
        hashes = {name: hashlib.sha256(content).hexdigest() for name, content in files.items()}
        manifest = dict(target='windows-x64', model_version='square-core-filter-4.0', contains_runtime=True,
                        contains_training_images=False, files=hashes,
                        model_sha256=hashes['program/local_app/model/particle_filter.joblib'])
        archive = self.root / 'fixture.zip'
        with zipfile.ZipFile(archive, 'w') as zipped:
            for name, content in files.items():
                zipped.writestr(verify.PACKAGE_NAME + '/' + name, content)
            zipped.writestr(verify.PACKAGE_NAME + '/manifest.json', json.dumps(manifest))
        return archive

    def test_release_source_is_fixed_and_arguments_cannot_be_code_or_paths(self):
        url = verify.validate_source('example/square-particle-counter', 'v1.0.0')
        self.assertEqual(url, 'https://github.com/example/square-particle-counter/releases/download/v1.0.0/' + verify.ASSET)
        for repository, tag in [('example/../other', 'v1.0.0'), ('example/repo', '../v1'),
                                ('example/repo', '$(anything)'), ('example/repo', 'v1.0.0; exit'),
                                ('https://other/repo', 'v1.0.0')]:
            with self.subTest(repository=repository, tag=tag), self.assertRaises(ValueError):
                verify.validate_source(repository, tag)

    def test_archive_checksum_requires_one_exact_asset_entry(self):
        sums = self.root / 'SHA256SUMS.txt'
        expected = 'a' * 64
        line = expected + '  ' + verify.ASSET + '\n'
        sums.write_text(line + 'b' * 64 + '  other.zip\n')
        self.assertEqual(verify.expected_checksum(sums), expected)
        for content in [line + line, 'b' * 64 + '  other.zip\n']:
            sums.write_text(content)
            with self.assertRaises(ValueError):
                verify.expected_checksum(sums)

    def test_safe_extraction_checks_all_manifest_hashes_and_extra_files(self):
        root = verify.extract_safely(self.fixture_archive(), self.root / '解压 验证')
        self.assertEqual(len(verify.verify_manifest(root)['files']), 5)
        (root / 'unexpected.txt').write_text('not declared')
        with self.assertRaisesRegex(ValueError, 'unlisted'):
            verify.verify_manifest(root)
        (root / 'unexpected.txt').unlink()
        (root / 'program/start.py').write_text('modified')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            verify.verify_manifest(root)

    def test_zip_paths_symlinks_and_windows_aliases_are_rejected_before_extraction(self):
        for index, name in enumerate(['../escape', '/absolute', 'C:/escape', 'root/CON.txt',
                                      'root/a.', 'root/dir\\file', 'root/./file']):
            with self.subTest(name=name):
                archive = self.root / f'bad-{index}.zip'
                with zipfile.ZipFile(archive, 'w') as zipped:
                    zipped.writestr('good/first', 'must not be extracted yet')
                    zipped.writestr(name, 'bad')
                destination = self.root / f'extracted-{index}'
                with self.assertRaises(ValueError):
                    verify.extract_safely(archive, destination)
                self.assertEqual(list(destination.iterdir()), [])
        archive = self.root / 'link.zip'
        info = zipfile.ZipInfo('root/link')
        info.create_system = 3
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        with zipfile.ZipFile(archive, 'w') as zipped:
            zipped.writestr(info, '../outside')
        with self.assertRaisesRegex(ValueError, 'symlink'):
            verify.extract_safely(archive, self.root / 'link-out')

    def test_case_colliding_windows_paths_are_rejected(self):
        archive = self.root / 'case.zip'
        with zipfile.ZipFile(archive, 'w') as zipped:
            zipped.writestr('root/Image.png', 'a')
            zipped.writestr('root/image.png', 'b')
        with self.assertRaisesRegex(ValueError, 'case-colliding'):
            verify.extract_safely(archive, self.root / 'case-out')

    def test_synthetic_png_is_complete_rgb_with_two_generated_cores(self):
        png = verify.synthetic_png()
        self.assertEqual(verify.png_size(png), (160, 160))
        offset, compressed = 8, bytearray()
        while offset < len(png):
            length = struct.unpack('>I', png[offset:offset + 4])[0]
            kind = png[offset + 4:offset + 8]
            payload = png[offset + 8:offset + 8 + length]
            crc = struct.unpack('>I', png[offset + 8 + length:offset + 12 + length])[0]
            self.assertEqual(crc, zlib.crc32(kind + payload) & 0xffffffff)
            if kind == b'IDAT':
                compressed.extend(payload)
            offset += length + 12
        self.assertEqual(offset, len(png))
        pixels = zlib.decompress(compressed)
        self.assertEqual(len(pixels), 160 * (1 + 160 * 3))
        for x, y, expected in [(45, 80, (195, 220, 160)), (105, 80, (218, 225, 205)), (0, 0, (245, 245, 245))]:
            index = y * 481 + 1 + x * 3
            self.assertEqual(tuple(pixels[index:index + 3]), expected)

    def test_local_result_links_cannot_escape_loopback_origin(self):
        self.assertEqual(verify.local_link('http://127.0.0.1:1234/', '/api/jobs/id'), 'http://127.0.0.1:1234/api/jobs/id')
        for link in ['https://elsewhere.test/a', '//elsewhere.test/a', 'relative']:
            with self.assertRaises(ValueError):
                verify.local_link('http://127.0.0.1:1234/', link)


if __name__ == '__main__':
    unittest.main()
