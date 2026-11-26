"""Background jobs for the web app: validated uploads, one shared engine, progress, history, cleanup.

Layout on disk:  <data_dir>/<owner hash>/<job id>/{input, output, previews, job.json}
"""

import csv
import hashlib
import io
import json
import logging
import os
import shutil
import threading
import time
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass

import cv2
import numpy as np
from PIL import Image
from werkzeug.utils import secure_filename

log = logging.getLogger('ipc.web.jobs')

ALLOWED_EXT = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}
PNG_FAST = [cv2.IMWRITE_PNG_COMPRESSION, 3]
PREVIEW_EDGE = 1400


class JobError(Exception):
    """An error that is safe to show the user."""

    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


@dataclass
class Params:
    iterations: int = 8
    max_size: int = 1500
    resize_to: int = 0
    force_width: int = 0
    force_height: int = 0
    detect_plates: bool = True
    post_enhance: bool = False


def _clamp_int(value, default, lo, hi):
    try:
        return max(lo, min(hi, int(float(value))))
    except (TypeError, ValueError):
        return default


def parse_params(raw):
    """Validate user-supplied options; anything invalid falls back to a safe default."""
    raw = raw if isinstance(raw, dict) else {}
    p = Params(
        iterations=_clamp_int(raw.get('iterations'), 8, 1, 20),
        max_size=_clamp_int(raw.get('max_size'), 1500, 512, 3000),
        resize_to=_clamp_int(raw.get('resize_to'), 0, 0, 3000),
        force_width=_clamp_int(raw.get('force_width'), 0, 0, 4000),
        force_height=_clamp_int(raw.get('force_height'), 0, 0, 4000),
        detect_plates=bool(raw.get('detect_plates', True)),
        post_enhance=bool(raw.get('post_enhance', False)),
    )
    if not (p.force_width and p.force_height):  # a forced size needs both sides
        p.force_width = p.force_height = 0
    return p


def friendly_error(exc):
    try:
        import torch
        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return 'The GPU ran out of memory. Try the Fast preset or a smaller image.'
    except ImportError:
        pass
    return f'Processing failed: {str(exc)[:200] or type(exc).__name__}'


def _safe_stem(filename, used):
    stem = secure_filename(os.path.splitext(filename)[0])[:60] or 'image'
    candidate, n = stem, 1
    while candidate in used:
        n += 1
        candidate = f'{stem}_{n}'
    used.add(candidate)
    return candidate


def _save_preview(image, path):
    h, w = image.shape[:2]
    scale = PREVIEW_EDGE / max(h, w)
    if scale < 1:
        image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(path, image, [cv2.IMWRITE_JPEG_QUALITY, 88])


def validate_upload(filename, data, max_pixels):
    """Return the file extension if ``data`` really is a reasonably sized image, else raise JobError."""
    ext = os.path.splitext(filename or '')[1].lower()
    if ext not in ALLOWED_EXT:
        raise JobError(f'"{filename}" is not a supported image type (use PNG, JPG, BMP, TIFF or WebP).')
    if not data:
        raise JobError(f'"{filename}" is empty.')
    try:
        with Image.open(io.BytesIO(data)) as im:   # reads only the header: cheap and bomb-safe
            width, height = im.size
    except Exception:
        raise JobError(f'"{filename}" is not a valid image.')
    if width * height > max_pixels:
        raise JobError(f'"{filename}" is too large ({width}x{height}); the limit is {max_pixels // 1_000_000} megapixels.')
    if min(width, height) < 16:
        raise JobError(f'"{filename}" is too small to process.')
    return ext


class JobManager:
    def __init__(self, settings, engine, cleanup_interval=900):
        self.settings = settings
        self.engine = engine
        self.root = settings.data_dir
        os.makedirs(self.root, exist_ok=True)
        self._jobs = {}
        self._lock = threading.RLock()
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='ipc-worker')  # one GPU, one job
        self._load_existing()
        if cleanup_interval:
            threading.Thread(target=self._cleanup_loop, args=(cleanup_interval,), daemon=True,
                             name='ipc-cleanup').start()

    # ------------------------------------------------------------------ paths / persistence
    @staticmethod
    def owner_key(owner_id):
        return hashlib.sha256(owner_id.encode('utf-8')).hexdigest()[:16]

    def job_dir(self, job):
        return os.path.join(self.root, self.owner_key(job['owner']), job['id'])

    def _save(self, job):
        path = os.path.join(self.job_dir(job), 'job.json')
        tmp = path + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(job, f)
        os.replace(tmp, path)

    def _load_existing(self):
        """Reload history after a restart; jobs that were mid-flight are marked interrupted."""
        for owner_dir in glob_dirs(self.root):
            for job_dir in glob_dirs(owner_dir):
                path = os.path.join(job_dir, 'job.json')
                try:
                    with open(path, encoding='utf-8') as f:
                        job = json.load(f)
                except (OSError, ValueError):
                    continue
                if job.get('status') in ('queued', 'running'):
                    job['status'] = 'error'
                    job['error'] = 'The server restarted while this job was running.'
                    for item in job['items']:
                        if item['status'] in ('queued', 'processing'):
                            item['status'] = 'error'
                            item['error'] = 'Interrupted by a server restart.'
                    self._jobs[job['id']] = job
                    self._save(job)
                else:
                    self._jobs[job['id']] = job

    # ------------------------------------------------------------------ queries
    def get(self, owner_id, job_id):
        with self._lock:
            job = self._jobs.get(job_id)
            return job if job and job['owner'] == owner_id else None

    def list_for(self, owner_id, limit=30):
        with self._lock:
            jobs = [j for j in self._jobs.values() if j['owner'] == owner_id]
        return sorted(jobs, key=lambda j: j['created'], reverse=True)[:limit]

    def active_count(self, owner_id):
        with self._lock:
            return sum(1 for j in self._jobs.values()
                       if j['owner'] == owner_id and j['status'] in ('queued', 'running'))

    # ------------------------------------------------------------------ create / control
    def create(self, owner_id, uploads, raw_params):
        """Validate and store ``uploads`` [(filename, bytes)], then queue the job."""
        s = self.settings
        if not uploads:
            raise JobError('Choose at least one image.')
        if len(uploads) > s.max_images:
            raise JobError(f'You can process at most {s.max_images} images at once.')
        if self.active_count(owner_id) >= s.max_active_jobs_per_user:
            raise JobError('You already have several jobs waiting. Wait for one to finish or cancel it.', 429)

        validated = [(name, data, validate_upload(name, data, s.max_pixels)) for name, data in uploads]
        params = parse_params(raw_params)

        job_id = uuid.uuid4().hex[:16]
        job = {
            'id': job_id, 'owner': owner_id, 'status': 'queued', 'created': time.time(),
            'started': None, 'finished': None, 'error': None, 'cancel': False,
            'params': asdict(params), 'items': [],
        }
        jdir = self.job_dir(job)
        for sub in ('input', 'output', 'previews'):
            os.makedirs(os.path.join(jdir, sub), exist_ok=True)

        used = set()
        for name, data, ext in validated:
            stem = _safe_stem(name, used)
            with open(os.path.join(jdir, 'input', stem + ext), 'wb') as f:
                f.write(data)
            job['items'].append({
                'id': stem, 'name': secure_filename(name) or stem + ext, 'input': stem + ext,
                'status': 'queued', 'error': None, 'seconds': None, 'width': None, 'height': None,
                'plates': [], 'fallback': False, 'files': {},
            })

        with self._lock:
            self._jobs[job_id] = job
            self._save(job)
        self._pool.submit(self._run, job_id)
        log.info('Job %s queued: %d image(s)', job_id, len(validated))
        return job

    def cancel(self, job):
        with self._lock:
            if job['status'] in ('queued', 'running'):
                job['cancel'] = True
                self._save(job)

    def delete(self, job):
        with self._lock:
            if job['status'] in ('queued', 'running'):
                raise JobError('Cancel the job before deleting it.', 409)
            self._jobs.pop(job['id'], None)
        shutil.rmtree(self.job_dir(job), ignore_errors=True)

    # ------------------------------------------------------------------ worker
    def _run(self, job_id):
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            if job['cancel']:
                self._finish(job, 'cancelled')
                return
            job['status'], job['started'] = 'running', time.time()
            self._save(job)

        for item in job['items']:
            if job['cancel']:
                item['status'] = 'cancelled'
                continue
            with self._lock:
                item['status'] = 'processing'
                self._save(job)
            try:
                self._process_item(job, item)
                item['status'] = 'done'
            except Exception as exc:
                log.exception('Job %s: %s failed', job_id, item['name'])
                item['status'], item['error'] = 'error', friendly_error(exc)
            with self._lock:
                self._save(job)

        statuses = {i['status'] for i in job['items']}
        if job['cancel']:
            self._finish(job, 'cancelled')
        elif statuses <= {'error'}:
            self._finish(job, 'error', 'None of the images could be processed.')
        else:
            self._write_plates_csv(job)
            self._finish(job, 'done')

    def _finish(self, job, status, error=None):
        with self._lock:
            job['status'], job['finished'], job['error'] = status, time.time(), error
            self._save(job)
        log.info('Job %s finished: %s', job['id'], status)

    def _process_item(self, job, item):
        from ipc_dehaze.enhance import enhance_plate_region

        jdir, p = self.job_dir(job), job['params']
        image = cv2.imread(os.path.join(jdir, 'input', item['input']), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError('The image could not be read.')
        item['height'], item['width'] = image.shape[:2]
        force = (p['force_height'], p['force_width']) if p['force_width'] and p['force_height'] else None

        result = self.engine.process(image, p['iterations'], p['max_size'], p['resize_to'], force,
                                     detect_plates=p['detect_plates'], post_enhance=p['post_enhance'])

        stem, out, prev = item['id'], os.path.join(jdir, 'output'), os.path.join(jdir, 'previews')
        cv2.imwrite(os.path.join(out, f'{stem}_dehazed.png'), result.dehazed, PNG_FAST)
        cv2.imwrite(os.path.join(out, f'{stem}_result.png'), result.final, PNG_FAST)
        _save_preview(image, os.path.join(prev, f'{stem}_original.jpg'))
        _save_preview(result.dehazed, os.path.join(prev, f'{stem}_dehazed.jpg'))
        _save_preview(result.final, os.path.join(prev, f'{stem}_result.jpg'))

        plates = []
        for k, plate in enumerate(result.plates, 1):
            crop = enhance_plate_region(result.dehazed, plate['bbox'])
            if crop is not None and crop.size:
                cv2.imwrite(os.path.join(out, f'{stem}_plate_{k}.png'), crop, PNG_FAST)
                plate['crop'] = f'{stem}_plate_{k}.png'
            plates.append(plate)

        item.update({
            'seconds': round(result.seconds, 2), 'plates': plates, 'fallback': result.used_fallback,
            'files': {
                'original': f'previews/{stem}_original.jpg', 'dehazed': f'previews/{stem}_dehazed.jpg',
                'result': f'previews/{stem}_result.jpg',
                'dehazed_png': f'output/{stem}_dehazed.png', 'result_png': f'output/{stem}_result.png',
            },
        })

    def _write_plates_csv(self, job):
        rows = []
        for item in job['items']:
            for plate in item['plates']:
                rows.append([item['name'], plate['formatted_text'] or plate['text'], 'yes' if plate['valid'] else 'no',
                             f"{plate['confidence']:.3f}", plate['color'], plate['method'], *plate['bbox']])
        path = os.path.join(self.job_dir(job), 'output', 'plates.csv')
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['image', 'plate', 'valid_indian_format', 'confidence', 'color', 'method',
                             'x1', 'y1', 'x2', 'y2'])
            writer.writerows(rows)

    # ------------------------------------------------------------------ downloads
    def build_zip(self, job):
        """In-memory ZIP of every result file (PNG/JPEG are already compressed, so just store them)."""
        buffer = io.BytesIO()
        out = os.path.join(self.job_dir(job), 'output')
        with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_STORED) as z:
            for name in sorted(os.listdir(out)):
                z.write(os.path.join(out, name), name)
        buffer.seek(0)
        return buffer

    def file_path(self, job, kind, name):
        """Absolute path of a job file, or None. Only the known sub-folders, never a traversal."""
        if kind not in ('output', 'previews', 'input') or name != os.path.basename(name) or name.startswith('.'):
            return None
        path = os.path.join(self.job_dir(job), kind, name)
        return path if os.path.isfile(path) else None

    # ------------------------------------------------------------------ cleanup
    def purge_expired(self, now=None):
        """Delete finished jobs older than the retention time. Returns how many were removed."""
        now = now or time.time()
        cutoff = now - self.settings.ttl_hours * 3600
        with self._lock:
            old = [j for j in self._jobs.values()
                   if j['status'] not in ('queued', 'running') and (j['finished'] or j['created']) < cutoff]
            for job in old:
                self._jobs.pop(job['id'], None)
        for job in old:
            shutil.rmtree(self.job_dir(job), ignore_errors=True)
        return len(old)

    def _cleanup_loop(self, interval):
        while True:
            time.sleep(interval)
            try:
                removed = self.purge_expired()
                if removed:
                    log.info('Removed %d expired job(s)', removed)
            except Exception:
                log.exception('Cleanup failed')

    # ------------------------------------------------------------------ presentation
    def public_view(self, job, url_prefix='/api/jobs'):
        """JSON for the browser: same data, with ready-to-use file URLs and a progress summary."""
        def url(rel):
            kind, name = rel.split('/', 1)
            return f'{url_prefix}/{job["id"]}/files/{kind}/{name}'

        items = []
        for item in job['items']:
            view = {k: item[k] for k in ('id', 'name', 'status', 'error', 'seconds', 'width', 'height', 'fallback')}
            view['files'] = {k: url(v) for k, v in item['files'].items()}
            view['plates'] = [{**{k: v for k, v in pl.items() if k != 'crop'},
                               'crop': url('output/' + pl['crop']) if pl.get('crop') else None}
                              for pl in item['plates']]
            items.append(view)

        total = len(job['items'])
        done = sum(1 for i in job['items'] if i['status'] in ('done', 'error', 'cancelled'))
        plates = [pl for i in job['items'] for pl in i['plates']]
        return {
            'id': job['id'], 'status': job['status'], 'error': job['error'], 'created': job['created'],
            'started': job['started'], 'finished': job['finished'], 'params': job['params'], 'items': items,
            'progress': {'done': done, 'total': total, 'percent': round(100 * done / total) if total else 0},
            'summary': {'plates_found': len(plates), 'plates_valid': sum(1 for pl in plates if pl['valid'])},
            'downloads': {'zip': f'{url_prefix}/{job["id"]}/zip', 'csv': url('output/plates.csv')}
            if job['status'] == 'done' else None,
        }

    @staticmethod
    def summary_view(job):
        return {'id': job['id'], 'status': job['status'], 'created': job['created'],
                'images': len(job['items']),
                'plates': sum(len(i['plates']) for i in job['items']),
                'thumb': next((f'/api/jobs/{job["id"]}/files/{i["files"]["original"]}'
                               for i in job['items'] if i['files']), None)}


def glob_dirs(path):
    try:
        return [os.path.join(path, d) for d in sorted(os.listdir(path)) if os.path.isdir(os.path.join(path, d))]
    except OSError:
        return []
