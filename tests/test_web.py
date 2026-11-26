import io
import json
import os
import sys
import time
import zipfile

import cv2
import numpy as np
import pytest

WEB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'web')
sys.path.insert(0, WEB_DIR)

from app import create_app  # noqa: E402
from web_auth import safe_next  # noqa: E402
from web_jobs import JobManager, parse_params  # noqa: E402
from web_settings import Settings  # noqa: E402

from ipc_dehaze.engine import DehazeResult  # noqa: E402


class FakeEngine:
    """Stands in for the neural networks so these tests run in milliseconds."""

    def __init__(self, fail=False):
        self.fail = fail

    def info(self):
        return {'ready': True, 'error': None, 'device': 'cpu', 'gpu_name': None, 'half_precision': False}

    def warm_up(self, *args, **kwargs):
        pass

    def process(self, image, iterations, max_size, resize_to, force_size, detect_plates=True, post_enhance=False):
        if self.fail:
            raise RuntimeError('boom')
        out = 255 - image
        plates = [{'text': 'MH01AB1234', 'formatted_text': 'MH 01 AB 1234', 'valid': True, 'confidence': 1.0,
                   'segments': {}, 'bbox': [2, 2, 30, 14], 'method': 'yolo', 'color': 'unknown'}] if detect_plates else []
        return DehazeResult(out, out, image, plates, 0.01)


def png_bytes(w=64, h=48):
    ok, buf = cv2.imencode('.png', np.random.randint(0, 255, (h, w, 3), np.uint8))
    assert ok
    return buf.tobytes()


def make_settings(tmp_path, **overrides):
    values = dict(secret_key='k' * 32, data_dir=str(tmp_path / 'data'))
    values.update(overrides)
    return Settings(**values)


@pytest.fixture
def open_client(tmp_path):
    app = create_app(make_settings(tmp_path), FakeEngine(), warm_up=False)
    app.config['TESTING'] = True
    return app.test_client()


def csrf(client):
    return client.get('/api/me').get_json()['csrf']


def submit(client, names=('a.png',), params=None, data=None):
    files = [(io.BytesIO(data or png_bytes()), n) for n in names]
    return client.post('/api/jobs', data={'files': files, 'params': json.dumps(params or {})},
                       headers={'X-CSRF-Token': csrf(client)}, content_type='multipart/form-data')


def wait_done(client, job_id, timeout=15):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f'/api/jobs/{job_id}').get_json()
        if job['status'] not in ('queued', 'running'):
            return job
        time.sleep(0.05)
    raise AssertionError('job did not finish')


# ----------------------------------------------------------------------------- open mode (no Google keys)
def test_pages_and_assets_in_open_mode(open_client):
    page = open_client.get('/')
    assert page.status_code == 200 and b'Clear the haze' in page.data and b'Google sign-in is off' in page.data
    for asset in ('/static/css/style.css', '/static/js/app.js', '/static/favicon.svg'):
        assert open_client.get(asset).status_code == 200
    assert open_client.get('/login').status_code == 302  # nothing to sign in to
    health = open_client.get('/health').get_json()
    assert health['status'] == 'ok' and health['auth_enabled'] is False
    assert open_client.get('/api/me').get_json()['user']['guest'] is True


def test_security_headers(open_client):
    resp = open_client.get('/')
    assert resp.headers['X-Frame-Options'] == 'DENY'
    assert "script-src 'self'" in resp.headers['Content-Security-Policy']
    assert open_client.get('/api/jobs').headers['Cache-Control'] == 'no-store'


def test_job_end_to_end(open_client):
    resp = submit(open_client, names=('one.png', 'two.png'))
    assert resp.status_code == 201
    job = wait_done(open_client, resp.get_json()['id'])
    assert job['status'] == 'done' and job['progress'] == {'done': 2, 'total': 2, 'percent': 100}
    assert job['summary'] == {'plates_found': 2, 'plates_valid': 2}

    item = job['items'][0]
    assert item['status'] == 'done' and item['plates'][0]['formatted_text'] == 'MH 01 AB 1234'
    for url in list(item['files'].values()) + [item['plates'][0]['crop'], job['downloads']['csv']]:
        assert open_client.get(url).status_code == 200, url

    zipped = zipfile.ZipFile(io.BytesIO(open_client.get(job['downloads']['zip']).data))
    names = zipped.namelist()
    assert 'plates.csv' in names and 'one_result.png' in names and 'two_dehazed.png' in names
    assert b'MH 01 AB 1234' in zipped.read('plates.csv')

    listed = open_client.get('/api/jobs').get_json()['jobs']
    assert listed[0]['id'] == job['id'] and listed[0]['images'] == 2 and listed[0]['plates'] == 2


def test_plate_detection_can_be_turned_off(open_client):
    job = wait_done(open_client, submit(open_client, params={'detect_plates': False}).get_json()['id'])
    assert job['summary']['plates_found'] == 0 and job['items'][0]['plates'] == []


def test_delete_and_cancel(open_client):
    job = wait_done(open_client, submit(open_client).get_json()['id'])
    token = {'X-CSRF-Token': csrf(open_client)}
    assert open_client.delete(f'/api/jobs/{job["id"]}', headers=token).status_code == 200
    assert open_client.get(f'/api/jobs/{job["id"]}').status_code == 404


def test_failing_engine_is_reported_not_crashed(tmp_path):
    app = create_app(make_settings(tmp_path), FakeEngine(fail=True), warm_up=False)
    client = app.test_client()
    job = wait_done(client, submit(client).get_json()['id'])
    assert job['status'] == 'error'
    assert job['items'][0]['status'] == 'error' and 'boom' in job['items'][0]['error']


# ----------------------------------------------------------------------------- validation and safety
def test_state_changing_requests_need_csrf(open_client):
    resp = open_client.post('/api/jobs', data={'files': (io.BytesIO(png_bytes()), 'a.png')},
                            content_type='multipart/form-data')
    assert resp.status_code == 403
    assert open_client.post('/api/jobs/x/cancel').status_code == 403


@pytest.mark.parametrize('name,data,message', [
    ('notes.txt', b'hello', 'not a supported image type'),
    ('fake.png', b'this is not a png', 'not a valid image'),
    ('empty.png', b'', 'empty'),
])
def test_upload_validation(open_client, name, data, message):
    resp = open_client.post('/api/jobs', data={'files': (io.BytesIO(data), name)},
                            headers={'X-CSRF-Token': csrf(open_client)}, content_type='multipart/form-data')
    assert resp.status_code == 400 and message in resp.get_json()['error']


def test_too_many_images_and_huge_images_rejected(tmp_path):
    app = create_app(make_settings(tmp_path, max_images=2, max_pixels=1000), FakeEngine(), warm_up=False)
    client = app.test_client()
    assert 'at most 2' in submit(client, names=('a.png', 'b.png', 'c.png'), data=png_bytes(8, 8)).get_json()['error']
    assert 'too large' in submit(client, data=png_bytes(64, 48)).get_json()['error']


def test_no_files_gives_a_clear_error(open_client):
    resp = open_client.post('/api/jobs', data={}, headers={'X-CSRF-Token': csrf(open_client)},
                            content_type='multipart/form-data')
    assert resp.status_code == 400 and 'at least one image' in resp.get_json()['error']


def test_file_route_blocks_traversal(open_client):
    job = wait_done(open_client, submit(open_client).get_json()['id'])
    for bad in ('../job.json', '..%2Fjob.json', '.hidden'):
        assert open_client.get(f'/api/jobs/{job["id"]}/files/output/{bad}').status_code == 404
    assert open_client.get(f'/api/jobs/{job["id"]}/files/secrets/x').status_code == 404
    assert open_client.get(f'/api/jobs/{job["id"]}/files/output/job.json').status_code == 404


def test_params_are_clamped():
    p = parse_params({'iterations': 999, 'max_size': 'abc', 'force_width': 640, 'resize_to': -5})
    assert (p.iterations, p.max_size, p.resize_to) == (20, 1500, 0)
    assert p.force_width == p.force_height == 0  # a forced size needs both sides
    assert parse_params('nonsense').iterations == 8
    both = parse_params({'force_width': 640, 'force_height': 480})
    assert (both.force_width, both.force_height) == (640, 480)


def test_safe_next_only_allows_local_paths():
    assert safe_next('/?job=abc') == '/?job=abc'
    for bad in (None, '', 'https://evil.example', '//evil.example', '/\\evil', 'javascript:alert(1)'):
        assert safe_next(bad) == '/'


# ----------------------------------------------------------------------------- Google sign-in (mocked)
class FakeGoogle:
    def __init__(self, info=None, fail=False):
        self.info, self.fail = info, fail

    def authorize_redirect(self, redirect_uri, **kwargs):
        from flask import redirect
        return redirect('https://accounts.google.test/auth?redirect_uri=' + redirect_uri)

    def authorize_access_token(self):
        if self.fail:
            raise RuntimeError('state mismatch')
        return {'userinfo': self.info}


def google_user(sub='123', email='ana@example.com', verified=True, name='Ana'):
    return {'sub': sub, 'email': email, 'email_verified': verified, 'name': name, 'picture': ''}


@pytest.fixture
def auth_app(tmp_path):
    def build(**overrides):
        settings = make_settings(tmp_path, google_client_id='id.apps.googleusercontent.com',
                                 google_client_secret='secret', **overrides)
        app = create_app(settings, FakeEngine(), warm_up=False)
        app.config['TESTING'] = True
        return app
    return build


def sign_in(app, info, client=None):
    app.extensions['google'] = FakeGoogle(info)
    client = client or app.test_client()
    return client, client.get('/auth/google/callback?code=x&state=y')


def test_signed_out_visitors_are_sent_to_login(auth_app):
    client = auth_app().test_client()
    resp = client.get('/')
    assert resp.status_code == 302 and '/login' in resp.headers['Location']
    api = client.get('/api/jobs')
    assert api.status_code == 401 and 'sign in' in api.get_json()['error'].lower()
    assert client.get('/health').status_code == 200   # health stays public
    login = client.get('/login')
    assert login.status_code == 200 and b'Sign in with Google' in login.data
    assert client.post('/api/jobs', data={}).status_code == 403  # CSRF fails before anything else


def test_google_button_starts_oauth(auth_app):
    app = auth_app()
    app.extensions['google'] = FakeGoogle()
    resp = app.test_client().get('/auth/google?next=/%3Fjob%3Dabc')
    assert resp.status_code == 302 and resp.headers['Location'].startswith('https://accounts.google.test/auth')
    assert 'callback' in resp.headers['Location']


def test_successful_sign_in_creates_a_session(auth_app):
    app = auth_app()
    client, resp = sign_in(app, google_user())
    assert resp.status_code == 302 and resp.headers['Location'].endswith('/')
    me = client.get('/api/me').get_json()
    assert me['user']['email'] == 'ana@example.com' and me['user']['guest'] is False
    page = client.get('/')
    assert page.status_code == 200 and b'ana@example.com' in page.data and b'Sign out' in page.data
    assert client.post('/logout', headers={'X-CSRF-Token': me['csrf']}).status_code == 200
    assert client.get('/api/me').status_code == 401


@pytest.mark.parametrize('info,error', [
    (google_user(verified=False), 'unverified'),
    ({'sub': '', 'email': ''}, 'failed'),
])
def test_bad_google_accounts_are_refused(auth_app, info, error):
    client, resp = sign_in(auth_app(), info)
    assert resp.status_code == 302 and f'error={error}' in resp.headers['Location']
    assert client.get('/api/me').status_code == 401


def test_cancelled_or_failed_google_flow(auth_app):
    app = auth_app()
    client = app.test_client()
    assert 'error=denied' in client.get('/auth/google/callback?error=access_denied').headers['Location']
    app.extensions['google'] = FakeGoogle(fail=True)
    assert 'error=failed' in client.get('/auth/google/callback?code=x').headers['Location']
    assert b'sign-in failed' in client.get('/login?error=failed').data


def test_email_allow_list(auth_app):
    app = auth_app(allowed_domains={'example.com'}, allowed_emails={'friend@gmail.com'})
    assert sign_in(app, google_user(email='ana@example.com'))[1].headers['Location'].endswith('/')
    assert sign_in(app, google_user(sub='2', email='friend@gmail.com'))[1].headers['Location'].endswith('/')
    _, refused = sign_in(app, google_user(sub='3', email='eve@evil.com'))
    assert 'error=not_allowed' in refused.headers['Location']


def test_users_cannot_see_each_others_jobs(auth_app):
    app = auth_app()
    ana, _ = sign_in(app, google_user(sub='A', email='ana@example.com'))
    job_id = wait_done(ana, submit(ana).get_json()['id'])['id']

    bob, _ = sign_in(app, google_user(sub='B', email='bob@example.com'))
    assert bob.get(f'/api/jobs/{job_id}').status_code == 404
    assert bob.get('/api/jobs').get_json()['jobs'] == []
    assert bob.get(f'/api/jobs/{job_id}/zip').status_code == 404
    assert ana.get(f'/api/jobs/{job_id}').status_code == 200


def test_login_redirect_cannot_leave_the_site(auth_app):
    app = auth_app()
    client = app.test_client()
    client.get('/auth/google?next=https://evil.example')
    app.extensions['google'] = FakeGoogle(google_user())
    assert client.get('/auth/google/callback?code=x').headers['Location'].endswith('/')


# ----------------------------------------------------------------------------- job manager
def test_expired_jobs_are_purged_and_history_survives_restart(tmp_path):
    settings = make_settings(tmp_path, ttl_hours=1)
    manager = JobManager(settings, FakeEngine(), cleanup_interval=0)
    job = manager.create('local', [('a.png', png_bytes())], {})
    deadline = time.time() + 10
    while manager.get('local', job['id'])['status'] != 'done' and time.time() < deadline:
        time.sleep(0.05)

    reloaded = JobManager(settings, FakeEngine(), cleanup_interval=0)   # "server restart"
    assert reloaded.get('local', job['id'])['status'] == 'done'

    assert manager.purge_expired(now=time.time() + 30 * 60) == 0
    assert manager.purge_expired(now=time.time() + 2 * 3600) == 1
    assert manager.get('local', job['id']) is None and not os.path.exists(manager.job_dir(job))


def test_interrupted_jobs_are_marked_after_restart(tmp_path):
    settings = make_settings(tmp_path)
    manager = JobManager(settings, FakeEngine(), cleanup_interval=0)
    job = manager.create('local', [('a.png', png_bytes())], {})
    deadline = time.time() + 10
    while manager.get('local', job['id'])['status'] != 'done' and time.time() < deadline:
        time.sleep(0.05)
    path = os.path.join(manager.job_dir(job), 'job.json')
    stored = json.load(open(path))
    stored['status'], stored['items'][0]['status'] = 'running', 'processing'
    json.dump(stored, open(path, 'w'))

    reloaded = JobManager(settings, FakeEngine(), cleanup_interval=0)
    got = reloaded.get('local', job['id'])
    assert got['status'] == 'error' and 'restarted' in got['error']
    assert got['items'][0]['status'] == 'error'
