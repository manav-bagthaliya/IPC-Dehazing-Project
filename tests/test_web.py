import io
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'web'))


@pytest.fixture
def client():
    from app import app
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


def test_index_and_static_assets(client):
    page = client.get('/')
    assert page.status_code == 200
    assert b'css/style.css' in page.data and b'js/app.js' in page.data
    assert client.get('/static/css/style.css').status_code == 200
    assert client.get('/static/js/app.js').status_code == 200


def test_status_without_session(client):
    assert client.get('/status').get_json() == {'session_id': None, 'has_files': False}


def test_upload_rejects_bad_extension(client):
    resp = client.post('/upload', data={'files': (io.BytesIO(b'x'), 'notes.txt')},
                       content_type='multipart/form-data')
    assert resp.status_code == 400


def test_process_requires_session(client):
    assert client.post('/process', json={}).status_code == 400
