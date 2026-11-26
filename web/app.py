"""IPC-Dehaze web interface (Flask).

Run from the project root:   python web/app.py   ->   http://localhost:5000

Google sign-in is optional and configured with GOOGLE_CLIENT_ID / GOOGLE_CLIENT_SECRET
(see docs/GOOGLE_LOGIN.md). Without them the app runs open as one local user.
"""

import json
import logging
import os
import sys
import threading
from datetime import timedelta

from flask import Flask, jsonify, render_template, request, send_file, send_from_directory
from werkzeug.exceptions import HTTPException

WEB_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(WEB_DIR)
for path in (ROOT_DIR, WEB_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from web_auth import check_csrf, csrf_token, current_user, init_auth, login_required, settings as _settings  # noqa: E402
from web_jobs import JobError, JobManager  # noqa: E402
from web_settings import load_settings  # noqa: E402

log = logging.getLogger('ipc.web')


def create_app(settings=None, engine=None, warm_up=True):
    """Build the Flask app. ``engine`` is injectable so tests can run without the neural networks."""
    settings = settings or load_settings()
    app = Flask(__name__, template_folder=os.path.join(WEB_DIR, 'templates'),
                static_folder=os.path.join(WEB_DIR, 'static'))
    app.secret_key = settings.secret_key
    app.config.update(
        MAX_CONTENT_LENGTH=settings.max_upload_mb * 1024 * 1024,
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE='Lax',          # still sent on the redirect back from Google
        SESSION_COOKIE_SECURE=settings.secure_cookies,
        PERMANENT_SESSION_LIFETIME=timedelta(days=7),
        JSON_SORT_KEYS=False,
    )
    if settings.trust_proxy:
        from werkzeug.middleware.proxy_fix import ProxyFix
        app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)

    init_auth(app, settings)

    if engine is None:
        from ipc_dehaze.engine import Engine
        engine = Engine()
    jobs = JobManager(settings, engine)
    app.extensions['jobs'], app.extensions['engine'] = jobs, engine

    if warm_up:  # load the networks in the background so the first image is not slow
        def _warm():
            try:
                engine.warm_up()
                log.info('Engine ready: %s', engine.info())
            except Exception:
                log.exception('Engine failed to load')
        threading.Thread(target=_warm, daemon=True, name='ipc-warmup').start()

    # ------------------------------------------------------------------ security
    @app.before_request
    def csrf_protect():
        if request.method in ('POST', 'PUT', 'PATCH', 'DELETE') and not check_csrf():
            return jsonify({'error': 'Your session expired. Reload the page and try again.'}), 403

    @app.after_request
    def security_headers(response):
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
        response.headers['Content-Security-Policy'] = (
            "default-src 'self'; img-src 'self' data: blob: https://*.googleusercontent.com; "
            "style-src 'self' 'unsafe-inline'; script-src 'self'; frame-ancestors 'none'; form-action 'self'")
        if settings.secure_cookies:
            response.headers['Strict-Transport-Security'] = 'max-age=31536000'
        if request.path.startswith('/api/'):
            response.headers['Cache-Control'] = 'no-store'
        return response

    @app.errorhandler(HTTPException)
    def http_error(exc):
        if request.path.startswith('/api/') or request.method != 'GET':
            message = {413: f'The upload is too large (limit {settings.max_upload_mb} MB).',
                       404: 'Not found.', 405: 'Method not allowed.'}.get(exc.code, exc.description)
            return jsonify({'error': message}), exc.code
        return render_template('error.html', code=exc.code, message=exc.name), exc.code

    @app.errorhandler(JobError)
    def job_error(exc):
        return jsonify({'error': str(exc)}), exc.status

    @app.errorhandler(Exception)
    def unexpected(exc):
        log.exception('Unhandled error')
        return jsonify({'error': 'Something went wrong on the server. Please try again.'}), 500

    # ------------------------------------------------------------------ pages
    @app.route('/')
    @login_required
    def index():
        return render_template('index.html', user=current_user(), csrf=csrf_token(),
                               auth_enabled=settings.auth_enabled, max_images=settings.max_images,
                               max_upload_mb=settings.max_upload_mb, ttl_hours=settings.ttl_hours)

    @app.route('/health')
    def health():
        info = engine.info()
        status = 'ok' if info['ready'] else ('error' if info['error'] else 'loading')
        return jsonify({'status': status, 'engine': info, 'auth_enabled': settings.auth_enabled}), \
            (503 if status == 'error' else 200)

    # ------------------------------------------------------------------ API
    def _owned_job(job_id):
        job = jobs.get(current_user()['id'], job_id)
        if job is None:
            raise JobError('That job was not found (it may have expired).', 404)
        return job

    @app.route('/api/me')
    @login_required
    def me():
        user = current_user()
        return jsonify({'user': {k: user[k] for k in ('name', 'email', 'picture', 'guest')},
                        'csrf': csrf_token(), 'auth_enabled': settings.auth_enabled})

    @app.route('/api/jobs', methods=['GET'])
    @login_required
    def list_jobs():
        return jsonify({'jobs': [jobs.summary_view(j) for j in jobs.list_for(current_user()['id'])]})

    @app.route('/api/jobs', methods=['POST'])
    @login_required
    def create_job():
        uploads = [(f.filename, f.read()) for f in request.files.getlist('files') if f and f.filename]
        try:
            raw_params = json.loads(request.form.get('params') or '{}')
        except ValueError:
            raise JobError('The processing options were not valid.')
        job = jobs.create(current_user()['id'], uploads, raw_params)
        return jsonify(jobs.public_view(job)), 201

    @app.route('/api/jobs/<job_id>')
    @login_required
    def get_job(job_id):
        return jsonify(jobs.public_view(_owned_job(job_id)))

    @app.route('/api/jobs/<job_id>', methods=['DELETE'])
    @login_required
    def delete_job(job_id):
        jobs.delete(_owned_job(job_id))
        return jsonify({'ok': True})

    @app.route('/api/jobs/<job_id>/cancel', methods=['POST'])
    @login_required
    def cancel_job(job_id):
        job = _owned_job(job_id)
        jobs.cancel(job)
        return jsonify(jobs.public_view(job))

    @app.route('/api/jobs/<job_id>/files/<kind>/<name>')
    @login_required
    def job_file(job_id, kind, name):
        job = _owned_job(job_id)
        path = jobs.file_path(job, kind, name)
        if path is None:
            raise JobError('File not found.', 404)
        return send_from_directory(os.path.dirname(path), name, as_attachment=request.args.get('download') == '1',
                                   max_age=3600 if kind != 'output' else 0)

    @app.route('/api/jobs/<job_id>/zip')
    @login_required
    def job_zip(job_id):
        job = _owned_job(job_id)
        if job['status'] != 'done':
            raise JobError('Results are not ready yet.', 409)
        return send_file(jobs.build_zip(job), mimetype='application/zip', as_attachment=True,
                         download_name=f'ipc-dehaze-{job_id}.zip')

    return app


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    app = create_app()
    s = app.config['IPC_SETTINGS']
    where = f'http://{"localhost" if s.host in ("127.0.0.1", "0.0.0.0") else s.host}:{s.port}'
    print(f'\n  IPC-Dehaze is starting on {where}  '
          f'(Google sign-in: {"ON" if s.auth_enabled else "off - see docs/GOOGLE_LOGIN.md"})\n')
    try:
        from waitress import serve
        if not s.debug:
            serve(app, host=s.host, port=s.port, threads=8, max_request_body_size=s.max_upload_mb * 1024 * 1024)
            return
    except ImportError:
        pass
    app.run(host=s.host, port=s.port, debug=s.debug, threaded=True, use_reloader=False)


if __name__ == '__main__':
    main()
