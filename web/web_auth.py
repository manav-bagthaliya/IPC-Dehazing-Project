"""Google sign-in (OpenID Connect) plus the session / CSRF helpers the web app uses.

When GOOGLE_CLIENT_ID / GOOGLE_CLIENT_SECRET are not set the app runs open as a single local
user and says so in the UI. When they are set, every page and API call needs a signed-in user.
"""

import hashlib
import hmac
import logging
import os
import secrets
from functools import wraps
from urllib.parse import urlparse

from flask import Blueprint, current_app, jsonify, redirect, render_template, request, session, url_for

log = logging.getLogger('ipc.web.auth')
auth_bp = Blueprint('auth', __name__)

LOCAL_USER = {'id': 'local', 'name': 'Local user', 'email': '', 'picture': '', 'guest': True}
LOGIN_ERRORS = {
    'denied': 'Sign-in was cancelled.',
    'failed': 'Google sign-in failed. Please try again.',
    'not_allowed': 'This Google account is not allowed to use this app.',
    'unverified': 'Your Google email address is not verified.',
    'not_configured': 'Google sign-in is not configured yet.',
}


# ----------------------------------------------------------------------------- helpers
def settings():
    return current_app.config['IPC_SETTINGS']


def current_user():
    """The signed-in user dict, the shared local user when sign-in is off, or None."""
    if not settings().auth_enabled:
        return LOCAL_USER
    return session.get('user')


def owner_dir_name(user):
    """Filesystem-safe, stable folder name for a user (never contains user-controlled text)."""
    return hashlib.sha256(user['id'].encode('utf-8')).hexdigest()[:16]


def safe_next(target):
    """Only same-site relative paths may be used as a post-login redirect."""
    if not target or not target.startswith('/') or target.startswith('//') or '\\' in target:
        return '/'
    return target if not urlparse(target).netloc else '/'


def csrf_token():
    if 'csrf' not in session:
        session['csrf'] = secrets.token_urlsafe(32)
    return session['csrf']


def check_csrf():
    """Reject state-changing requests that lack the session's CSRF token."""
    sent = request.headers.get('X-CSRF-Token') or request.form.get('csrf_token') or ''
    expected = session.get('csrf', '')
    return bool(expected) and hmac.compare_digest(sent, expected)


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if current_user() is None:
            if request.path.startswith('/api/'):
                return jsonify({'error': 'Please sign in first.', 'login': url_for('auth.login')}), 401
            return redirect(url_for('auth.login', next=request.full_path.rstrip('?')))
        return view(*args, **kwargs)
    return wrapped


def _allowed(settings_, email):
    email = email.lower()
    if not settings_.allowed_emails and not settings_.allowed_domains:
        return True
    return email in settings_.allowed_emails or email.rsplit('@', 1)[-1] in settings_.allowed_domains


# ----------------------------------------------------------------------------- setup
def init_auth(app, settings_):
    app.config['IPC_SETTINGS'] = settings_
    app.register_blueprint(auth_bp)
    app.extensions['google'] = None
    if not settings_.auth_enabled:
        log.warning('Google sign-in is NOT configured: the app is open to anyone who can reach it. '
                    'See docs/GOOGLE_LOGIN.md.')
        return

    from authlib.integrations.flask_client import OAuth
    if settings_.public_base_url.startswith('http://localhost') or settings_.public_base_url == '':
        # Authlib refuses plain-http redirects unless told this is local development
        os.environ.setdefault('AUTHLIB_INSECURE_TRANSPORT', '1')
    oauth = OAuth(app)
    app.extensions['google'] = oauth.register(
        name='google',
        client_id=settings_.google_client_id,
        client_secret=settings_.google_client_secret,
        server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
        client_kwargs={'scope': 'openid email profile'},
    )


def _redirect_uri():
    base = settings().public_base_url
    return f'{base}/auth/google/callback' if base else url_for('auth.callback', _external=True)


# ----------------------------------------------------------------------------- routes
@auth_bp.route('/login')
def login():
    if current_user() is not None and settings().auth_enabled:
        return redirect(safe_next(request.args.get('next')))
    if not settings().auth_enabled:
        return redirect('/')
    return render_template('login.html', error=LOGIN_ERRORS.get(request.args.get('error', '')),
                           next=safe_next(request.args.get('next')), csrf=csrf_token(),
                           ttl_hours=settings().ttl_hours)


@auth_bp.route('/auth/google')
def google_start():
    google = current_app.extensions.get('google')
    if google is None:
        return redirect(url_for('auth.login', error='not_configured'))
    session['next'] = safe_next(request.args.get('next'))
    return google.authorize_redirect(_redirect_uri(), prompt='select_account')


@auth_bp.route('/auth/google/callback')
def callback():
    google = current_app.extensions.get('google')
    if google is None:
        return redirect(url_for('auth.login', error='not_configured'))
    if request.args.get('error'):
        return redirect(url_for('auth.login', error='denied'))
    try:
        token = google.authorize_access_token()
        info = token.get('userinfo') or google.userinfo()
    except Exception:
        log.exception('Google sign-in failed')
        return redirect(url_for('auth.login', error='failed'))

    email = (info.get('email') or '').strip()
    if not info.get('sub') or not email:
        return redirect(url_for('auth.login', error='failed'))
    if not info.get('email_verified', False):
        return redirect(url_for('auth.login', error='unverified'))
    if not _allowed(settings(), email):
        log.warning('Sign-in refused for %s (not in allow-list)', email)
        return redirect(url_for('auth.login', error='not_allowed'))

    target = safe_next(session.pop('next', '/'))
    session.clear()  # new session on login: prevents session fixation
    session['user'] = {
        'id': 'google:' + info['sub'],
        'name': info.get('name') or email.split('@')[0],
        'email': email,
        'picture': info.get('picture', ''),
        'guest': False,
    }
    session.permanent = True
    csrf_token()
    log.info('Signed in %s', email)
    return redirect(target)


@auth_bp.route('/logout', methods=['POST'])
def logout():
    session.clear()
    return jsonify({'ok': True, 'login': url_for('auth.login')})
