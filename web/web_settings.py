"""Web app settings, read from environment variables (and a ``.env`` file in the project root).

See docs/GOOGLE_LOGIN.md for the Google sign-in variables.
"""

import os
import secrets
from dataclasses import dataclass, field

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _csv_set(value):
    return {item.strip().lower() for item in (value or '').split(',') if item.strip()}


def _load_or_create_secret_key(instance_dir):
    """A stable random key, generated once and kept in instance/secret_key (git-ignored)."""
    path = os.path.join(instance_dir, 'secret_key')
    try:
        with open(path, encoding='utf-8') as f:
            key = f.read().strip()
        if len(key) >= 32:
            return key
    except OSError:
        pass
    key = secrets.token_hex(32)
    os.makedirs(instance_dir, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(key)
    return key


@dataclass
class Settings:
    secret_key: str = ''
    google_client_id: str = ''
    google_client_secret: str = ''
    allowed_emails: set = field(default_factory=set)      # empty = any Google account may sign in
    allowed_domains: set = field(default_factory=set)
    public_base_url: str = ''                             # e.g. https://dehaze.example.com (behind a proxy)
    host: str = '127.0.0.1'                               # localhost only unless you opt in
    port: int = 5000
    data_dir: str = os.path.join(ROOT_DIR, 'web_data')
    ttl_hours: float = 24.0                               # how long uploads/results are kept
    max_images: int = 10
    max_upload_mb: int = 100
    max_pixels: int = 50_000_000                          # per image, guards against decompression bombs
    max_active_jobs_per_user: int = 3
    trust_proxy: bool = False
    debug: bool = False

    @property
    def auth_enabled(self):
        return bool(self.google_client_id and self.google_client_secret)

    @property
    def secure_cookies(self):
        return self.public_base_url.lower().startswith('https://')


def load_settings(environ=None):
    """Build Settings from ``environ`` (default: os.environ after loading .env)."""
    if environ is None:
        try:
            from dotenv import load_dotenv
            load_dotenv(os.path.join(ROOT_DIR, '.env'))
        except ImportError:
            pass
        environ = os.environ

    instance_dir = os.path.join(ROOT_DIR, 'instance')
    s = Settings(
        secret_key=environ.get('SECRET_KEY') or _load_or_create_secret_key(instance_dir),
        google_client_id=environ.get('GOOGLE_CLIENT_ID', '').strip(),
        google_client_secret=environ.get('GOOGLE_CLIENT_SECRET', '').strip(),
        allowed_emails=_csv_set(environ.get('ALLOWED_EMAILS')),
        allowed_domains=_csv_set(environ.get('ALLOWED_EMAIL_DOMAINS')),
        public_base_url=environ.get('PUBLIC_BASE_URL', '').strip().rstrip('/'),
        host=environ.get('HOST', '127.0.0.1'),
        port=int(environ.get('PORT', 5000)),
        data_dir=environ.get('DATA_DIR') or os.path.join(ROOT_DIR, 'web_data'),
        ttl_hours=float(environ.get('DATA_TTL_HOURS', 24)),
        max_images=int(environ.get('MAX_IMAGES', 10)),
        max_upload_mb=int(environ.get('MAX_UPLOAD_MB', 100)),
        trust_proxy=environ.get('TRUST_PROXY') == '1',
        debug=environ.get('FLASK_DEBUG') == '1',
    )
    return s
