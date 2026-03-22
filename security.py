"""
BOOKIFY — Security Middleware Module
=====================================
Provides enterprise-grade security features:
  • Security headers (OWASP recommended)
  • CSRF token generation & validation
  • Rate limiting (sliding-window, per-IP)
  • Input sanitization (anti-XSS, anti-SQLi)
  • HTTPS enforcement
  • IP blocking (auto-ban after repeated abuse)
  • File upload validation
  • Suspicious request detection

All features are implemented with ZERO external dependencies.
"""

import os
import re
import time
import hmac
import hashlib
import secrets
import logging
from functools import wraps
from collections import defaultdict
from flask import request, session, abort, redirect, make_response

# ══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════

# SECURITY: Secret used for CSRF token HMAC — falls back to app secret
CSRF_SECRET = os.environ.get('CSRF_SECRET', '')

# Rate limit windows (seconds) and max requests
RATE_LIMITS = {
    'login':     {'window': 60, 'max': 5},     # 5 login attempts per minute
    'signup':    {'window': 60, 'max': 3},     # 3 signups per minute per IP
    'api':       {'window': 60, 'max': 60},    # 60 API calls per minute
    'default':   {'window': 60, 'max': 120},   # 120 page loads per minute
}

# SECURITY: Auto-block IP after this many violations in the window
IP_BLOCK_THRESHOLD = 20       # violations before block
IP_BLOCK_WINDOW    = 900      # 15-minute window
IP_BLOCK_DURATION  = 3600     # 1-hour block

# File upload restrictions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
MAX_FILE_SIZE_MB   = 5
MAX_FILENAME_LEN   = 100

# SQL injection patterns to detect in user input
# SECURITY: These patterns catch common SQLi probe strings
_SQLI_PATTERNS = re.compile(
    r"(\b(UNION|SELECT|INSERT|UPDATE|DELETE|DROP|ALTER|EXEC|EXECUTE|TRUNCATE|"
    r"DECLARE|xp_|sp_|0x)\b|--|;.*\b(DROP|ALTER|DELETE)\b|'.*OR.*'.*=|"
    r"\".*OR.*\".*=|'\s*;\s*--|1\s*=\s*1|1'\s*OR\s*'1)",
    re.IGNORECASE
)

# XSS patterns to detect in user input
_XSS_PATTERNS = re.compile(
    r"(<\s*script|<\s*img\s+[^>]*onerror|<\s*svg\s+[^>]*onload|"
    r"javascript\s*:|on\w+\s*=|<\s*iframe|<\s*object|<\s*embed|"
    r"<\s*link\s+[^>]*href\s*=\s*[\"']?javascript|"
    r"expression\s*\(|url\s*\(\s*[\"']?javascript)",
    re.IGNORECASE
)

# HTML tag stripper
_HTML_TAG_RE = re.compile(r'<[^>]+>')


# ══════════════════════════════════════════════════════════════════
# RATE LIMITER  (sliding-window, in-memory)
# ══════════════════════════════════════════════════════════════════

class RateLimiter:
    """
    In-memory sliding-window rate limiter.
    SECURITY: Prevents brute-force attacks, credential stuffing, and DoS.
    Tracks request timestamps per (IP, category) and rejects when over limit.
    """

    def __init__(self):
        # {(ip, category): [timestamp, ...]}
        self._hits = defaultdict(list)
        self._last_cleanup = time.time()

    def is_limited(self, ip, category='default'):
        """
        Check if the IP has exceeded the rate limit for this category.
        Returns (is_blocked: bool, retry_after_seconds: int).
        """
        cfg = RATE_LIMITS.get(category, RATE_LIMITS['default'])
        window = cfg['window']
        max_hits = cfg['max']
        now = time.time()
        key = (ip, category)

        # Prune old entries outside the window
        self._hits[key] = [t for t in self._hits[key] if now - t < window]

        if len(self._hits[key]) >= max_hits:
            oldest = self._hits[key][0]
            retry_after = int(window - (now - oldest)) + 1
            return True, retry_after

        self._hits[key].append(now)

        # Periodic cleanup of stale keys (every 5 min)
        if now - self._last_cleanup > 300:
            self._cleanup(now)

        return False, 0

    def _cleanup(self, now):
        """Remove entries older than the longest window."""
        max_window = max(c['window'] for c in RATE_LIMITS.values())
        stale_keys = [k for k, v in self._hits.items()
                      if not v or now - v[-1] > max_window]
        for k in stale_keys:
            del self._hits[k]
        self._last_cleanup = now


# Singleton
_rate_limiter = RateLimiter()


# ══════════════════════════════════════════════════════════════════
# IP BLOCKER
# ══════════════════════════════════════════════════════════════════

class IPBlocker:
    """
    SECURITY: Automatically blocks IPs that exhibit abusive patterns.
    Tracks failed auth attempts and rate-limit violations.
    After IP_BLOCK_THRESHOLD violations within IP_BLOCK_WINDOW seconds,
    the IP is blocked for IP_BLOCK_DURATION seconds.
    """

    def __init__(self):
        # {ip: [violation_timestamp, ...]}
        self._violations = defaultdict(list)
        # {ip: unblock_timestamp}
        self._blocked = {}

    def is_blocked(self, ip):
        """Check if an IP is currently blocked."""
        if ip in self._blocked:
            if time.time() < self._blocked[ip]:
                return True
            else:
                del self._blocked[ip]
        return False

    def record_violation(self, ip):
        """
        Record a violation for an IP. If threshold is exceeded, block it.
        Returns True if the IP was just blocked.
        """
        now = time.time()
        self._violations[ip] = [
            t for t in self._violations[ip] if now - t < IP_BLOCK_WINDOW
        ]
        self._violations[ip].append(now)

        if len(self._violations[ip]) >= IP_BLOCK_THRESHOLD:
            self._blocked[ip] = now + IP_BLOCK_DURATION
            self._violations[ip] = []
            return True
        return False

    def block_ip(self, ip, duration=None):
        """Manually block an IP."""
        self._blocked[ip] = time.time() + (duration or IP_BLOCK_DURATION)


# Singleton
_ip_blocker = IPBlocker()


# ══════════════════════════════════════════════════════════════════
# CSRF PROTECTION
# ══════════════════════════════════════════════════════════════════

def _get_csrf_secret(app):
    """Get the CSRF signing secret, falling back to the Flask secret key."""
    return CSRF_SECRET or app.secret_key or 'fallback-csrf-secret'


def generate_csrf_token():
    """
    Generate a per-session CSRF token.
    SECURITY: Uses HMAC-SHA256 with a server-side secret and a random nonce
    stored in the session. This prevents cross-site request forgery.
    """
    if '_csrf_nonce' not in session:
        session['_csrf_nonce'] = secrets.token_hex(32)

    nonce = session['_csrf_nonce']
    # We import current_app here to avoid circular imports
    from flask import current_app
    secret = _get_csrf_secret(current_app)
    signature = hmac.new(
        secret.encode(), nonce.encode(), hashlib.sha256
    ).hexdigest()
    return f"{nonce}:{signature}"


def validate_csrf_token(token):
    """
    Validate a CSRF token against the session nonce.
    SECURITY: Timing-safe comparison to prevent timing attacks.
    """
    if not token or ':' not in token:
        return False

    nonce, signature = token.split(':', 1)
    session_nonce = session.get('_csrf_nonce', '')

    if not hmac.compare_digest(nonce, session_nonce):
        return False

    from flask import current_app
    secret = _get_csrf_secret(current_app)
    expected = hmac.new(
        secret.encode(), nonce.encode(), hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(signature, expected)


# ══════════════════════════════════════════════════════════════════
# INPUT SANITISATION
# ══════════════════════════════════════════════════════════════════

def sanitize_input(value, max_length=500):
    """
    Sanitize a single string input.
    SECURITY: Strips HTML tags, enforces max length.
    Returns the cleaned string.
    """
    if not isinstance(value, str):
        return value
    # Strip HTML tags to prevent stored XSS
    cleaned = _HTML_TAG_RE.sub('', value)
    # Enforce max length to prevent buffer overflow / DoS
    return cleaned[:max_length].strip()


def detect_sqli(value):
    """
    SECURITY: Detect SQL injection patterns in input.
    Returns True if suspicious patterns are found.
    Note: This is a defense-in-depth measure — parameterized queries
    are the primary defense against SQLi.
    """
    if not isinstance(value, str):
        return False
    return bool(_SQLI_PATTERNS.search(value))


def detect_xss(value):
    """
    SECURITY: Detect XSS attack patterns in input.
    Returns True if suspicious patterns are found.
    Note: Jinja2's autoescaping is the primary XSS defense.
    """
    if not isinstance(value, str):
        return False
    return bool(_XSS_PATTERNS.search(value))


def is_input_malicious(value):
    """
    SECURITY: Combined check for SQLi and XSS patterns.
    Returns (is_malicious, attack_type).
    """
    if detect_sqli(value):
        return True, 'sqli'
    if detect_xss(value):
        return True, 'xss'
    return False, None


# ══════════════════════════════════════════════════════════════════
# FILE UPLOAD VALIDATION
# ══════════════════════════════════════════════════════════════════

def validate_file_upload(file_obj):
    """
    SECURITY: Validate an uploaded file.
    Checks: extension whitelist, file size, filename sanitization.
    Returns (is_valid, error_message, safe_filename).
    """
    if not file_obj or not file_obj.filename:
        return False, 'No file provided.', None

    filename = file_obj.filename

    # SECURITY: Check filename length
    if len(filename) > MAX_FILENAME_LEN:
        return False, 'Filename too long.', None

    # SECURITY: Check extension whitelist
    ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
    if ext not in ALLOWED_EXTENSIONS:
        return False, f'File type .{ext} not allowed. Use: {", ".join(ALLOWED_EXTENSIONS)}', None

    # SECURITY: Sanitize filename — remove path traversal and special chars
    safe_name = re.sub(r'[^\w\-.]', '_', filename)
    # Prevent path traversal
    safe_name = safe_name.replace('..', '_')

    # SECURITY: Check file size (read and rewind)
    file_obj.seek(0, 2)  # Seek to end
    size_mb = file_obj.tell() / (1024 * 1024)
    file_obj.seek(0)     # Rewind

    if size_mb > MAX_FILE_SIZE_MB:
        return False, f'File too large ({size_mb:.1f}MB). Max: {MAX_FILE_SIZE_MB}MB.', None

    return True, None, safe_name


# ══════════════════════════════════════════════════════════════════
# SECURITY HEADERS
# ══════════════════════════════════════════════════════════════════

def add_security_headers(response):
    """
    SECURITY: Add OWASP-recommended security headers to every response.
    These headers protect against XSS, clickjacking, MIME sniffing,
    and other browser-side attacks.
    """
    # SECURITY: Prevent clickjacking by disallowing framing
    response.headers['X-Frame-Options'] = 'DENY'

    # SECURITY: Prevent MIME-type sniffing
    response.headers['X-Content-Type-Options'] = 'nosniff'

    # SECURITY: Enable browser XSS filter (legacy, but no harm)
    response.headers['X-XSS-Protection'] = '1; mode=block'

    # SECURITY: Control referrer information leakage
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'

    # SECURITY: Restrict browser features (camera, mic, geolocation, etc.)
    response.headers['Permissions-Policy'] = (
        'camera=(), microphone=(self), geolocation=(), payment=()'
    )

    # SECURITY: Content Security Policy — whitelist allowed content sources
    # Allow Google's GIS library for OAuth, plus Google Fonts
    response.headers['Content-Security-Policy'] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' https://accounts.google.com https://apis.google.com; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
        "font-src 'self' https://fonts.gstatic.com; "
        "img-src 'self' data: https: http:; "
        "frame-src https://accounts.google.com; "
        "connect-src 'self' https://accounts.google.com https://oauth2.googleapis.com; "
        "object-src 'none'; "
        "base-uri 'self'; "
        "form-action 'self' https://accounts.google.com; "
    )

    # SECURITY: HSTS — force HTTPS for 1 year (only set in production)
    if request.is_secure or request.headers.get('X-Forwarded-Proto') == 'https':
        response.headers['Strict-Transport-Security'] = (
            'max-age=31536000; includeSubDomains; preload'
        )

    # SECURITY: Prevent caching of authenticated pages
    if 'user_id' in session:
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, private'
        response.headers['Pragma'] = 'no-cache'

    return response


# ══════════════════════════════════════════════════════════════════
# HTTPS ENFORCEMENT
# ══════════════════════════════════════════════════════════════════

def enforce_https():
    """
    SECURITY: Redirect HTTP to HTTPS in production.
    Checks the X-Forwarded-Proto header set by reverse proxies (Render, Heroku, etc.)
    Skips enforcement on localhost for development convenience.
    """
    if request.host.startswith('localhost') or request.host.startswith('127.0.0.1'):
        return None  # Skip in development

    proto = request.headers.get('X-Forwarded-Proto', 'http')
    if proto != 'https':
        url = request.url.replace('http://', 'https://', 1)
        return redirect(url, code=301)
    return None


# ══════════════════════════════════════════════════════════════════
# REQUEST GUARD  (before_request hook)
# ══════════════════════════════════════════════════════════════════

def get_client_ip():
    """
    SECURITY: Get the real client IP, accounting for reverse proxies.
    Checks X-Forwarded-For first, then falls back to remote_addr.
    """
    forwarded = request.headers.get('X-Forwarded-For', '')
    if forwarded:
        # First IP in the chain is the client
        return forwarded.split(',')[0].strip()
    return request.remote_addr or '0.0.0.0'


def request_guard():
    """
    SECURITY: Master before_request hook that runs all security checks:
    1. HTTPS enforcement
    2. IP block check
    3. Rate limiting
    4. CSRF validation (on POST/PUT/DELETE)
    5. Input sanitization & attack detection
    """
    from flask import current_app

    logger = current_app.logger if current_app else logging.getLogger('bookify')

    # ── 1. HTTPS enforcement ──
    https_redirect = enforce_https()
    if https_redirect:
        return https_redirect

    ip = get_client_ip()

    # ── 2. IP block check ──
    if _ip_blocker.is_blocked(ip):
        logger.warning(f"SECURITY: Blocked IP attempted access: {ip}")
        abort(403)

    # ── 3. Rate limiting ──
    # Determine category based on the endpoint
    path = request.path
    if path in ('/login', '/signup'):
        category = path.lstrip('/')
    elif path.startswith('/api/') or path.startswith('/recommend') or path.startswith('/mood') or path.startswith('/multimodal'):
        category = 'api'
    else:
        category = 'default'

    is_limited, retry_after = _rate_limiter.is_limited(ip, category)
    if is_limited:
        _ip_blocker.record_violation(ip)
        logger.warning(f"SECURITY: Rate limit exceeded for {ip} on {category}")
        response = make_response(
            f'Rate limit exceeded. Try again in {retry_after} seconds.', 429
        )
        response.headers['Retry-After'] = str(retry_after)
        return response

    # ── 4. CSRF validation on state-changing methods ──
    if request.method in ('POST', 'PUT', 'DELETE'):
        # Skip CSRF for specific endpoints that use other auth (Google callback, API JSON)
        csrf_exempt = ('/google/callback', '/analyze_cover', '/voice_search',
                       '/history', '/multimodal_recommend', '/mood_recommend',
                       '/recommend_books', '/autocomplete', '/rate', '/history/remove',
                       '/onboarding')
        if path not in csrf_exempt:
            token = (request.form.get('csrf_token') or
                     request.headers.get('X-CSRF-Token', ''))
            if not validate_csrf_token(token):
                logger.warning(f"SECURITY: CSRF validation failed for {ip} on {path}")
                _ip_blocker.record_violation(ip)
                abort(403)

    # ── 5. Input sanitization & attack detection ──
    # Check all form data and query params for injection attempts
    all_values = list(request.args.values()) + list(request.form.values())
    for val in all_values:
        is_malicious, attack_type = is_input_malicious(val)
        if is_malicious:
            logger.warning(
                f"SECURITY: {attack_type.upper()} attempt detected from {ip}: "
                f"{val[:100]}..."
            )
            _ip_blocker.record_violation(ip)
            abort(400)

    return None


# ══════════════════════════════════════════════════════════════════
# FLASK APP INTEGRATION
# ══════════════════════════════════════════════════════════════════

def init_security(app):
    """
    Register all security hooks with the Flask app.
    Call this once in app.py after creating the Flask app.

    Usage:
        from security import init_security
        init_security(app)
    """
    # Register before-request guard
    app.before_request(request_guard)

    # Register after-request security headers
    app.after_request(add_security_headers)

    # Make CSRF token available in all templates via Jinja2 globals
    app.jinja_env.globals['csrf_token'] = generate_csrf_token

    # SECURITY: Configure secure session cookies
    app.config.update(
        SESSION_COOKIE_HTTPONLY=True,    # SECURITY: Prevent JS access to cookies
        SESSION_COOKIE_SAMESITE='Lax',  # SECURITY: Prevent CSRF via cross-site requests
        SESSION_COOKIE_SECURE=False,    # Set True in production (requires HTTPS)
        PERMANENT_SESSION_LIFETIME=1800, # SECURITY: 30-minute session timeout
    )

    # SECURITY: In production, enforce secure cookies
    if os.environ.get('FLASK_ENV') == 'production':
        app.config['SESSION_COOKIE_SECURE'] = True

    # Secure error handlers that hide stack traces
    @app.errorhandler(400)
    def bad_request(e):
        return '<h1>400 — Bad Request</h1><p>Your request could not be processed.</p>', 400

    @app.errorhandler(403)
    def forbidden(e):
        return '<h1>403 — Forbidden</h1><p>Access denied.</p>', 403

    @app.errorhandler(404)
    def not_found(e):
        return '<h1>404 — Page Not Found</h1><p>The page you requested does not exist.</p>', 404

    @app.errorhandler(429)
    def too_many_requests(e):
        return '<h1>429 — Too Many Requests</h1><p>Please slow down.</p>', 429

    @app.errorhandler(500)
    def internal_error(e):
        # SECURITY: Never expose stack traces or internal details to users
        app.logger.error(f"Internal Server Error: {e}")
        return '<h1>500 — Internal Server Error</h1><p>Something went wrong.</p>', 500

    app.logger.info("✅ Security middleware initialized")
