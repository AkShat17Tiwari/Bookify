"""
BOOKIFY — Audit Logging Module
================================
Structured security event logging for:
  • Login attempts (success / failure)
  • Account creation
  • Account lockouts
  • Suspicious activity (SQLi, XSS, rate limit violations)
  • Admin actions

Events are written to both:
  1. security.log file (human-readable, for log aggregation)
  2. SQLite audit_log table (queryable, for dashboards)
"""

import sqlite3
import os
import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler

DB_PATH = os.path.join(os.path.dirname(__file__), 'users.db')
LOG_DIR = os.path.join(os.path.dirname(__file__), 'logs')

# ══════════════════════════════════════════════════════════════════
# FILE LOGGER SETUP
# ══════════════════════════════════════════════════════════════════

# Ensure logs directory exists
os.makedirs(LOG_DIR, exist_ok=True)

# SECURITY: Rotating log file — prevents disk exhaustion attacks
_security_logger = logging.getLogger('bookify.security')
_security_logger.setLevel(logging.INFO)

_file_handler = RotatingFileHandler(
    os.path.join(LOG_DIR, 'security.log'),
    maxBytes=5 * 1024 * 1024,   # 5 MB per file
    backupCount=5               # Keep 5 rotated files
)
_file_handler.setFormatter(
    logging.Formatter('[%(asctime)s] %(levelname)s — %(message)s')
)
_security_logger.addHandler(_file_handler)


# ══════════════════════════════════════════════════════════════════
# DATABASE TABLE
# ══════════════════════════════════════════════════════════════════

def init_audit_db():
    """Create the audit_log table if it doesn't exist."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            event_type TEXT NOT NULL,
            severity TEXT NOT NULL DEFAULT 'INFO',
            ip_address TEXT,
            user_email TEXT,
            user_agent TEXT,
            details TEXT,
            path TEXT
        )
    ''')
    # SECURITY: Index for fast queries by event_type and timestamp
    conn.execute('''
        CREATE INDEX IF NOT EXISTS idx_audit_event_time
        ON audit_log (event_type, timestamp)
    ''')
    conn.commit()
    conn.close()


# Initialize on import
init_audit_db()


# ══════════════════════════════════════════════════════════════════
# EVENT TYPES
# ══════════════════════════════════════════════════════════════════

# Constants for event types — keep them consistent
LOGIN_SUCCESS    = 'LOGIN_SUCCESS'
LOGIN_FAILURE    = 'LOGIN_FAILURE'
ACCOUNT_CREATED  = 'ACCOUNT_CREATED'
ACCOUNT_LOCKED   = 'ACCOUNT_LOCKED'
ACCOUNT_UNLOCKED = 'ACCOUNT_UNLOCKED'
LOGOUT           = 'LOGOUT'
SQLI_ATTEMPT     = 'SQLI_ATTEMPT'
XSS_ATTEMPT      = 'XSS_ATTEMPT'
RATE_LIMITED      = 'RATE_LIMITED'
IP_BLOCKED       = 'IP_BLOCKED'
CSRF_FAILURE     = 'CSRF_FAILURE'
ADMIN_ACTION     = 'ADMIN_ACTION'
FILE_UPLOAD      = 'FILE_UPLOAD'
SUSPICIOUS       = 'SUSPICIOUS'

# Severity mapping
_SEVERITY = {
    LOGIN_SUCCESS:    'INFO',
    LOGIN_FAILURE:    'WARNING',
    ACCOUNT_CREATED:  'INFO',
    ACCOUNT_LOCKED:   'WARNING',
    ACCOUNT_UNLOCKED: 'INFO',
    LOGOUT:           'INFO',
    SQLI_ATTEMPT:     'CRITICAL',
    XSS_ATTEMPT:      'CRITICAL',
    RATE_LIMITED:      'WARNING',
    IP_BLOCKED:        'CRITICAL',
    CSRF_FAILURE:     'WARNING',
    ADMIN_ACTION:     'INFO',
    FILE_UPLOAD:      'INFO',
    SUSPICIOUS:       'WARNING',
}


# ══════════════════════════════════════════════════════════════════
# CORE LOGGING FUNCTION
# ══════════════════════════════════════════════════════════════════

def log_security_event(event_type, ip=None, email=None, user_agent=None,
                       details=None, path=None):
    """
    Log a security event to both the log file and the database.

    Args:
        event_type: One of the event type constants above.
        ip: Client IP address.
        email: User email (if applicable).
        user_agent: Client User-Agent header.
        details: Additional context string.
        path: Request path.
    """
    severity = _SEVERITY.get(event_type, 'INFO')

    # ── File log ──
    msg = f"[{event_type}] IP={ip or 'N/A'} email={email or 'N/A'} path={path or 'N/A'}"
    if details:
        msg += f" | {details}"

    if severity == 'CRITICAL':
        _security_logger.critical(msg)
    elif severity == 'WARNING':
        _security_logger.warning(msg)
    else:
        _security_logger.info(msg)

    # ── Database log ──
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            '''INSERT INTO audit_log
               (event_type, severity, ip_address, user_email, user_agent, details, path)
               VALUES (?, ?, ?, ?, ?, ?, ?)''',
            (event_type, severity, ip, email,
             (user_agent or '')[:500], (details or '')[:1000], path)
        )
        conn.commit()
        conn.close()
    except Exception as e:
        _security_logger.error(f"Failed to write audit log to DB: {e}")


# ══════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ══════════════════════════════════════════════════════════════════

def log_login_success(ip, email, user_agent=None, path='/login'):
    """Log a successful login."""
    log_security_event(LOGIN_SUCCESS, ip, email, user_agent, path=path)


def log_login_failure(ip, email, reason=None, user_agent=None, path='/login'):
    """Log a failed login attempt."""
    log_security_event(LOGIN_FAILURE, ip, email, user_agent,
                       details=f"Reason: {reason}", path=path)


def log_account_created(ip, email, method='email', user_agent=None):
    """Log a new account creation."""
    log_security_event(ACCOUNT_CREATED, ip, email, user_agent,
                       details=f"Method: {method}", path='/signup')


def log_account_locked(ip, email, reason='too many failed attempts'):
    """Log an account lockout."""
    log_security_event(ACCOUNT_LOCKED, ip, email, details=reason)


def log_suspicious_activity(ip, activity_type, details=None, path=None):
    """Log a suspicious activity (SQLi, XSS, etc.)."""
    event = {
        'sqli': SQLI_ATTEMPT,
        'xss': XSS_ATTEMPT,
        'rate_limit': RATE_LIMITED,
        'ip_block': IP_BLOCKED,
        'csrf': CSRF_FAILURE,
    }.get(activity_type, SUSPICIOUS)
    log_security_event(event, ip, details=details, path=path)


def get_recent_events(event_type=None, limit=50):
    """
    Query recent security events from the database.
    Returns a list of dicts.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    if event_type:
        rows = conn.execute(
            'SELECT * FROM audit_log WHERE event_type = ? ORDER BY id DESC LIMIT ?',
            (event_type, limit)
        ).fetchall()
    else:
        rows = conn.execute(
            'SELECT * FROM audit_log ORDER BY id DESC LIMIT ?',
            (limit,)
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]
