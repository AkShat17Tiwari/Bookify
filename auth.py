"""
BOOKIFY — Authentication & Authorization Module
=================================================
Production-ready auth with:
  • Bcrypt password hashing (cost factor 12)
  • Role-Based Access Control (admin / user)
  • Brute-force protection (account lockout)
  • Session hardening (IP binding, regeneration)
  • Audit logging integration
  • Parameterized queries (anti-SQLi)

All database queries use parameterized statements — the PRIMARY
defense against SQL injection.
"""

import sqlite3
import os
import bcrypt
from functools import wraps
from datetime import datetime, timedelta
from flask import session, redirect, url_for, request

DB_PATH = os.path.join(os.path.dirname(__file__), 'users.db')

# SECURITY: Bcrypt cost factor — higher = slower + more secure
# 12 is the recommended minimum for production
BCRYPT_ROUNDS = 12

# SECURITY: Account lockout settings
MAX_FAILED_ATTEMPTS = 5        # Lock after 5 failures
LOCKOUT_DURATION_MIN = 15      # Lock for 15 minutes


# ══════════════════════════════════════════════════════════════════
# DATABASE ACCESS
# ══════════════════════════════════════════════════════════════════

def get_db():
    """
    Get a database connection with row_factory for dict-style access.
    SECURITY: Each call opens a fresh connection to avoid shared state issues.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """
    Create / migrate the users table and wishlist table.
    SECURITY: Adds role, failed_attempts, and locked_until columns
    for RBAC and brute-force protection.
    """
    conn = get_db()

    # Create users table if it doesn't exist (with new columns)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT,
            google_id TEXT UNIQUE,
            role TEXT NOT NULL DEFAULT 'user',
            failed_attempts INTEGER NOT NULL DEFAULT 0,
            locked_until TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Create wishlist table
    conn.execute('''
        CREATE TABLE IF NOT EXISTS wishlist (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            book_title TEXT NOT NULL,
            book_author TEXT NOT NULL DEFAULT '',
            book_image TEXT NOT NULL DEFAULT '',
            added_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(user_id, book_title),
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    ''')

    # Create reading_history table
    conn.execute('''
        CREATE TABLE IF NOT EXISTS reading_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            book_title TEXT NOT NULL,
            book_author TEXT NOT NULL DEFAULT '',
            book_image TEXT NOT NULL DEFAULT '',
            read_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(user_id, book_title),
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    ''')

    # Create ratings table
    conn.execute('''
        CREATE TABLE IF NOT EXISTS ratings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            book_title TEXT NOT NULL,
            rating INTEGER NOT NULL CHECK(rating >= 1 AND rating <= 5),
            rated_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(user_id, book_title),
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    ''')

    # Create user_preferences table (for onboarding genre quiz)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS user_preferences (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            genre TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(user_id, genre),
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    ''')

    # MIGRATION: Add new columns to existing tables if they don't exist
    # SQLite doesn't support IF NOT EXISTS for ALTER TABLE, so we try/except
    for col, col_def in [
        ('role', "TEXT NOT NULL DEFAULT 'user'"),
        ('failed_attempts', 'INTEGER NOT NULL DEFAULT 0'),
        ('locked_until', 'TEXT'),
    ]:
        try:
            conn.execute(f'ALTER TABLE users ADD COLUMN {col} {col_def}')
        except sqlite3.OperationalError:
            pass  # Column already exists

    conn.commit()
    conn.close()
    print("✅ User database initialized (with RBAC + lockout + wishlist + history + ratings support)")


# ══════════════════════════════════════════════════════════════════
# PASSWORD HASHING  (bcrypt)
# ══════════════════════════════════════════════════════════════════

def hash_password(password):
    """
    SECURITY: Hash a password using bcrypt with the configured cost factor.
    Bcrypt is designed to be slow, making brute-force attacks impractical.
    The salt is automatically generated and embedded in the hash.
    """
    return bcrypt.hashpw(
        password.encode('utf-8'),
        bcrypt.gensalt(rounds=BCRYPT_ROUNDS)
    ).decode('utf-8')


def verify_password(password, hashed):
    """
    SECURITY: Verify a password against a bcrypt hash.
    Bcrypt's comparison is constant-time, preventing timing attacks.
    Also handles legacy werkzeug hashes for backward compatibility.
    """
    if not hashed:
        return False

    # Handle legacy werkzeug hashes (from before the bcrypt migration)
    if hashed.startswith(('pbkdf2:', 'scrypt:')):
        try:
            from werkzeug.security import check_password_hash
            return check_password_hash(hashed, password)
        except Exception:
            return False

    # Bcrypt verification
    try:
        return bcrypt.checkpw(
            password.encode('utf-8'),
            hashed.encode('utf-8')
        )
    except Exception:
        return False


# ══════════════════════════════════════════════════════════════════
# BRUTE-FORCE PROTECTION
# ══════════════════════════════════════════════════════════════════

def _is_account_locked(user):
    """
    SECURITY: Check if an account is currently locked due to failed attempts.
    Returns (is_locked, minutes_remaining).
    """
    locked_until = user.get('locked_until')
    if not locked_until:
        return False, 0

    try:
        lock_time = datetime.fromisoformat(locked_until)
        now = datetime.now()
        if now < lock_time:
            remaining = int((lock_time - now).total_seconds() / 60) + 1
            return True, remaining
        else:
            # Lock has expired — reset
            _reset_failed_attempts(user['email'])
            return False, 0
    except (ValueError, TypeError):
        return False, 0


def _increment_failed_attempts(email):
    """
    SECURITY: Increment the failed login counter for an email.
    If it exceeds MAX_FAILED_ATTEMPTS, lock the account.
    Returns True if the account was just locked.
    """
    conn = get_db()
    user = conn.execute('SELECT * FROM users WHERE email = ?', (email,)).fetchone()
    if not user:
        conn.close()
        return False

    new_count = (user['failed_attempts'] or 0) + 1

    if new_count >= MAX_FAILED_ATTEMPTS:
        lock_until = (datetime.now() + timedelta(minutes=LOCKOUT_DURATION_MIN)).isoformat()
        conn.execute(
            'UPDATE users SET failed_attempts = ?, locked_until = ? WHERE email = ?',
            (new_count, lock_until, email)
        )
        conn.commit()
        conn.close()

        # Log the lockout event
        try:
            from audit_log import log_account_locked
            from security import get_client_ip
            log_account_locked(get_client_ip(), email)
        except Exception:
            pass

        return True
    else:
        conn.execute(
            'UPDATE users SET failed_attempts = ? WHERE email = ?',
            (new_count, email)
        )
        conn.commit()
        conn.close()
        return False


def _reset_failed_attempts(email):
    """Reset failed attempt counter and unlock the account."""
    conn = get_db()
    conn.execute(
        'UPDATE users SET failed_attempts = 0, locked_until = NULL WHERE email = ?',
        (email,)
    )
    conn.commit()
    conn.close()


# ══════════════════════════════════════════════════════════════════
# USER MANAGEMENT
# ══════════════════════════════════════════════════════════════════

def create_user(name, email, password=None, google_id=None, role='user'):
    """
    Create a new user. Returns (user_dict, error_string).
    SECURITY: Password is bcrypt-hashed before storage.
    All queries use parameterized statements to prevent SQLi.
    """
    conn = get_db()
    try:
        pw_hash = hash_password(password) if password else None
        conn.execute(
            'INSERT INTO users (name, email, password_hash, google_id, role) VALUES (?, ?, ?, ?, ?)',
            (name, email, pw_hash, google_id, role)
        )
        conn.commit()
        user = conn.execute('SELECT * FROM users WHERE email = ?', (email,)).fetchone()
        conn.close()

        # Audit log
        try:
            from audit_log import log_account_created
            from security import get_client_ip
            method = 'google' if google_id else 'email'
            log_account_created(get_client_ip(), email, method)
        except Exception:
            pass

        return dict(user), None
    except sqlite3.IntegrityError:
        conn.close()
        return None, 'An account with this email already exists.'


def verify_user(email, password):
    """
    Verify email/password. Returns (user_dict, error_string).
    SECURITY: Checks account lockout, verifies password with bcrypt,
    increments failure counter on wrong password, resets on success.
    """
    conn = get_db()
    user = conn.execute('SELECT * FROM users WHERE email = ?', (email,)).fetchone()
    conn.close()

    ip = '0.0.0.0'
    ua = ''
    try:
        from security import get_client_ip
        ip = get_client_ip()
        ua = request.headers.get('User-Agent', '')
    except Exception:
        pass

    if not user:
        # SECURITY: Log failure but use a generic message to prevent user enumeration
        try:
            from audit_log import log_login_failure
            log_login_failure(ip, email, 'account not found', ua)
        except Exception:
            pass
        return None, 'Invalid email or password.'

    user_dict = dict(user)

    # SECURITY: Check if account is locked
    is_locked, minutes = _is_account_locked(user_dict)
    if is_locked:
        try:
            from audit_log import log_login_failure
            log_login_failure(ip, email, 'account locked', ua)
        except Exception:
            pass
        return None, f'Account is locked. Try again in {minutes} minutes.'

    if not user_dict.get('password_hash'):
        return None, 'This account uses Google Sign-In. Please sign in with Google.'

    if not verify_password(password, user_dict['password_hash']):
        # SECURITY: Increment failed attempts
        just_locked = _increment_failed_attempts(email)
        try:
            from audit_log import log_login_failure
            log_login_failure(ip, email, 'wrong password', ua)
        except Exception:
            pass

        if just_locked:
            return None, f'Account locked after {MAX_FAILED_ATTEMPTS} failed attempts. Try again in {LOCKOUT_DURATION_MIN} minutes.'
        return None, 'Invalid email or password.'

    # SECURITY: Reset failed attempts on successful login
    _reset_failed_attempts(email)

    # Audit log — success
    try:
        from audit_log import log_login_success
        log_login_success(ip, email, ua)
    except Exception:
        pass

    return user_dict, None


def get_user_by_email(email):
    """Look up a user by email. Uses parameterized query."""
    conn = get_db()
    user = conn.execute('SELECT * FROM users WHERE email = ?', (email,)).fetchone()
    conn.close()
    return dict(user) if user else None


def get_user_by_google_id(google_id):
    """Look up a user by Google ID. Uses parameterized query."""
    conn = get_db()
    user = conn.execute('SELECT * FROM users WHERE google_id = ?', (google_id,)).fetchone()
    conn.close()
    return dict(user) if user else None


def get_current_user():
    """Get the currently logged-in user from session."""
    user_id = session.get('user_id')
    if not user_id:
        return None
    conn = get_db()
    user = conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
    conn.close()
    return dict(user) if user else None


# ══════════════════════════════════════════════════════════════════
# SESSION MANAGEMENT
# ══════════════════════════════════════════════════════════════════

def login_user(user):
    """
    Store user in session with hardened attributes.
    SECURITY: Records login IP and timestamp for audit trail.
    Makes session permanent with the configured timeout.
    """
    session.permanent = True     # SECURITY: Use PERMANENT_SESSION_LIFETIME
    session['user_id'] = user['id']
    session['user_name'] = user['name']
    session['user_email'] = user['email']
    session['user_role'] = user.get('role', 'user')

    # SECURITY: Track login metadata for session validation
    try:
        from security import get_client_ip
        session['login_ip'] = get_client_ip()
    except Exception:
        session['login_ip'] = request.remote_addr
    session['login_time'] = datetime.now().isoformat()


def logout_user():
    """
    SECURITY: Clear the entire session on logout, not just specific keys.
    This prevents any stale session data from persisting.
    """
    session.clear()


# ══════════════════════════════════════════════════════════════════
# DECORATORS (Authentication + Authorization)
# ══════════════════════════════════════════════════════════════════

def login_required(f):
    """
    Decorator to protect routes — redirects to /login if not authenticated.
    SECURITY: Validates that the session contains a valid user_id.
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login_page', next=request.path))
        return f(*args, **kwargs)
    return decorated_function


def admin_required(f):
    """
    Decorator to protect admin routes — requires 'admin' role.
    SECURITY: Checks both authentication AND authorization.
    Returns 403 if user is not an admin.
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login_page', next=request.path))
        if session.get('user_role') != 'admin':
            from flask import abort
            abort(403)
        return f(*args, **kwargs)
    return decorated_function


# ══════════════════════════════════════════════════════════════════
# ADMIN UTILITIES
# ══════════════════════════════════════════════════════════════════

def promote_to_admin(email):
    """Promote a user to admin role. Returns success boolean."""
    conn = get_db()
    result = conn.execute(
        'UPDATE users SET role = ? WHERE email = ?', ('admin', email)
    )
    conn.commit()
    changed = result.rowcount > 0
    conn.close()
    return changed


def list_users(limit=100):
    """List all users (for admin dashboard). Excludes password hashes."""
    conn = get_db()
    rows = conn.execute(
        'SELECT id, name, email, role, failed_attempts, locked_until, created_at '
        'FROM users ORDER BY id DESC LIMIT ?',
        (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def unlock_account(email):
    """Manually unlock a locked account."""
    _reset_failed_attempts(email)
    return True


# ══════════════════════════════════════════════════════════════════
# WISHLIST
# ══════════════════════════════════════════════════════════════════

def add_to_wishlist(user_id, book_title, book_author='', book_image=''):
    """Add a book to a user's wishlist. Returns (success, error)."""
    conn = get_db()
    try:
        conn.execute(
            'INSERT OR IGNORE INTO wishlist (user_id, book_title, book_author, book_image) VALUES (?, ?, ?, ?)',
            (user_id, book_title, book_author, book_image)
        )
        conn.commit()
        added = conn.execute(
            'SELECT COUNT(*) FROM wishlist WHERE user_id=? AND book_title=?',
            (user_id, book_title)
        ).fetchone()[0]
        conn.close()
        return True, None
    except Exception as e:
        conn.close()
        return False, str(e)


def remove_from_wishlist(user_id, book_title):
    """Remove a book from a user's wishlist."""
    conn = get_db()
    conn.execute(
        'DELETE FROM wishlist WHERE user_id=? AND book_title=?',
        (user_id, book_title)
    )
    conn.commit()
    conn.close()


def get_wishlist(user_id):
    """Return the user's wishlist as a list of dicts."""
    conn = get_db()
    rows = conn.execute(
        'SELECT book_title, book_author, book_image, added_at '
        'FROM wishlist WHERE user_id=? ORDER BY added_at DESC',
        (user_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def is_wishlisted(user_id, book_title):
    """Check if a specific book is in the user's wishlist."""
    conn = get_db()
    row = conn.execute(
        'SELECT 1 FROM wishlist WHERE user_id=? AND book_title=?',
        (user_id, book_title)
    ).fetchone()
    conn.close()
    return row is not None


# ══════════════════════════════════════════════════════════════════
# READING HISTORY
# ══════════════════════════════════════════════════════════════════

def add_to_history(user_id, book_title, book_author='', book_image=''):
    """Add a book to a user's reading history. Upserts (updates timestamp if exists)."""
    conn = get_db()
    try:
        conn.execute(
            '''INSERT INTO reading_history (user_id, book_title, book_author, book_image)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(user_id, book_title) DO UPDATE SET read_at = CURRENT_TIMESTAMP''',
            (user_id, book_title, book_author, book_image)
        )
        conn.commit()
        conn.close()
        return True, None
    except Exception as e:
        conn.close()
        return False, str(e)


def get_reading_history(user_id, limit=50):
    """Return the user's reading history as a list of dicts, most recent first."""
    conn = get_db()
    rows = conn.execute(
        'SELECT book_title, book_author, book_image, read_at '
        'FROM reading_history WHERE user_id=? ORDER BY read_at DESC LIMIT ?',
        (user_id, limit)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def remove_from_history(user_id, book_title):
    """Remove a book from a user's reading history."""
    conn = get_db()
    conn.execute(
        'DELETE FROM reading_history WHERE user_id=? AND book_title=?',
        (user_id, book_title)
    )
    conn.commit()
    conn.close()


def get_history_titles(user_id):
    """Return just the titles from reading history (for genre analysis)."""
    conn = get_db()
    rows = conn.execute(
        'SELECT book_title FROM reading_history WHERE user_id=? ORDER BY read_at DESC',
        (user_id,)
    ).fetchall()
    conn.close()
    return [r['book_title'] for r in rows]


# ══════════════════════════════════════════════════════════════════
# RATINGS
# ══════════════════════════════════════════════════════════════════

def rate_book(user_id, book_title, rating):
    """Rate a book 1-5. Upserts (updates rating if already rated)."""
    if not isinstance(rating, int) or rating < 1 or rating > 5:
        return False, 'Rating must be an integer between 1 and 5.'
    conn = get_db()
    try:
        conn.execute(
            '''INSERT INTO ratings (user_id, book_title, rating)
               VALUES (?, ?, ?)
               ON CONFLICT(user_id, book_title) DO UPDATE SET rating = ?, rated_at = CURRENT_TIMESTAMP''',
            (user_id, book_title, rating, rating)
        )
        conn.commit()
        conn.close()
        return True, None
    except Exception as e:
        conn.close()
        return False, str(e)


def get_user_rating(user_id, book_title):
    """Get a user's rating for a specific book. Returns int or None."""
    conn = get_db()
    row = conn.execute(
        'SELECT rating FROM ratings WHERE user_id=? AND book_title=?',
        (user_id, book_title)
    ).fetchone()
    conn.close()
    return row['rating'] if row else None


def get_user_ratings(user_id, limit=100):
    """Return all books rated by a user, most recent first."""
    conn = get_db()
    rows = conn.execute(
        'SELECT book_title, rating, rated_at '
        'FROM ratings WHERE user_id=? ORDER BY rated_at DESC LIMIT ?',
        (user_id, limit)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_book_avg_rating(book_title):
    """Get the average user rating for a book. Returns (avg, count) or (None, 0)."""
    conn = get_db()
    row = conn.execute(
        'SELECT AVG(rating) as avg_rating, COUNT(*) as count '
        'FROM ratings WHERE book_title=?',
        (book_title,)
    ).fetchone()
    conn.close()
    if row and row['count'] > 0:
        return round(row['avg_rating'], 1), row['count']
    return None, 0


# ══════════════════════════════════════════════════════════════════
# USER PREFERENCES (Onboarding)
# ══════════════════════════════════════════════════════════════════

def save_genre_preferences(user_id, genres):
    """Save a list of genre preferences for a user (replaces existing)."""
    conn = get_db()
    try:
        conn.execute('DELETE FROM user_preferences WHERE user_id=?', (user_id,))
        for genre in genres:
            conn.execute(
                'INSERT OR IGNORE INTO user_preferences (user_id, genre) VALUES (?, ?)',
                (user_id, genre.strip())
            )
        conn.commit()
        conn.close()
        return True, None
    except Exception as e:
        conn.close()
        return False, str(e)


def get_genre_preferences(user_id):
    """Get a user's saved genre preferences."""
    conn = get_db()
    rows = conn.execute(
        'SELECT genre FROM user_preferences WHERE user_id=? ORDER BY created_at',
        (user_id,)
    ).fetchall()
    conn.close()
    return [r['genre'] for r in rows]


def has_completed_onboarding(user_id):
    """Check if a user has completed the onboarding quiz."""
    conn = get_db()
    count = conn.execute(
        'SELECT COUNT(*) FROM user_preferences WHERE user_id=?',
        (user_id,)
    ).fetchone()[0]
    conn.close()
    return count > 0


# Initialize database on import
init_db()
