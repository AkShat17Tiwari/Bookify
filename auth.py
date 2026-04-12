"""
BOOKIFY — Authentication & Authorization Module (Clerk Integration)
====================================================================
Uses Clerk for authentication:
  • Frontend: Clerk JS SDK handles sign-in/sign-up UI
  • Backend: Verifies Clerk session tokens via clerk-backend-api
  • Local DB: Stores user data, wishlist, history, ratings, preferences

All database queries use parameterized statements — the PRIMARY
defense against SQL injection.
"""

import sqlite3
import os
from functools import wraps
from datetime import datetime
from flask import session, redirect, request, jsonify

DB_PATH = os.path.join(os.path.dirname(__file__), 'users.db')

# Clerk keys (loaded from environment)
CLERK_SECRET_KEY = os.environ.get('CLERK_SECRET_KEY', '')
CLERK_PUBLISHABLE_KEY = os.environ.get('CLERK_PUBLISHABLE_KEY', '')


# ══════════════════════════════════════════════════════════════════
# DATABASE ACCESS
# ══════════════════════════════════════════════════════════════════

def get_db():
    """
    Get a database connection with row_factory for dict-style access.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """
    Create / migrate the users table and related tables.
    Adds clerk_id column for Clerk integration.
    """
    conn = get_db()

    # Create users table with clerk_id
    conn.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT,
            google_id TEXT UNIQUE,
            clerk_id TEXT UNIQUE,
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
    for col, col_def in [
        ('role', "TEXT NOT NULL DEFAULT 'user'"),
        ('failed_attempts', 'INTEGER NOT NULL DEFAULT 0'),
        ('locked_until', 'TEXT'),
        ('clerk_id', 'TEXT'),
    ]:
        try:
            conn.execute(f'ALTER TABLE users ADD COLUMN {col} {col_def}')
        except sqlite3.OperationalError:
            pass  # Column already exists

    conn.commit()
    conn.close()
    print("✅ User database initialized (Clerk auth + wishlist + history + ratings)")


# ══════════════════════════════════════════════════════════════════
# CLERK SESSION VERIFICATION
# ══════════════════════════════════════════════════════════════════

def _verify_clerk_session():
    """
    Verify the Clerk session from the __session cookie.
    Returns the Clerk user ID if valid, None otherwise.
    Uses PyJWT to decode the session token.
    """
    session_token = request.cookies.get('__session')
    if not session_token:
        return None, None, None

    try:
        import jwt
        import httpx

        # Decode the Clerk publishable key to get the frontend API domain
        # pk_test_<base64 encoded domain>
        import base64
        pk = CLERK_PUBLISHABLE_KEY
        if pk.startswith('pk_test_') or pk.startswith('pk_live_'):
            encoded = pk.split('_', 2)[2]
            # Add padding if needed
            padding = 4 - len(encoded) % 4
            if padding != 4:
                encoded += '=' * padding
            frontend_api = base64.b64decode(encoded).decode('utf-8').rstrip('$')
        else:
            frontend_api = 'clerk.accounts.dev'

        # Fetch JWKS from Clerk
        jwks_url = f'https://{frontend_api}/.well-known/jwks.json'
        resp = httpx.get(jwks_url, timeout=5)
        jwks = resp.json()

        # Get the signing key
        header = jwt.get_unverified_header(session_token)
        kid = header.get('kid')

        # Find matching key
        signing_key = None
        for key_data in jwks.get('keys', []):
            if key_data.get('kid') == kid:
                from jwt.algorithms import RSAAlgorithm
                signing_key = RSAAlgorithm.from_jwk(key_data)
                break

        if not signing_key:
            print("⚠️ Clerk: No matching signing key found")
            return None, None, None

        # Verify and decode the token
        payload = jwt.decode(
            session_token,
            signing_key,
            algorithms=['RS256'],
            options={
                'verify_aud': False,  # Clerk doesn't set aud in session tokens
                'verify_iss': True,
            },
            issuer=f'https://{frontend_api}',
        )

        clerk_user_id = payload.get('sub')
        email = payload.get('email', '')
        name = payload.get('name', '')

        # If no email/name in token, try to get from Clerk API
        if clerk_user_id and (not email or not name):
            try:
                from clerk_backend_api import Clerk
                sdk = Clerk(bearer_auth=CLERK_SECRET_KEY)
                clerk_user = sdk.users.get(user_id=clerk_user_id)
                if clerk_user:
                    if not email:
                        emails = clerk_user.email_addresses or []
                        if emails:
                            email = emails[0].email_address
                    if not name:
                        name = f"{clerk_user.first_name or ''} {clerk_user.last_name or ''}".strip()
                        if not name:
                            name = email.split('@')[0] if email else 'User'
            except Exception as e:
                print(f"⚠️ Clerk API lookup failed: {e}")
                if not name:
                    name = email.split('@')[0] if email else 'User'

        return clerk_user_id, email, name

    except jwt.ExpiredSignatureError:
        return None, None, None
    except jwt.InvalidTokenError as e:
        print(f"⚠️ Clerk token invalid: {e}")
        return None, None, None
    except Exception as e:
        print(f"⚠️ Clerk session verification error: {e}")
        return None, None, None


def _verify_bearer_token(token):
    """
    Verify a Bearer token from the Authorization header (React frontend).
    Uses the same JWT verification logic as _verify_clerk_session.
    """
    if not token:
        return None, None, None

    try:
        import jwt
        import httpx
        import base64

        pk = CLERK_PUBLISHABLE_KEY
        if pk.startswith('pk_test_') or pk.startswith('pk_live_'):
            encoded = pk.split('_', 2)[2]
            padding = 4 - len(encoded) % 4
            if padding != 4:
                encoded += '=' * padding
            frontend_api = base64.b64decode(encoded).decode('utf-8').rstrip('$')
        else:
            frontend_api = 'clerk.accounts.dev'

        jwks_url = f'https://{frontend_api}/.well-known/jwks.json'
        resp = httpx.get(jwks_url, timeout=5)
        jwks = resp.json()

        header = jwt.get_unverified_header(token)
        kid = header.get('kid')

        signing_key = None
        for key_data in jwks.get('keys', []):
            if key_data.get('kid') == kid:
                from jwt.algorithms import RSAAlgorithm
                signing_key = RSAAlgorithm.from_jwk(key_data)
                break

        if not signing_key:
            return None, None, None

        payload = jwt.decode(
            token,
            signing_key,
            algorithms=['RS256'],
            options={'verify_aud': False, 'verify_iss': True},
            issuer=f'https://{frontend_api}',
        )

        clerk_user_id = payload.get('sub')
        email = payload.get('email', '')
        name = payload.get('name', '')

        if clerk_user_id and (not email or not name):
            try:
                from clerk_backend_api import Clerk
                sdk = Clerk(bearer_auth=CLERK_SECRET_KEY)
                clerk_user = sdk.users.get(user_id=clerk_user_id)
                if clerk_user:
                    if not email:
                        emails = clerk_user.email_addresses or []
                        if emails:
                            email = emails[0].email_address
                    if not name:
                        name = f"{clerk_user.first_name or ''} {clerk_user.last_name or ''}".strip()
                        if not name:
                            name = email.split('@')[0] if email else 'User'
            except Exception as e:
                print(f"⚠️ Clerk API lookup failed: {e}")
                if not name:
                    name = email.split('@')[0] if email else 'User'

        return clerk_user_id, email, name

    except Exception as e:
        print(f"⚠️ Bearer token verification error: {e}")
        return None, None, None


def _get_or_create_local_user(clerk_id, email, name):
    """
    Get or create a local user record from Clerk user data.
    This maps the Clerk user to the local SQLite database.
    """
    if not clerk_id:
        return None

    conn = get_db()

    # Try to find by clerk_id first
    user = conn.execute('SELECT * FROM users WHERE clerk_id = ?', (clerk_id,)).fetchone()
    if user:
        # Update email/name if changed
        conn.execute(
            'UPDATE users SET email = ?, name = ? WHERE clerk_id = ?',
            (email or user['email'], name or user['name'], clerk_id)
        )
        conn.commit()
        user = conn.execute('SELECT * FROM users WHERE clerk_id = ?', (clerk_id,)).fetchone()
        conn.close()
        return dict(user)

    # Try to find by email (existing user migrating to Clerk)
    if email:
        user = conn.execute('SELECT * FROM users WHERE email = ?', (email,)).fetchone()
        if user:
            # Link existing account to Clerk
            conn.execute(
                'UPDATE users SET clerk_id = ?, name = ? WHERE email = ?',
                (clerk_id, name or user['name'], email)
            )
            conn.commit()
            user = conn.execute('SELECT * FROM users WHERE email = ?', (email,)).fetchone()
            conn.close()
            return dict(user)

    # Create new user
    try:
        conn.execute(
            'INSERT INTO users (name, email, clerk_id, role) VALUES (?, ?, ?, ?)',
            (name or 'User', email or f'{clerk_id}@clerk.user', clerk_id, 'user')
        )
        conn.commit()
        user = conn.execute('SELECT * FROM users WHERE clerk_id = ?', (clerk_id,)).fetchone()
        conn.close()
        return dict(user)
    except sqlite3.IntegrityError:
        conn.close()
        return None


# ══════════════════════════════════════════════════════════════════
# USER ACCESS FUNCTIONS
# ══════════════════════════════════════════════════════════════════

def get_current_user():
    """
    Get the currently authenticated user.
    First checks Clerk session token, then falls back to Flask session.
    """
    # Try Clerk session first
    clerk_id, email, name = _verify_clerk_session()
    if clerk_id:
        user = _get_or_create_local_user(clerk_id, email, name)
        if user:
            # Also set Flask session for backward compatibility
            session['user_id'] = user['id']
            session['user_name'] = user['name']
            session['user_email'] = user.get('email', '')
            return user

    # Fallback to Flask session (for legacy support during migration)
    user_id = session.get('user_id')
    if not user_id:
        return None
    conn = get_db()
    user = conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
    conn.close()
    return dict(user) if user else None


def login_required(f):
    """
    Decorator to protect routes — redirects to /auth if not authenticated.
    Checks:
      1. Bearer token from Authorization header (React frontend)
      2. Clerk __session cookie
      3. Flask session
    Returns JSON 401 for API/AJAX requests instead of HTML redirect.
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # 1. Check Bearer token from React frontend
        auth_header = request.headers.get('Authorization', '')
        if auth_header.startswith('Bearer '):
            bearer_token = auth_header[7:]
            if bearer_token:
                # Temporarily set the token as __session for verification
                clerk_id, email, name = _verify_bearer_token(bearer_token)
                if clerk_id:
                    user = _get_or_create_local_user(clerk_id, email, name)
                    if user:
                        session['user_id'] = user['id']
                        session['user_name'] = user['name']
                        return f(*args, **kwargs)

        # 2. Check Clerk __session cookie
        clerk_id, email, name = _verify_clerk_session()
        if clerk_id:
            user = _get_or_create_local_user(clerk_id, email, name)
            if user:
                session['user_id'] = user['id']
                session['user_name'] = user['name']
                return f(*args, **kwargs)

        # 3. Fallback: check Flask session
        if 'user_id' not in session:
            # Return JSON 401 for API/AJAX requests
            if (request.headers.get('Accept', '').startswith('application/json') or
                request.headers.get('Content-Type', '').startswith('application/json') or
                request.headers.get('X-Requested-With') == 'XMLHttpRequest' or
                request.path.startswith('/api/') or
                request.path.startswith('/autocomplete') or
                request.path.startswith('/mood_recommend') or
                request.path.startswith('/wishlist') or
                request.path.startswith('/history') or
                request.path.startswith('/rate') or
                'fetch' in request.headers.get('Sec-Fetch-Mode', '')):
                return jsonify({'error': 'Authentication required'}), 401
            return redirect('/auth?redirect_url=' + request.path)
        return f(*args, **kwargs)
    return decorated_function


def admin_required(f):
    """
    Decorator to protect admin routes — requires 'admin' role.
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        user = get_current_user()
        if not user:
            return redirect('/auth?redirect_url=' + request.path)
        if user.get('role') != 'admin':
            from flask import abort
            abort(403)
        return f(*args, **kwargs)
    return decorated_function


# Legacy functions kept for backward compatibility
def login_user(user):
    """Store user in session (legacy support)."""
    session.permanent = True
    session['user_id'] = user['id']
    session['user_name'] = user['name']
    session['user_email'] = user.get('email', '')
    session['user_role'] = user.get('role', 'user')


def logout_user():
    """Clear the session."""
    session.clear()


def get_user_by_email(email):
    """Look up a user by email."""
    conn = get_db()
    user = conn.execute('SELECT * FROM users WHERE email = ?', (email,)).fetchone()
    conn.close()
    return dict(user) if user else None


def create_user(name, email, password=None, google_id=None, role='user'):
    """Create a new user (legacy support for migration)."""
    conn = get_db()
    try:
        import bcrypt
        pw_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt(rounds=12)).decode('utf-8') if password else None
        conn.execute(
            'INSERT INTO users (name, email, password_hash, google_id, role) VALUES (?, ?, ?, ?, ?)',
            (name, email, pw_hash, google_id, role)
        )
        conn.commit()
        user = conn.execute('SELECT * FROM users WHERE email = ?', (email,)).fetchone()
        conn.close()
        return dict(user), None
    except sqlite3.IntegrityError:
        conn.close()
        return None, 'An account with this email already exists.'


def verify_user(email, password):
    """Verify email/password (legacy support)."""
    conn = get_db()
    user = conn.execute('SELECT * FROM users WHERE email = ?', (email,)).fetchone()
    conn.close()
    if not user:
        return None, 'Invalid email or password.'
    user_dict = dict(user)
    if not user_dict.get('password_hash'):
        return None, 'Please sign in with Clerk.'
    try:
        import bcrypt
        if bcrypt.checkpw(password.encode('utf-8'), user_dict['password_hash'].encode('utf-8')):
            return user_dict, None
    except Exception:
        pass
    return None, 'Invalid email or password.'


def get_user_by_google_id(google_id):
    """Look up a user by Google ID."""
    conn = get_db()
    user = conn.execute('SELECT * FROM users WHERE google_id = ?', (google_id,)).fetchone()
    conn.close()
    return dict(user) if user else None


# ══════════════════════════════════════════════════════════════════
# ADMIN UTILITIES
# ══════════════════════════════════════════════════════════════════

def list_users(limit=100):
    """List all users (for admin dashboard)."""
    conn = get_db()
    rows = conn.execute(
        'SELECT id, name, email, role, clerk_id, created_at '
        'FROM users ORDER BY id DESC LIMIT ?',
        (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ══════════════════════════════════════════════════════════════════
# WISHLIST
# ══════════════════════════════════════════════════════════════════

def add_to_wishlist(user_id, book_title, book_author='', book_image=''):
    """Add a book to a user's wishlist."""
    conn = get_db()
    try:
        conn.execute(
            'INSERT OR IGNORE INTO wishlist (user_id, book_title, book_author, book_image) VALUES (?, ?, ?, ?)',
            (user_id, book_title, book_author, book_image)
        )
        conn.commit()
        conn.close()
        return True, None
    except Exception as e:
        conn.close()
        return False, str(e)


def remove_from_wishlist(user_id, book_title):
    """Remove a book from a user's wishlist."""
    conn = get_db()
    conn.execute('DELETE FROM wishlist WHERE user_id=? AND book_title=?', (user_id, book_title))
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
    """Add a book to a user's reading history."""
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
    """Return the user's reading history."""
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
    conn.execute('DELETE FROM reading_history WHERE user_id=? AND book_title=?', (user_id, book_title))
    conn.commit()
    conn.close()


def get_history_titles(user_id):
    """Return just the titles from reading history."""
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
    """Rate a book 1-5."""
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
    """Get a user's rating for a specific book."""
    conn = get_db()
    row = conn.execute(
        'SELECT rating FROM ratings WHERE user_id=? AND book_title=?',
        (user_id, book_title)
    ).fetchone()
    conn.close()
    return row['rating'] if row else None


def get_user_ratings(user_id, limit=100):
    """Return all books rated by a user."""
    conn = get_db()
    rows = conn.execute(
        'SELECT book_title, rating, rated_at '
        'FROM ratings WHERE user_id=? ORDER BY rated_at DESC LIMIT ?',
        (user_id, limit)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ══════════════════════════════════════════════════════════════════
# USER PREFERENCES (Onboarding)
# ══════════════════════════════════════════════════════════════════

def save_genre_preferences(user_id, genres):
    """Save a list of genre preferences for a user."""
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
