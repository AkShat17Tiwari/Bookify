from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import pickle
import numpy as np
import os
import random
import json
from difflib import get_close_matches
from cover_analyzer import analyze_cover

with open('popular.pkl', 'rb') as f:
    popular_df = pickle.load(f)
with open('pt.pkl', 'rb') as f:
    pt = pickle.load(f)
with open('books_slim.pkl', 'rb') as f:
    book_info_lookup = pickle.load(f)
with open('similarity_scores.pkl', 'rb') as f:
    similarity_scores = pickle.load(f)

# Load NCF deep learning similarity scores (if available)
ncf_available = os.path.exists('ncf_similarity_scores.pkl')
if ncf_available:
    with open('ncf_similarity_scores.pkl', 'rb') as f:
        ncf_similarity_scores = pickle.load(f)
    print("✅ NCF deep learning model loaded!")
else:
    ncf_similarity_scores = None
    print("⚠️  NCF model not found — using classic mode only")

# Load genre data (if available)
genre_data = None
genre_available = os.path.exists('genre_data.pkl')
if genre_available:
    with open('genre_data.pkl', 'rb') as f:
        genre_data = pickle.load(f)
    genre_books = genre_data['genre_books']   # genre → [titles]
    genre_map = genre_data['genre_map']       # title → [genres]
    all_genres = sorted(genre_books.keys())
    # Build lowercase lookup for genre matching
    genre_lookup = {g.lower(): g for g in all_genres}
    # Also support partial/alias matching
    genre_aliases = {
        'sci-fi': 'Science Fiction', 'scifi': 'Science Fiction', 'science fiction': 'Science Fiction',
        'sf': 'Science Fiction',
        'mystery': 'Mystery/Thriller', 'thriller': 'Mystery/Thriller', 'crime': 'Mystery/Thriller',
        'suspense': 'Mystery/Thriller', 'detective': 'Mystery/Thriller',
        'romance': 'Romance', 'love story': 'Romance', 'love stories': 'Romance',
        'horror': 'Horror', 'scary': 'Horror', 'ghost stories': 'Horror',
        'fantasy': 'Fantasy', 'magic': 'Fantasy',
        'literary fiction': 'Literary Fiction', 'literary': 'Literary Fiction', 'literature': 'Literary Fiction',
        'non-fiction': 'Non-Fiction', 'nonfiction': 'Non-Fiction', 'non fiction': 'Non-Fiction',
        'history': 'History', 'historical': 'History',
        'biography': 'Biography', 'memoir': 'Biography', 'bio': 'Biography',
        'self-help': 'Self-Help', 'self help': 'Self-Help', 'selfhelp': 'Self-Help',
        'cooking': 'Cooking', 'cookbook': 'Cooking', 'recipes': 'Cooking', 'food': 'Cooking',
        'travel': 'Travel', 'adventure': 'Travel', 'travel guide': 'Travel',
        'young adult': 'Young Adult', 'ya': 'Young Adult', 'teen': 'Young Adult',
        'children': 'Children', 'kids': 'Children', "children's": 'Children',
        'classics': 'Classics', 'classic': 'Classics', 'classic literature': 'Classics',
        'poetry': 'Poetry', 'poems': 'Poetry',
        'fiction': 'Fiction', 'general fiction': 'Fiction',
        'religious': 'Religious/Spiritual', 'spiritual': 'Religious/Spiritual',
        'religion': 'Religious/Spiritual', 'faith': 'Religious/Spiritual',
    }
    print(f"✅ Genre data loaded: {len(all_genres)} genres, {len(genre_map)} classified books")
else:
    genre_books = {}
    genre_map = {}
    all_genres = []
    genre_lookup = {}
    genre_aliases = {}
    print("⚠️  Genre data not found — genre search disabled")

# Pre-compute list of all book titles for fuzzy matching
all_titles = list(pt.index)
title_to_index = {title: i for i, title in enumerate(pt.index)}

# Pre-compute genre index arrays for fast vectorized scoring
genre_index_cache = {}
if genre_available:
    for genre_name, titles in genre_books.items():
        indices = [title_to_index[t] for t in titles if t in title_to_index]
        genre_index_cache[genre_name] = np.array(indices, dtype=int)
    print(f"✅ Genre index cache built for {len(genre_index_cache)} genres")

app = Flask(__name__)

# SECURITY: Generate a cryptographically random secret key if not set in env.
# Never use a hardcoded default in production.
import secrets as _secrets
app.secret_key = os.environ.get('SECRET_KEY') or _secrets.token_hex(32)

# ── Security middleware (headers, rate limiting, CSRF, input validation) ──
from security import init_security
init_security(app)

# ── Auth system (bcrypt, RBAC, brute-force protection) ──
from auth import (
    create_user, verify_user, get_user_by_email, get_user_by_google_id,
    login_user, logout_user, get_current_user, login_required, admin_required,
    list_users
)

# In-memory reading history (per-session fallback; primary storage is localStorage on client)
reading_history = {}


# ══════════════════════════════════════════════════════════════════
# AUTH ROUTES
# ══════════════════════════════════════════════════════════════════

@app.route('/login', methods=['GET', 'POST'])
def login_page():
    if 'user_id' in session:
        return redirect('/')
    if request.method == 'POST':
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        next_url = request.form.get('next', '/')
        user, error = verify_user(email, password)
        if error:
            return render_template('login.html', error=error, next_url=next_url, google_client_id=GOOGLE_CLIENT_ID)
        login_user(user)
        return redirect(next_url or '/')
    next_url = request.args.get('next', '/')
    return render_template('login.html', next_url=next_url, google_client_id=GOOGLE_CLIENT_ID)


@app.route('/signup', methods=['GET', 'POST'])
def signup_page():
    if 'user_id' in session:
        return redirect('/')
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        confirm = request.form.get('confirm_password', '')
        if not name or not email or not password:
            return render_template('signup.html', error='All fields are required.', google_client_id=GOOGLE_CLIENT_ID)
        if len(password) < 6:
            return render_template('signup.html', error='Password must be at least 6 characters.', google_client_id=GOOGLE_CLIENT_ID)
        if password != confirm:
            return render_template('signup.html', error='Passwords do not match.', google_client_id=GOOGLE_CLIENT_ID)
        user, error = create_user(name, email, password)
        if error:
            return render_template('signup.html', error=error, google_client_id=GOOGLE_CLIENT_ID)
        login_user(user)
        return redirect('/')
    return render_template('signup.html', google_client_id=GOOGLE_CLIENT_ID)


@app.route('/logout')
def logout():
    logout_user()
    return redirect('/login')


GOOGLE_CLIENT_ID = os.environ.get('GOOGLE_CLIENT_ID', '')

@app.route('/google/login')
def google_login():
    """Redirect to login with a note if Google is not configured."""
    if not GOOGLE_CLIENT_ID:
        return render_template('login.html',
            error='Google Sign-In is not configured. Set GOOGLE_CLIENT_ID in your environment.', google_client_id=GOOGLE_CLIENT_ID)
    # The actual Google sign-in is handled client-side via GIS JS library
    return redirect('/login')


@app.route('/google/callback', methods=['POST'])
def google_callback():
    """Verify Google ID token and log the user in."""
    import requests as http_requests
    token = request.form.get('credential', '')
    if not token:
        return redirect('/login')

    # Verify token with Google
    try:
        resp = http_requests.get(
            'https://oauth2.googleapis.com/tokeninfo',
            params={'id_token': token},
            timeout=5
        )
        if resp.status_code != 200:
            return render_template('login.html', error='Google authentication failed. Please try again.', google_client_id=GOOGLE_CLIENT_ID)

        info = resp.json()

        # Verify the token is for our app
        if info.get('aud') != GOOGLE_CLIENT_ID:
            return render_template('login.html', error='Invalid Google token.', google_client_id=GOOGLE_CLIENT_ID)

        email = info.get('email', '')
        name = info.get('name', info.get('given_name', email.split('@')[0]))
        google_id = info.get('sub', '')

        if not email:
            return render_template('login.html', error='Could not retrieve email from Google.', google_client_id=GOOGLE_CLIENT_ID)

        # Check if user exists
        existing = get_user_by_email(email)
        if existing:
            login_user(existing)
            return redirect('/')

        # Create new user via Google
        user, err = create_user(name, email, google_id=google_id)
        if err:
            # Account exists but was created with email/password
            existing = get_user_by_email(email)
            if existing:
                login_user(existing)
                return redirect('/')
            return render_template('login.html', error=err, google_client_id=GOOGLE_CLIENT_ID)

        login_user(user)
        return redirect('/')
    except Exception as e:
        print(f"Google OAuth error: {e}")
        return render_template('login.html', error='Google authentication failed. Please try again.', google_client_id=GOOGLE_CLIENT_ID)


def detect_genre(query):
    """Check if the user is searching for a genre rather than a specific book title."""
    q = query.strip().lower()

    # Direct match
    if q in genre_aliases:
        return genre_aliases[q]
    if q in genre_lookup:
        return genre_lookup[q]

    # Partial match
    for alias, genre in genre_aliases.items():
        if q == alias or q.startswith(alias + ' ') or q.endswith(' ' + alias):
            return genre

    return None


def get_genre_recommendations(genre_name, mode='classic', count=8, extra_reasons=None):
    """Get top books from a genre, ranked by genre centrality (vectorized).
    Returns (data, genre_name) where data items are [title, author, image, explanations].
    """
    if genre_name not in genre_index_cache:
        return [], genre_name

    genre_indices = genre_index_cache[genre_name]
    if len(genre_indices) == 0:
        return [], genre_name

    # Choose similarity matrix
    if mode == 'ai' and ncf_available:
        sim = ncf_similarity_scores
    else:
        sim = similarity_scores
    
    # Vectorized: get the submatrix for genre books and compute mean similarity
    sim_sub = sim[np.ix_(genre_indices, genre_indices)].copy()
    np.fill_diagonal(sim_sub, 0)
    n = len(genre_indices)
    centrality = sim_sub.sum(axis=1) / max(n - 1, 1)
    max_centrality = centrality.max() if centrality.max() > 0 else 1

    # Sort by centrality descending
    ranked = np.argsort(centrality)[::-1]

    genre_total = len(genre_books.get(genre_name, []))

    data = []
    for rank_num, rank_pos in enumerate(ranked[:count]):
        idx = genre_indices[rank_pos]
        title = pt.index[idx]
        if title in book_info_lookup:
            info = book_info_lookup[title]
            image = info['image']

            # Build explanations
            reasons = []
            if extra_reasons:
                reasons.extend(extra_reasons)
            reasons.append(f"📂 Top-ranked in {genre_name} ({genre_total:,} books)")
            score_pct = centrality[rank_pos] / max_centrality if max_centrality > 0 else 0
            if score_pct > 0.7:
                reasons.append(f"🔥 High relevance score ({score_pct:.0%})")
            elif score_pct > 0.4:
                reasons.append(f"⭐ Good relevance score ({score_pct:.0%})")

            data.append([title, info['author'], image, reasons])

    return data, genre_name


@app.route('/')
@login_required
def index():
    user = get_current_user()
    return render_template('index.html',
                           book_name = list(popular_df['Book-Title'].values),
                           author=list(popular_df['Book-Author'].values),
                           image=list(popular_df['Image-URL-M'].values),
                           votes=list(popular_df['num_ratings'].values),
                           rating=list(popular_df['avg_rating'].values),
                           user=user
                           )

@app.route('/recommend')
@login_required
def recommend_ui():
    user = get_current_user()
    return render_template('recommend.html', ncf_available=ncf_available,
                           genre_available=genre_available, all_genres=all_genres, user=user)

# ── Emotion → Genre mapping ──
EMOTION_GENRES = {
    'happy':     ['Romance', 'Travel', 'Cooking'],
    'sad':       ['Self-Help', 'Poetry', 'Religious/Spiritual'],
    'angry':     ['Mystery/Thriller', 'Horror'],
    'fearful':   ['Self-Help', 'Fantasy', 'Children'],
    'disgusted': ['Science Fiction', 'Fantasy'],
    'surprised': ['Mystery/Thriller', 'Science Fiction'],
    'neutral':   ['Literary Fiction', 'Classics', 'Non-Fiction'],
}

@app.route('/mood')
@login_required
def mood_ui():
    user = get_current_user()
    return render_template('mood.html', ncf_available=ncf_available, user=user)

@app.route('/mood_recommend', methods=['POST'])
@login_required
def mood_recommend():
    data = request.get_json()
    emotion = data.get('emotion', 'neutral').lower().strip()
    mode = data.get('mode', 'classic')

    genres = EMOTION_GENRES.get(emotion, EMOTION_GENRES['neutral'])
    
    # Gather recommendations from each mapped genre with XAI
    all_data = []
    seen = set()
    for genre_name in genres:
        mood_reason = f"😊 Recommended for your {emotion} mood"
        recs, _ = get_genre_recommendations(genre_name, mode, count=4, extra_reasons=[mood_reason])
        for book in recs:
            if book[0] not in seen:
                seen.add(book[0])
                all_data.append(book)

    return jsonify({
        'emotion': emotion,
        'genres': genres,
        'books': all_data[:12],
    })

@app.route('/autocomplete')
@login_required
def autocomplete():
    """Return matching book titles and genres for the live search dropdown."""
    query = request.args.get('q', '').strip().lower()
    if len(query) < 2:
        return jsonify([])

    suggestions = []

    # Check for genre matches first
    if genre_available:
        for alias, genre in genre_aliases.items():
            if query in alias or alias.startswith(query):
                label = f"📂 Genre: {genre}"
                if label not in suggestions:
                    suggestions.append(label)
        # Limit genre suggestions
        suggestions = suggestions[:3]

    # Substring match for book titles
    title_matches = [t for t in all_titles if query in t.lower()][:10]
    suggestions.extend(title_matches)

    # If few matches, supplement with fuzzy matches
    if len(suggestions) < 5:
        fuzzy = get_close_matches(query, all_titles, n=10, cutoff=0.4)
        for f in fuzzy:
            if f not in suggestions:
                suggestions.append(f)
            if len(suggestions) >= 12:
                break

    return jsonify(suggestions[:12])

@app.route('/recommend_books', methods=['POST'])
@login_required
def recommend():
    user_input = request.form.get('user_input', '').strip()
    mode = request.form.get('mode', 'classic')  # 'classic' or 'ai'

    # Check if user typed a genre
    if user_input.startswith('📂 Genre: '):
        genre_name = user_input.replace('📂 Genre: ', '').strip()
        if genre_name in genre_books:
            data, matched_genre = get_genre_recommendations(genre_name, mode)
            used_mode = 'ai' if (mode == 'ai' and ncf_available) else 'classic'
            return render_template('recommend.html', data=data, genre_mode=True,
                                   matched_genre=matched_genre, mode=used_mode,
                                   ncf_available=ncf_available, genre_available=genre_available,
                                   all_genres=all_genres, genre_count=len(genre_books.get(genre_name, [])))

    # Then check if the raw text is a genre keyword
    detected_genre = detect_genre(user_input)
    if detected_genre:
        data, matched_genre = get_genre_recommendations(detected_genre, mode)
        used_mode = 'ai' if (mode == 'ai' and ncf_available) else 'classic'
        return render_template('recommend.html', data=data, genre_mode=True,
                               matched_genre=matched_genre, mode=used_mode,
                               ncf_available=ncf_available, genre_available=genre_available,
                               all_genres=all_genres, genre_count=len(genre_books.get(detected_genre, [])))

    # ── Book title search logic ──
    matches = np.where(pt.index == user_input)[0]

    if len(matches) == 0:
        close = get_close_matches(user_input, all_titles, n=1, cutoff=0.4)
        if close:
            user_input = close[0]
            matches = np.where(pt.index == user_input)[0]

    if len(matches) == 0:
        return render_template('recommend.html', data=[], error=f'No book found matching "{request.form.get("user_input")}". Try a different title or search by genre.',
                               ncf_available=ncf_available, genre_available=genre_available, all_genres=all_genres)

    index = matches[0]
    input_genres = genre_map.get(user_input, []) if genre_available else []
    input_author = book_info_lookup.get(user_input, {}).get('author', '') if user_input in book_info_lookup else ''

    # Choose similarity source based on mode
    if mode == 'ai' and ncf_available:
        scores = ncf_similarity_scores
        used_mode = 'ai'
    else:
        scores = similarity_scores
        used_mode = 'classic'

    similar_items = sorted(list(enumerate(scores[index])), key=lambda x: x[1], reverse=True)[1:5]

    data = []
    for i in similar_items:
        title = pt.index[i[0]]
        sim_score = float(i[1])
        if title in book_info_lookup:
            info = book_info_lookup[title]
            image = info['image']

            # Build XAI explanations
            reasons = []
            reasons.append(f"📖 Similar to {user_input} ({sim_score:.0%} match)")

            # Shared genres
            rec_genres = genre_map.get(title, []) if genre_available else []
            shared = set(input_genres) & set(rec_genres)
            shared.discard('Fiction')  # skip generic
            if shared:
                reasons.append(f"📂 Shares genre: {', '.join(sorted(shared))}")

            # Same author
            if input_author and info.get('author', '') == input_author:
                reasons.append(f"✍️ Same author: {input_author}")

            # Popularity signal
            if title in popular_df['Book-Title'].values:
                reasons.append("🔥 Popular among readers")

            data.append([title, info['author'], image, reasons])

    return render_template('recommend.html', data=data, matched_title=user_input, mode=used_mode,
                           ncf_available=ncf_available, genre_available=genre_available, all_genres=all_genres)


# ══════════════════════════════════════════════════════════════════
# MULTI-MODAL RECOMMENDATION SYSTEM
# ══════════════════════════════════════════════════════════════════

@app.route('/multimodal')
@login_required
def multimodal_ui():
    user = get_current_user()
    return render_template('multimodal.html', ncf_available=ncf_available,
                           genre_available=genre_available, all_genres=all_genres, user=user)


@app.route('/analyze_cover', methods=['POST'])
@login_required
def analyze_cover_route():
    """Analyze a book cover image and return genre predictions."""
    data = request.get_json()
    image_url = data.get('image_url', '').strip()
    if not image_url:
        return jsonify({'error': 'No image URL provided', 'genres': [], 'palette': []})

    result = analyze_cover(image_url)
    return jsonify(result)


@app.route('/voice_search', methods=['POST'])
@login_required
def voice_search():
    """Process speech-to-text input and return book/genre matches."""
    data = request.get_json()
    text = data.get('text', '').strip()
    if not text:
        return jsonify({'matches': [], 'genre': None})

    # Check for genre
    detected = detect_genre(text)
    if detected:
        return jsonify({'matches': [], 'genre': detected})

    # Fuzzy match book titles
    close = get_close_matches(text, all_titles, n=5, cutoff=0.35)
    # Also try substring match
    substr = [t for t in all_titles if text.lower() in t.lower()][:5]
    combined = list(dict.fromkeys(close + substr))[:8]  # deduplicate, keep order

    return jsonify({'matches': combined, 'genre': None})


@app.route('/history', methods=['GET', 'POST'])
@login_required
def history_route():
    """GET: return reading history genres. POST: add a book to history."""
    if request.method == 'POST':
        data = request.get_json()
        title = data.get('title', '').strip()
        if title:
            hist = session.get('history', [])
            if title not in hist:
                hist.append(title)
                hist = hist[-20:]  # keep last 20
                session['history'] = hist
            # Return genres for this title
            genres = genre_map.get(title, []) if genre_available else []
            return jsonify({'status': 'added', 'title': title, 'genres': genres})
        return jsonify({'status': 'error', 'message': 'No title provided'})

    # GET: return history with genre info
    hist = session.get('history', [])
    history_data = []
    for title in hist:
        genres = genre_map.get(title, []) if genre_available else []
        history_data.append({'title': title, 'genres': genres})
    return jsonify({'history': history_data})


@app.route('/multimodal_recommend', methods=['POST'])
@login_required
def multimodal_recommend():
    """
    Multi-modal fusion endpoint.
    Accepts JSON with optional keys: text, image_genres, emotion, voice_text, history_genres
    Each modality contributes genre votes; results are fused and deduplicated.
    """
    data = request.get_json()
    mode = data.get('mode', 'classic')

    genre_votes = {}  # genre → total score
    active_modalities = 0

    # ── 1. Text modality ──
    text_input = data.get('text', '').strip()
    if text_input:
        active_modalities += 1
        detected = detect_genre(text_input)
        if detected:
            genre_votes[detected] = genre_votes.get(detected, 0) + 1.0
        else:
            # Find book, get its genres
            matches = np.where(pt.index == text_input)[0]
            if len(matches) == 0:
                close = get_close_matches(text_input, all_titles, n=1, cutoff=0.4)
                if close:
                    text_input = close[0]
                    matches = np.where(pt.index == text_input)[0]
            if len(matches) > 0 and genre_available:
                book_genres = genre_map.get(text_input, [])
                for g in book_genres:
                    genre_votes[g] = genre_votes.get(g, 0) + 0.8

    # ── 2. Image modality (cover analysis results) ──
    image_genres = data.get('image_genres', [])
    if image_genres:
        active_modalities += 1
        for item in image_genres:
            g = item[0] if isinstance(item, (list, tuple)) else item
            s = item[1] if isinstance(item, (list, tuple)) and len(item) > 1 else 0.7
            if g in genre_books or g in genre_lookup.values() or g in [v for v in genre_aliases.values()]:
                genre_votes[g] = genre_votes.get(g, 0) + float(s)

    # ── 3. Voice modality (treated same as text) ──
    voice_text = data.get('voice_text', '').strip()
    if voice_text:
        active_modalities += 1
        detected = detect_genre(voice_text)
        if detected:
            genre_votes[detected] = genre_votes.get(detected, 0) + 1.0
        else:
            close = get_close_matches(voice_text, all_titles, n=1, cutoff=0.4)
            if close and genre_available:
                book_genres = genre_map.get(close[0], [])
                for g in book_genres:
                    genre_votes[g] = genre_votes.get(g, 0) + 0.8

    # ── 4. Emotion modality ──
    emotion = data.get('emotion', '').strip().lower()
    if emotion and emotion in EMOTION_GENRES:
        active_modalities += 1
        for g in EMOTION_GENRES[emotion]:
            genre_votes[g] = genre_votes.get(g, 0) + 0.9

    # ── 5. History modality ──
    history_genres = data.get('history_genres', [])
    if history_genres:
        active_modalities += 1
        for g in history_genres:
            genre_votes[g] = genre_votes.get(g, 0) + 0.5

    # ── Fusion: get recommendations from top-scoring genres ──
    if not genre_votes:
        return jsonify({'books': [], 'genres_used': [], 'modalities': 0,
                        'error': 'No input provided. Enable at least one modality.'})

    # Sort genres by vote score
    sorted_genres = sorted(genre_votes.items(), key=lambda x: x[1], reverse=True)
    top_genres = [g for g, s in sorted_genres[:4]]  # top 4 genres

    # Build modality explanation
    modality_names = []
    if text_input: modality_names.append('text search')
    if image_genres: modality_names.append('cover analysis')
    if voice_text: modality_names.append('voice input')
    if emotion: modality_names.append('emotion detection')
    if history_genres: modality_names.append('reading history')
    modality_reason = f"🔗 Matched via {' + '.join(modality_names)}" if modality_names else ''

    all_books = []
    seen_titles = set()
    for gname in top_genres:
        extra = [modality_reason] if modality_reason else None
        recs, _ = get_genre_recommendations(gname, mode, count=6, extra_reasons=extra)
        for book in recs:
            if book[0] not in seen_titles:
                seen_titles.add(book[0])
                all_books.append(book)

    return jsonify({
        'books': all_books[:12],
        'genres_used': top_genres,
        'genre_scores': dict(sorted_genres[:8]),
        'modalities': active_modalities,
    })

# ══════════════════════════════════════════════════════════════════
# ADMIN ROUTES  (RBAC protected)
# ══════════════════════════════════════════════════════════════════

@app.route('/admin')
@admin_required
def admin_dashboard():
    """Admin dashboard — only accessible by users with 'admin' role."""
    from audit_log import get_recent_events
    user = get_current_user()
    users = list_users(limit=50)
    events = get_recent_events(limit=30)
    return render_template('index.html', user=user,
                           books=popular_df[popular_df.columns[0:1]].values.tolist())


@app.route('/admin/users')
@admin_required
def admin_users_api():
    """API: List all users (admin only)."""
    users = list_users(limit=100)
    # SECURITY: Never expose password hashes in API responses
    return jsonify(users)


@app.route('/admin/audit')
@admin_required
def admin_audit_api():
    """API: Get recent security audit events (admin only)."""
    from audit_log import get_recent_events
    event_type = request.args.get('type')
    events = get_recent_events(event_type=event_type, limit=100)
    return jsonify(events)


if __name__ == '__main__':
    app.run(debug=True, port=5001)