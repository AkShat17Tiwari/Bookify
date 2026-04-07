from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import pickle
import numpy as np
import os
import random
import json
import html as html_module
from difflib import get_close_matches
from cover_analyzer import analyze_cover

with open('popular.pkl', 'rb') as f:
    popular_df = pickle.load(f)
# Unescape HTML entities in book titles (e.g. '&amp;' -> '&')
popular_df['Book-Title'] = popular_df['Book-Title'].apply(lambda x: html_module.unescape(x) if isinstance(x, str) else x)
popular_df['Book-Author'] = popular_df['Book-Author'].apply(lambda x: html_module.unescape(x) if isinstance(x, str) else x)
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

# Load model accuracy metrics (if available)
model_accuracy = None
if os.path.exists('model_accuracy.json'):
    with open('model_accuracy.json', 'r') as f:
        model_accuracy = json.load(f)
    print(f"✅ Model accuracy loaded: {model_accuracy['accuracy']['accuracy_pct']}%")

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

# Build expanded searchable titles from pt + genre_map (1M+)
all_searchable_titles_set = set(all_titles)
if genre_available:
    all_searchable_titles_set.update(genre_map.keys())
all_searchable_titles_sorted = sorted(all_searchable_titles_set, key=str.lower)
all_searchable_titles_lower = [(t.lower(), t) for t in all_searchable_titles_sorted]
print(f"\u2705 Searchable titles pool: {len(all_searchable_titles_sorted)} books")

# Pre-compute genre index arrays for fast vectorized scoring
genre_index_cache = {}
if genre_available:
    for genre_name, titles in genre_books.items():
        indices = [title_to_index[t] for t in titles if t in title_to_index]
        genre_index_cache[genre_name] = np.array(indices, dtype=int)
    print(f"✅ Genre index cache built for {len(genre_index_cache)} genres")

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)

# SECURITY: Generate a cryptographically random secret key if not set in env.
import secrets as _secrets
app.secret_key = os.environ.get('SECRET_KEY') or _secrets.token_hex(32)

# ── Clerk configuration ──
CLERK_PUBLISHABLE_KEY = os.environ.get('CLERK_PUBLISHABLE_KEY', '')

# ── Security middleware (headers, rate limiting, CSRF, input validation) ──
from security import init_security
init_security(app)

# ── Auth system (Clerk + local DB) ──
from auth import (
    get_current_user, login_required, admin_required,
    login_user, logout_user, list_users,
    add_to_wishlist, remove_from_wishlist, get_wishlist, is_wishlisted,
    add_to_history, get_reading_history, remove_from_history, get_history_titles,
    rate_book, get_user_rating, get_user_ratings,
    save_genre_preferences, get_genre_preferences, has_completed_onboarding
)

# Make Clerk publishable key available to all templates
@app.context_processor
def inject_clerk():
    return dict(clerk_publishable_key=CLERK_PUBLISHABLE_KEY)


# ══════════════════════════════════════════════════════════════════
# AUTH ROUTES (Clerk)
# ══════════════════════════════════════════════════════════════════

@app.route('/auth')
def auth_page():
    """Render the Clerk sign-in page."""
    # If already authenticated, redirect to home
    user = get_current_user()
    if user:
        if not has_completed_onboarding(user['id']):
            return redirect('/onboarding')
        return redirect('/')
    redirect_url = request.args.get('redirect_url', '/')
    return render_template('auth.html', redirect_url=redirect_url)

@app.route('/login')
def login_page():
    """Legacy redirect — send to Clerk auth page."""
    next_url = request.args.get('next', '/')
    return redirect('/auth?redirect_url=' + next_url)

@app.route('/signup')
def signup_page():
    """Legacy redirect — send to Clerk auth page."""
    return redirect('/auth')

@app.route('/logout')
def logout():
    logout_user()
    return redirect('/auth')


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
    popular_titles = set(popular_df['Book-Title'].values)

    data = []
    for rank_num, rank_pos in enumerate(ranked[:count]):
        idx = genre_indices[rank_pos]
        title = pt.index[idx]
        if title in book_info_lookup:
            info = book_info_lookup[title]
            image = info['image']
            author = info.get('author', 'Unknown Author')

            # Build varied explanations
            reasons = []
            if extra_reasons:
                reasons.extend(extra_reasons)

            score_pct = centrality[rank_pos] / max_centrality if max_centrality > 0 else 0

            # 1) Rank-specific reason (varies by position)
            if rank_num == 0:
                reasons.append(f"🏆 #1 most representative {genre_name} book")
            elif rank_num <= 2:
                reasons.append(f"🥇 Top {rank_num + 1} in {genre_name} ({genre_total:,} books analyzed)")
            elif rank_num <= 4:
                reasons.append(f"📊 Top {rank_num + 1} by reader pattern analysis")
            else:
                reasons.append(f"📂 Ranked #{rank_num + 1} in {genre_name}")

            # 2) Author info
            if author and author != 'Unknown Author':
                reasons.append(f"✍️ By {author}")

            # 3) Cross-genre signal
            if genre_available:
                book_genres = genre_map.get(title, [])
                other_genres = [g for g in book_genres if g != genre_name and g != 'Fiction']
                if other_genres:
                    reasons.append(f"🔗 Also in: {', '.join(other_genres[:2])}")

            # 4) Popularity check
            if title in popular_titles:
                reasons.append("🔥 Popular among readers")

            # 5) Centrality/relevance score
            if score_pct > 0.85:
                reasons.append(f"💎 Exceptional fit ({score_pct:.0%} relevance)")
            elif score_pct > 0.7:
                reasons.append(f"🔥 Strong fit ({score_pct:.0%} relevance)")
            elif score_pct > 0.5:
                reasons.append(f"⭐ Good fit ({score_pct:.0%} relevance)")

            data.append([title, author, image, reasons])

    return data, genre_name


@app.route('/')
@login_required
def index():
    user = get_current_user()
    total_users_count = len(list_users(limit=10000))
    return render_template('index.html',
                           book_name = list(popular_df['Book-Title'].values),
                           author=list(popular_df['Book-Author'].values),
                           image=list(popular_df['Image-URL-M'].values),
                           votes=list(popular_df['num_ratings'].values),
                           rating=list(popular_df['avg_rating'].values),
                           user=user,
                           total_books=pt.shape[0],
                           total_genres=len(all_genres) if genre_available else 23,
                           total_users=pt.shape[1] + total_users_count,
                           model_accuracy=model_accuracy
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

    # Efficient prefix + substring search on 1M+ titles
    import bisect
    # Binary search for prefix matches
    lo = bisect.bisect_left(all_searchable_titles_lower, (query,))
    prefix_matches = []
    for idx in range(lo, min(lo + 20, len(all_searchable_titles_lower))):
        lower_t, orig_t = all_searchable_titles_lower[idx]
        if lower_t.startswith(query):
            prefix_matches.append(orig_t)
        else:
            break
    suggestions.extend(prefix_matches[:8])

    # Substring match on collaborative-filtering titles (5K — fast)
    if len(suggestions) < 8:
        substr = [t for t in all_titles if query in t.lower() and t not in suggestions][:8]
        suggestions.extend(substr)

    # If few matches, supplement with fuzzy matches on CF titles
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
    user = get_current_user()
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
                                   all_genres=all_genres, genre_count=len(genre_books.get(genre_name, [])),
                                   user=user)

    # Then check if the raw text is a genre keyword
    detected_genre = detect_genre(user_input)
    if detected_genre:
        data, matched_genre = get_genre_recommendations(detected_genre, mode)
        used_mode = 'ai' if (mode == 'ai' and ncf_available) else 'classic'
        return render_template('recommend.html', data=data, genre_mode=True,
                               matched_genre=matched_genre, mode=used_mode,
                               ncf_available=ncf_available, genre_available=genre_available,
                               all_genres=all_genres, genre_count=len(genre_books.get(detected_genre, [])),
                               user=user)

    # ── Book title search logic ──
    matches = np.where(pt.index == user_input)[0]

    if len(matches) == 0:
        close = get_close_matches(user_input, all_titles, n=1, cutoff=0.4)
        if close:
            user_input = close[0]
            matches = np.where(pt.index == user_input)[0]

    if len(matches) == 0:
        # Title not in collaborative-filtering set — try genre-based recommendations
        if genre_available and user_input in genre_map:
            book_genres = genre_map[user_input]
            if book_genres:
                primary_genre = book_genres[0]
                data, matched_genre = get_genre_recommendations(primary_genre, mode, count=8,
                    extra_reasons=[f'📚 Related to "{user_input}"'])
                used_mode = 'ai' if (mode == 'ai' and ncf_available) else 'classic'
                return render_template('recommend.html', data=data, genre_mode=True,
                                       matched_genre=f'Books like "{user_input}" ({matched_genre})', mode=used_mode,
                                       ncf_available=ncf_available, genre_available=genre_available,
                                       all_genres=all_genres, genre_count=len(genre_books.get(primary_genre, [])),
                                       user=user)
        return render_template('recommend.html', data=[], error=f'No book found matching "{request.form.get("user_input")}". Try a different title or search by genre.',
                               ncf_available=ncf_available, genre_available=genre_available, all_genres=all_genres,
                               user=user)

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
                           ncf_available=ncf_available, genre_available=genre_available, all_genres=all_genres,
                           user=user)


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
    """GET: return reading history genres. POST: add a book to history (DB-backed)."""
    user = get_current_user()
    if request.method == 'POST':
        data = request.get_json()
        title = data.get('title', '').strip()
        author = data.get('author', '').strip()
        image = data.get('image', '').strip()
        if title:
            add_to_history(user['id'], title, author, image)
            genres = genre_map.get(title, []) if genre_available else []
            return jsonify({'status': 'added', 'title': title, 'genres': genres})
        return jsonify({'status': 'error', 'message': 'No title provided'})

    # GET: return history with genre info
    history_rows = get_reading_history(user['id'])
    history_data = []
    for row in history_rows:
        genres = genre_map.get(row['book_title'], []) if genre_available else []
        history_data.append({'title': row['book_title'], 'genres': genres})
    return jsonify({'history': history_data})


@app.route('/history/remove', methods=['POST'])
@login_required
def history_remove():
    """Remove a book from reading history."""
    user = get_current_user()
    data = request.get_json()
    title = data.get('title', '').strip()
    if not title:
        return jsonify({'status': 'error', 'message': 'No title provided'})
    remove_from_history(user['id'], title)
    return jsonify({'status': 'removed', 'title': title})


# ══════════════════════════════════════════════════════════════════
# WISHLIST ROUTES
# ══════════════════════════════════════════════════════════════════

@app.route('/wishlist', methods=['GET'])
@login_required
def wishlist_page():
    user = get_current_user()
    books = get_wishlist(user['id'])
    return jsonify({'wishlist': books})


@app.route('/wishlist/add', methods=['POST'])
@login_required
def wishlist_add():
    user = get_current_user()
    data = request.get_json()
    title = data.get('title', '').strip()
    author = data.get('author', '').strip()
    image = data.get('image', '').strip()
    if not title:
        return jsonify({'status': 'error', 'message': 'No title provided'})
    ok, err = add_to_wishlist(user['id'], title, author, image)
    if ok:
        return jsonify({'status': 'added', 'title': title})
    return jsonify({'status': 'error', 'message': err})


@app.route('/wishlist/remove', methods=['POST'])
@login_required
def wishlist_remove():
    user = get_current_user()
    data = request.get_json()
    title = data.get('title', '').strip()
    if not title:
        return jsonify({'status': 'error', 'message': 'No title provided'})
    remove_from_wishlist(user['id'], title)
    return jsonify({'status': 'removed', 'title': title})


@app.route('/wishlist/check')
@login_required
def wishlist_check():
    user = get_current_user()
    title = request.args.get('title', '').strip()
    return jsonify({'wishlisted': is_wishlisted(user['id'], title)})


# ══════════════════════════════════════════════════════════════════
# PROFILE & FOR YOU ROUTES
# ══════════════════════════════════════════════════════════════════

@app.route('/profile')
@login_required
def profile_page():
    user = get_current_user()
    wishlist_books = get_wishlist(user['id'])
    history_rows = get_reading_history(user['id'])
    history_data = []
    for row in history_rows:
        genres = genre_map.get(row['book_title'], []) if genre_available else []
        history_data.append({
            'title': row['book_title'],
            'genres': genres,
            'author': row.get('book_author', 'Unknown Author'),
            'image': row.get('book_image', '')
        })
    user_ratings = get_user_ratings(user['id'])
    # Enrich ratings with book info (image, author)
    for r in user_ratings:
        info = book_info_lookup.get(r['book_title'], {})
        r['author'] = info.get('author', 'Unknown Author')
        r['image'] = info.get('image', '')
    return render_template('profile.html',
                           user=user,
                           wishlist=wishlist_books,
                           history=history_data,
                           ratings=user_ratings)


@app.route('/for_you')
@login_required
def for_you_page():
    user = get_current_user()
    hist = get_history_titles(user['id'])

    # Aggregate genres from reading history
    genre_votes = {}
    for title in hist:
        book_genres = genre_map.get(title, []) if genre_available else []
        for g in book_genres:
            genre_votes[g] = genre_votes.get(g, 0) + 1

    # Fallback to onboarding preferences if no reading history
    if not genre_votes:
        prefs = get_genre_preferences(user['id'])
        if prefs:
            for g in prefs:
                genre_votes[g] = genre_votes.get(g, 0) + 1

    if not genre_votes:
        return render_template('for_you.html', user=user, books=[], inferred_genres=[], has_history=False)

    sorted_genres = sorted(genre_votes.items(), key=lambda x: x[1], reverse=True)
    top_genres = [g for g, _ in sorted_genres[:4]]

    all_books = []
    seen = set()
    for gname in top_genres:
        reason = '🏗️ Based on your reading history' if hist else '✨ Based on your genre preferences'
        recs, _ = get_genre_recommendations(gname, 'classic', count=6,
                                             extra_reasons=[reason])
        for book in recs:
            if book[0] not in seen:
                seen.add(book[0])
                all_books.append(book)

    return render_template('for_you.html', user=user, books=all_books[:12],
                           inferred_genres=top_genres, has_history=True)


# ══════════════════════════════════════════════════════════════════
# RATING ROUTES
# ══════════════════════════════════════════════════════════════════

@app.route('/rate', methods=['POST'])
@login_required
def rate_route():
    """Rate a book 1-5 stars."""
    user = get_current_user()
    data = request.get_json()
    title = data.get('title', '').strip()
    rating = data.get('rating')
    if not title or rating is None:
        return jsonify({'status': 'error', 'message': 'Title and rating required'})
    try:
        rating = int(rating)
    except (ValueError, TypeError):
        return jsonify({'status': 'error', 'message': 'Rating must be a number'})
    ok, err = rate_book(user['id'], title, rating)
    if ok:
        return jsonify({'status': 'rated', 'title': title, 'rating': rating})
    return jsonify({'status': 'error', 'message': err})


@app.route('/rating/check')
@login_required
def rating_check():
    """Get the current user's rating for a book."""
    user = get_current_user()
    title = request.args.get('title', '').strip()
    rating = get_user_rating(user['id'], title)
    return jsonify({'rating': rating})

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
# ONBOARDING
# ══════════════════════════════════════════════════════════════════

@app.route('/onboarding', methods=['GET', 'POST'])
@login_required
def onboarding_page():
    """Genre quiz for new users. GET shows the picker, POST saves preferences."""
    user = get_current_user()

    if request.method == 'POST':
        data = request.get_json()
        genres = data.get('genres', [])
        if not genres or len(genres) < 3:
            return jsonify({'status': 'error', 'message': 'Please pick at least 3 genres'})
        # Validate genres
        valid = [g for g in genres if g in genre_books] if genre_available else []
        if len(valid) < 3:
            return jsonify({'status': 'error', 'message': 'Please pick at least 3 valid genres'})
        ok, err = save_genre_preferences(user['id'], valid)
        if ok:
            return jsonify({'status': 'saved', 'redirect': '/'})
        return jsonify({'status': 'error', 'message': err})

    # GET — if already completed, redirect home
    if has_completed_onboarding(user['id']):
        return redirect('/')

    return render_template('onboarding.html', user=user,
                           all_genres=all_genres if genre_available else [])


# ══════════════════════════════════════════════════════════════════
# BOOK DETAILS (Open Library API Proxy)
# ══════════════════════════════════════════════════════════════════

@app.route('/book/details')
@login_required
def book_details():
    """Fetch book description from Open Library Search API."""
    import requests as http_req
    title = request.args.get('title', '').strip()
    if not title:
        return jsonify({'error': 'No title provided'})

    try:
        resp = http_req.get(
            'https://openlibrary.org/search.json',
            params={'title': title, 'limit': 1, 'fields': 'first_sentence,first_publish_year,number_of_pages_median,subject,isbn'},
            timeout=5
        )
        data = resp.json()
        docs = data.get('docs', [])
        if not docs:
            return jsonify({'description': None, 'year': None, 'pages': None, 'subjects': []})

        doc = docs[0]
        # Get description from first_sentence
        first_sentence = doc.get('first_sentence', [])
        description = first_sentence[0] if first_sentence else None

        subjects = doc.get('subject', [])[:5]
        year = doc.get('first_publish_year')
        pages = doc.get('number_of_pages_median')

        return jsonify({
            'description': description,
            'year': year,
            'pages': pages,
            'subjects': subjects
        })
    except Exception:
        return jsonify({'description': None, 'year': None, 'pages': None, 'subjects': []})


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
    return render_template('admin.html', user=user, users=users, events=events)


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


# ── Favicon route to prevent 404 errors ──
@app.route('/favicon.ico')
def favicon():
    return '', 204


if __name__ == '__main__':
    app.run(debug=True, port=5001)