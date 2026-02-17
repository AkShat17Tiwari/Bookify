from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import os
import random
from difflib import get_close_matches

with open('popular.pkl', 'rb') as f:
    popular_df = pickle.load(f)
with open('pt.pkl', 'rb') as f:
    pt = pickle.load(f)
with open('books.pkl', 'rb') as f:
    books = pickle.load(f)
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

# Pre-compute book info lookup for fast access (vectorized, no per-book DataFrame filter)
title_to_index = {title: i for i, title in enumerate(pt.index)}
_pt_titles = set(pt.index)
_lookup_df = books[books['Book-Title'].isin(_pt_titles)].drop_duplicates('Book-Title')
_lookup_df = _lookup_df.set_index('Book-Title')
book_info_lookup = {}
for title in all_titles:
    if title in _lookup_df.index:
        row = _lookup_df.loc[title]
        book_info_lookup[title] = {
            'title': title,
            'author': row['Book-Author'],
            'image': row['Image-URL-M'],
        }

# Pre-compute genre index arrays for fast vectorized scoring
genre_index_cache = {}
if genre_available:
    for genre_name, titles in genre_books.items():
        indices = [title_to_index[t] for t in titles if t in title_to_index]
        genre_index_cache[genre_name] = np.array(indices, dtype=int)
    print(f"✅ Genre index cache built for {len(genre_index_cache)} genres")

app = Flask(__name__)


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


def get_genre_recommendations(genre_name, mode='classic', count=8):
    """Get top books from a genre, ranked by genre centrality (vectorized)."""
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
    # sim_sub[i, j] = similarity between genre_book_i and genre_book_j
    sim_sub = sim[np.ix_(genre_indices, genre_indices)].copy()
    # Average similarity to other genre books (exclude self-similarity on diagonal)
    np.fill_diagonal(sim_sub, 0)
    n = len(genre_indices)
    centrality = sim_sub.sum(axis=1) / max(n - 1, 1)

    # Sort by centrality descending
    ranked = np.argsort(centrality)[::-1]

    data = []
    for rank_pos in ranked[:count]:
        idx = genre_indices[rank_pos]
        title = pt.index[idx]
        if title in book_info_lookup:
            info = book_info_lookup[title]
            data.append([info['title'], info['author'], info['image']])

    return data, genre_name


@app.route('/')
def index():
    return render_template('index.html',
                           book_name = list(popular_df['Book-Title'].values),
                           author=list(popular_df['Book-Author'].values),
                           image=list(popular_df['Image-URL-M'].values),
                           votes=list(popular_df['num_ratings'].values),
                           rating=list(popular_df['avg_rating'].values)
                           )

@app.route('/recommend')
def recommend_ui():
    return render_template('recommend.html', ncf_available=ncf_available,
                           genre_available=genre_available, all_genres=all_genres)

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
def mood_ui():
    return render_template('mood.html', ncf_available=ncf_available)

@app.route('/mood_recommend', methods=['POST'])
def mood_recommend():
    data = request.get_json()
    emotion = data.get('emotion', 'neutral').lower().strip()
    mode = data.get('mode', 'classic')

    genres = EMOTION_GENRES.get(emotion, EMOTION_GENRES['neutral'])

    # Gather recommendations from each mapped genre
    all_data = []
    for genre_name in genres:
        recs, _ = get_genre_recommendations(genre_name, mode, count=4)
        for book in recs:
            if book not in all_data:
                all_data.append(book)

    return jsonify({
        'emotion': emotion,
        'genres': genres,
        'books': all_data[:12],  # cap at 12 results
    })

@app.route('/autocomplete')
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

@app.route('/recommend_books', methods=['post'])
def recommend():
    user_input = request.form.get('user_input', '').strip()
    mode = request.form.get('mode', 'classic')  # 'classic' or 'ai'

    # Check if user typed a genre
    # First check if it was a genre autocomplete selection
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

    # ── Existing book title search logic ──
    matches = np.where(pt.index == user_input)[0]

    # If no exact match, try fuzzy matching
    if len(matches) == 0:
        close = get_close_matches(user_input, all_titles, n=1, cutoff=0.4)
        if close:
            user_input = close[0]
            matches = np.where(pt.index == user_input)[0]

    if len(matches) == 0:
        return render_template('recommend.html', data=[], error=f'No book found matching "{request.form.get("user_input")}". Try a different title or search by genre.',
                               ncf_available=ncf_available, genre_available=genre_available, all_genres=all_genres)

    index = matches[0]

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
        item = []
        temp_df = books[books['Book-Title'] == pt.index[i[0]]]
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Title'].values))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Author'].values))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Image-URL-M'].values))

        data.append(item)

    return render_template('recommend.html', data=data, matched_title=user_input, mode=used_mode,
                           ncf_available=ncf_available, genre_available=genre_available, all_genres=all_genres)

if __name__ == '__main__':
    app.run(debug=True, port=5001)