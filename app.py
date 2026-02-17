from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import os
from difflib import get_close_matches

popular_df = pickle.load(open('popular.pkl','rb'))
pt = pickle.load(open('pt.pkl','rb'))
books = pickle.load(open('books.pkl','rb'))
similarity_scores = pickle.load(open('similarity_scores.pkl','rb'))

# Load NCF deep learning similarity scores (if available)
ncf_available = os.path.exists('ncf_similarity_scores.pkl')
if ncf_available:
    ncf_similarity_scores = pickle.load(open('ncf_similarity_scores.pkl', 'rb'))
    print("✅ NCF deep learning model loaded!")
else:
    ncf_similarity_scores = None
    print("⚠️  NCF model not found — using classic mode only")

# Pre-compute list of all book titles for fuzzy matching
all_titles = list(pt.index)

app = Flask(__name__)

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
    return render_template('recommend.html', ncf_available=ncf_available)

@app.route('/autocomplete')
def autocomplete():
    """Return matching book titles for the live search dropdown."""
    query = request.args.get('q', '').strip().lower()
    if len(query) < 2:
        return jsonify([])

    # Substring match first (fast, intuitive)
    suggestions = [t for t in all_titles if query in t.lower()][:10]

    # If few substring matches, supplement with fuzzy matches
    if len(suggestions) < 5:
        fuzzy = get_close_matches(query, all_titles, n=10, cutoff=0.4)
        for f in fuzzy:
            if f not in suggestions:
                suggestions.append(f)
            if len(suggestions) >= 10:
                break

    return jsonify(suggestions)

@app.route('/recommend_books', methods=['post'])
def recommend():
    user_input = request.form.get('user_input', '').strip()
    mode = request.form.get('mode', 'classic')  # 'classic' or 'ai'

    # Try exact match first
    matches = np.where(pt.index == user_input)[0]

    # If no exact match, try fuzzy matching
    if len(matches) == 0:
        close = get_close_matches(user_input, all_titles, n=1, cutoff=0.4)
        if close:
            user_input = close[0]  # Use the best fuzzy match
            matches = np.where(pt.index == user_input)[0]

    if len(matches) == 0:
        return render_template('recommend.html', data=[], error=f'No book found matching "{request.form.get("user_input")}". Try a different title.', ncf_available=ncf_available)

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

    print(data)

    return render_template('recommend.html', data=data, matched_title=user_input, mode=used_mode, ncf_available=ncf_available)

if __name__ == '__main__':
    app.run(debug=True)