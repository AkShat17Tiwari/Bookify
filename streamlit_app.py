import streamlit as st
import pickle
import numpy as np
import os
from difflib import get_close_matches

# ── Page config ──
st.set_page_config(page_title="BOOKIFY", page_icon="📚", layout="wide")

# ── Load data ──
@st.cache_data
def load_data():
    popular_df = pickle.load(open('popular.pkl', 'rb'))
    pt = pickle.load(open('pt.pkl', 'rb'))
    books = pickle.load(open('books.pkl', 'rb'))
    similarity_scores = pickle.load(open('similarity_scores.pkl', 'rb'))
    all_titles = list(pt.index)
    # Load NCF deep learning scores if available
    ncf_scores = None
    if os.path.exists('ncf_similarity_scores.pkl'):
        ncf_scores = pickle.load(open('ncf_similarity_scores.pkl', 'rb'))
    return popular_df, pt, books, similarity_scores, all_titles, ncf_scores

popular_df, pt, books, similarity_scores, all_titles, ncf_similarity_scores = load_data()
ncf_available = ncf_similarity_scores is not None

# ── Custom CSS ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

.stApp {
    background: #0a0a0f !important;
    font-family: 'Inter', sans-serif;
}
header[data-testid="stHeader"] {
    background: rgba(10,10,15,0.85) !important;
    backdrop-filter: blur(20px);
}
#MainMenu, footer, .stDeployButton { display: none !important; }

/* Navbar */
.navbar {
    display: flex; align-items: center; justify-content: space-between;
    padding: 14px 30px; background: rgba(10,10,15,0.9);
    border-bottom: 1px solid rgba(255,255,255,0.06);
    margin: -1rem -1rem 1rem -1rem;
}
.nav-brand {
    font-size: 22px; font-weight: 700;
    background: linear-gradient(135deg, #00e68a, #00b4d8);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.owner-label { color: #fbbf24; font-weight: 700; font-size: 13px; letter-spacing: 1.5px; }
.owner-val {
    background: linear-gradient(135deg, #ff6b6b, #fbbf24);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    font-weight: 700; font-size: 13px; letter-spacing: 1.5px;
}

/* Hero */
.hero-title {
    text-align: center; font-size: 48px; font-weight: 800;
    background: linear-gradient(135deg, #fff, #a0a0a0);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 30px 0 8px; letter-spacing: -1.5px;
}
.hero-sub { text-align: center; color: #666; font-size: 17px; margin-bottom: 10px; }
.badge { text-align: center; margin-bottom: 30px; }
.badge span {
    display: inline-block; padding: 6px 16px; font-size: 12px;
    font-weight: 600; color: #00e68a; background: rgba(0,230,138,0.1);
    border: 1px solid rgba(0,230,138,0.2); border-radius: 20px;
}

/* Card styling via container */
div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlock"] {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 16px; padding: 14px;
    transition: all 0.3s;
}
div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlock"]:hover {
    border-color: rgba(0,230,138,0.3);
    box-shadow: 0 12px 40px rgba(0,230,138,0.06);
    transform: translateY(-4px);
}

/* Streamlit image rounding */
div[data-testid="stImage"] img { border-radius: 10px !important; }

/* Text colors */
.card-title { color: #f0f0f0; font-size: 14px; font-weight: 600; line-height: 1.4; margin: 0; }
.card-author { color: #00e68a; font-size: 13px; font-weight: 500; margin: 0; }
.card-stats { color: #888; font-size: 12px; margin: 0; }
.star { color: #fbbf24; }
.star-e { color: #333; }
.rec-author { color: #00b4d8; font-size: 13px; font-weight: 500; margin: 0; }
.rank { color: #00b4d8; font-size: 11px; font-weight: 700; background: rgba(0,180,216,0.15);
    border: 1px solid rgba(0,180,216,0.3); padding: 3px 10px; border-radius: 8px;
    display: inline-block; margin-bottom: 8px;
}

/* Match/Error banners */
.match-banner {
    padding: 14px 24px; background: rgba(0,230,138,0.06);
    border: 1px solid rgba(0,230,138,0.15); border-radius: 12px;
    color: #00e68a; font-size: 15px; text-align: center; margin-bottom: 20px;
}
.match-banner b { color: #fff; }
.err-banner {
    padding: 14px 24px; background: rgba(239,68,68,0.1);
    border: 1px solid rgba(239,68,68,0.2); border-radius: 12px;
    color: #f87171; font-size: 15px; text-align: center; margin-bottom: 20px;
}

/* Streamlit overrides */
.stButton > button {
    width: 100% !important; padding: 14px !important; font-size: 15px !important;
    font-weight: 600 !important; color: #0a0a0f !important;
    background: linear-gradient(135deg, #00e68a, #00b4d8) !important;
    border: none !important; border-radius: 12px !important;
}
.stButton > button:hover {
    box-shadow: 0 12px 40px rgba(0,230,138,0.2) !important;
}
div[data-testid="stTabs"] button { color: #aaa !important; font-weight: 500 !important; }
div[data-testid="stTabs"] button[aria-selected="true"] {
    color: #fff !important; background: rgba(255,255,255,0.08) !important; border-radius: 8px !important;
}
.stSelectbox label { color: #888 !important; }
</style>
""", unsafe_allow_html=True)

# ── Navbar ──
st.markdown("""
<div class="navbar">
    <span class="nav-brand">📚 BOOKIFY</span>
    <span><span class="owner-label">OWNER: </span><span class="owner-val">AKSHAT TIWARI</span></span>
</div>
""", unsafe_allow_html=True)

# ── Tabs ──
tab1, tab2 = st.tabs(["🏠 Home", "🔍 Recommend"])

# ════════════════════════════════════
# TAB 1 — Top 50 Books
# ════════════════════════════════════
with tab1:
    st.markdown(f'<div class="hero-title">Top 50 Books</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">The most popular and highest-rated books curated from thousands of reader reviews</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="badge"><span>{len(popular_df)} BOOKS</span></div>', unsafe_allow_html=True)

    num_cols = 4
    for row_start in range(0, len(popular_df), num_cols):
        cols = st.columns(num_cols, gap="medium")
        for col_idx, col in enumerate(cols):
            i = row_start + col_idx
            if i >= len(popular_df):
                break
            title = popular_df.iloc[i]['Book-Title']
            author = popular_df.iloc[i]['Book-Author']
            img = popular_df.iloc[i]['Image-URL-M']
            votes = int(popular_df.iloc[i]['num_ratings'])
            rating = float(popular_df.iloc[i]['avg_rating'])
            full = int(rating)
            stars = ''.join(['<span class="star">★</span>'] * full + ['<span class="star-e">★</span>'] * (5 - full))

            with col:
                with st.container(border=True):
                    st.image(img, use_container_width=True)
                    st.markdown(f'<p class="card-title">{title}</p>', unsafe_allow_html=True)
                    st.markdown(f'<p class="card-author">{author}</p>', unsafe_allow_html=True)
                    st.markdown(f'<p class="card-stats">Votes: {votes} &nbsp;|&nbsp; {stars}</p>', unsafe_allow_html=True)

# ════════════════════════════════════
# TAB 2 — Recommend
# ════════════════════════════════════
with tab2:
    st.markdown('<div class="hero-title">Discover Books</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Select a book and we\'ll find similar reads you\'ll love</div>', unsafe_allow_html=True)

    selected_book = st.selectbox(
        "Search for a book title",
        options=all_titles, index=None,
        placeholder="Start typing a book title..."
    )

    # Mode toggle
    if ncf_available:
        mode = st.radio(
            "Recommendation Engine",
            ["📊 Classic (Cosine Similarity)", "🤖 AI-Powered (Deep Learning)"],
            horizontal=True,
            help="Classic uses cosine similarity. AI-Powered uses Neural Collaborative Filtering."
        )
        use_ai = "AI-Powered" in mode
    else:
        use_ai = False

    if st.button("Get Recommendations →"):
        if not selected_book:
            st.markdown('<div class="err-banner">Please select a book title first.</div>', unsafe_allow_html=True)
        else:
            user_input = selected_book.strip()
            matches = np.where(pt.index == user_input)[0]

            if len(matches) == 0:
                close = get_close_matches(user_input, all_titles, n=1, cutoff=0.4)
                if close:
                    user_input = close[0]
                    matches = np.where(pt.index == user_input)[0]

            if len(matches) == 0:
                st.markdown(f'<div class="err-banner">No book found matching "{selected_book}". Try a different title.</div>', unsafe_allow_html=True)
            else:
                mode_label = ' <span style="background:rgba(167,139,250,0.12);border:1px solid rgba(167,139,250,0.25);color:#a78bfa;padding:3px 10px;font-size:11px;font-weight:700;border-radius:6px;margin-left:8px;letter-spacing:0.5px;">🤖 DEEP LEARNING</span>' if use_ai else ''
                st.markdown(f'<div class="match-banner">Showing recommendations for <b>"{user_input}"</b>{mode_label}</div>', unsafe_allow_html=True)

                index = matches[0]
                scores = ncf_similarity_scores if use_ai else similarity_scores
                similar_items = sorted(list(enumerate(scores[index])), key=lambda x: x[1], reverse=True)[1:5]

                data = []
                for item in similar_items:
                    temp_df = books[books['Book-Title'] == pt.index[item[0]]]
                    t = temp_df.drop_duplicates('Book-Title')['Book-Title'].values[0]
                    a = temp_df.drop_duplicates('Book-Title')['Book-Author'].values[0]
                    im = temp_df.drop_duplicates('Book-Title')['Image-URL-M'].values[0]
                    data.append((t, a, im))

                cols = st.columns(len(data), gap="medium")
                for rank, (col, (t, a, im)) in enumerate(zip(cols, data), 1):
                    with col:
                        with st.container(border=True):
                            st.markdown(f'<span class="rank">#{rank}</span>', unsafe_allow_html=True)
                            st.image(im, use_container_width=True)
                            st.markdown(f'<p class="card-title">{t}</p>', unsafe_allow_html=True)
                            st.markdown(f'<p class="rec-author">{a}</p>', unsafe_allow_html=True)
