# 📚 BOOKIFY — Intelligent Book Recommendation System

### 🚀 **[Live Demo](https://bookify-n7pc.onrender.com)** | 💻 **[Localhost](http://127.0.0.1:5001)**

A full-stack book recommendation engine featuring collaborative filtering, deep learning (Neural Collaborative Filtering), genre-based search, and webcam-powered mood detection for personalized reading suggestions.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        BOOKIFY                              │
├─────────────┬──────────────┬────────────────────────────────┤
│  Flask App  │ Streamlit App│       ML / Data Pipeline       │
│  (app.py)   │(streamlit_   │                                │
│             │  app.py)     │                                │
├─────────────┴──────────────┴────────────────────────────────┤
│                     Core Engine                             │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────────────┐ │
│  │ Collaborative│ │   NCF Deep   │ │   Genre-Based        │ │
│  │  Filtering   │ │   Learning   │ │   Recommendations    │ │
│  │(similarity_  │ │(ncf_simila-  │ │  (genre_data.pkl)    │ │
│  │ scores.pkl)  │ │ rity_scores) │ │                      │ │
│  └──────────────┘ └──────────────┘ └──────────────────────┘ │
│  ┌──────────────┐ ┌──────────────────────────────────────┐  │
│  │ Mood-Based   │ │   Data: books.pkl, pt.pkl,           │  │
│  │ (face-api.js │ │   popular.pkl                        │  │
│  │ + emotion →  │ │                                      │  │
│  │   genre map) │ │                                      │  │
│  └──────────────┘ └──────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Request Flow

```
User → Flask / Streamlit UI
         │
         ├─→ Home Page       → Top 50 popular books (pre-computed)
         ├─→ Recommend Page  → Book title input → similarity lookup → top-N
         │                   → Genre keyword    → genre centrality ranking
         └─→ Mood Page       → Webcam capture → face-api.js emotion detect
                              → Emotion → genre mapping → recommendations
```

---

## 🛠️ Tech Stack

| Layer           | Technology                                                 |
|:----------------|:-----------------------------------------------------------|
| **Backend**     | Python 3, Flask, Gunicorn                                  |
| **Frontend**    | HTML5, CSS3, Vanilla JavaScript, Jinja2 Templates          |
| **Alt Frontend**| Streamlit (standalone app)                                 |
| **ML / AI**     | PyTorch (NCF model), Scikit-learn (cosine similarity)      |
| **Face Detection** | face-api.js (TinyFaceDetector + FaceExpressionNet, CDN) |
| **Data**        | Pandas, NumPy, Pickle (serialized DataFrames & matrices)   |
| **Deployment**  | Procfile (Heroku-ready), Gunicorn WSGI                     |

---

## 📁 Project Structure

```
book-recommender-system-master/
│
├── app.py                      # Flask web application (main server)
├── streamlit_app.py            # Streamlit alternative UI
├── train_ncf.py                # NCF deep learning training script
├── expand_books.py             # Catalog expansion script (~706 → 1200+ books)
├── book-recommender-system.ipynb  # Jupyter notebook (EDA & prototyping)
│
├── templates/
│   ├── index.html              # Home page — Top 50 popular books
│   ├── recommend.html          # Recommend page — search by title or genre
│   └── mood.html               # Mood page — webcam emotion → book recs
│
├── books.pkl                   # Full book catalog (~271K books)
├── pt.pkl                      # User-book pivot table (collaborative filtering)
├── popular.pkl                 # Top 50 popular books (pre-computed)
├── similarity_scores.pkl       # Cosine similarity matrix (classic mode)
├── ncf_similarity_scores.pkl   # NCF-learned similarity matrix (AI mode)
├── ncf_book_embeddings.pkl     # Learned book embedding vectors
├── genre_data.pkl              # Genre classifications & mappings
│
├── requirements.txt            # Python dependencies
├── Procfile                    # Heroku deployment config
└── .gitignore
```

---

## ✨ Features

### 1. Top 50 Popular Books (`/`)
Displays the most popular and highest-rated books with cover images, author names, vote counts, and star ratings.

### 2. Smart Book Recommendations (`/recommend`)
- **Title Search**: Enter a book title → get 4 similar books using collaborative filtering
- **Genre Search**: Type a genre keyword (e.g., "sci-fi", "romance") → get top genre-ranked books
- **Fuzzy Matching**: Handles typos and partial inputs
- **Live Autocomplete**: Real-time suggestions as you type
- **Dual Mode**: Toggle between Classic (cosine similarity) and AI-Powered (NCF deep learning)

### 3. Mood-Based Recommendations (`/mood`)
- **Webcam Capture**: Take a photo using your device camera
- **Emotion Detection**: face-api.js analyzes facial expressions in-browser
- **Accuracy Display**: Shows confidence score with animated accuracy bar
- **Emotion → Genre Mapping**: Maps 7 emotions to relevant book genres:

  | Emotion    | Genres                              |
  |:-----------|:------------------------------------|
  | Happy 😊   | Romance, Travel, Cooking            |
  | Sad 😢     | Self-Help, Poetry, Religious        |
  | Angry 😠   | Mystery/Thriller, Horror            |
  | Fearful 😨 | Self-Help, Fantasy, Children        |
  | Disgusted 🤢| Science Fiction, Fantasy            |
  | Surprised 😲| Mystery/Thriller, Science Fiction   |
  | Neutral 😐 | Literary Fiction, Classics, Non-Fiction |

### 4. Contact Section
Owner contact details available at the footer of every page.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- pip

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd book-recommender-system-master

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Flask App

```bash
python app.py
```
Open **http://127.0.0.1:5001** in your browser.

### Running the Streamlit App

```bash
streamlit run streamlit_app.py
```

---

## 🌍 Deployment

### **[🔗 Visit Live App](https://bookify-n7pc.onrender.com)**

To deploy your own instance:
1. Fork the repository
2. Create a new Web Service on Render
3. Connect your GitHub repo
4. Render will automatically detect the `render.yaml` configuration

---

## 🧠 ML Pipeline

### Classic Mode — Cosine Similarity
1. Load user-book rating pivot table (`pt.pkl`)
2. Compute pairwise cosine similarity between all books
3. For a given book, return the top-N most similar books

### AI Mode — Neural Collaborative Filtering
1. Extract user-book rating triplets from `pt.pkl`
2. Train an NCF model with user/book embeddings + MLP layers
3. Extract learned book embeddings
4. Compute cosine similarity on learned embeddings
5. Use as drop-in replacement for classic similarity matrix

```bash
# Retrain the NCF model
python train_ncf.py
```

**NCF Architecture:**
```
User Embedding (64) ─┐
                     ├─→ Concat (128) → MLP [128 → 64 → 32] → Rating
Book Embedding (64) ─┘
```

### Genre System
Genre classification uses publisher-based heuristics via `expand_books.py`, supporting 15+ genres with alias matching (e.g., "sci-fi" → "Science Fiction").

### Catalog Expansion
```bash
# Expand from ~706 to ~1200+ books
python expand_books.py
python train_ncf.py   # Retrain after expansion
```

---

## 📬 Contact

| | |
|:--|:--|
| **Owner** | Akshat Tiwari |
| **Phone** | 7080046904 |
| **Email** | akshatr147@gmail.com |

---

<p align="center">Made with ❤️ by Akshat Tiwari</p>
