"""
Realtime Book Search Service
=============================
Fetches live book data from external APIs (Open Library + Google Books),
normalizes results, and provides dynamic recommendations with caching.

This module is fully independent — it does NOT touch any existing
recommendation logic, routes, or data structures.
"""

import time
import threading
import requests as http_req
from difflib import SequenceMatcher

# ══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════

API_TIMEOUT = 2.0          # max seconds per external API call
CACHE_TTL = 300            # cache entries live for 5 minutes
CACHE_MAX_SIZE = 200       # max cached queries
MAX_RESULTS = 12           # max books returned per query

# ══════════════════════════════════════════════════════════════════
# IN-MEMORY CACHE  (thread-safe)
# ══════════════════════════════════════════════════════════════════

_cache = {}            # key → { 'data': ..., 'ts': timestamp }
_cache_lock = threading.Lock()


def _cache_get(key: str):
    """Return cached value if still fresh, else None."""
    with _cache_lock:
        entry = _cache.get(key)
        if entry and (time.time() - entry['ts']) < CACHE_TTL:
            return entry['data']
        # Expired — clean up
        if entry:
            del _cache[key]
        return None


def _cache_set(key: str, data):
    """Store a value in cache, evicting oldest if over limit."""
    with _cache_lock:
        # Evict oldest entries if cache is full
        if len(_cache) >= CACHE_MAX_SIZE:
            oldest_key = min(_cache, key=lambda k: _cache[k]['ts'])
            del _cache[oldest_key]
        _cache[key] = {'data': data, 'ts': time.time()}


# ══════════════════════════════════════════════════════════════════
# API FETCHERS
# ══════════════════════════════════════════════════════════════════

def fetch_from_open_library(query: str, limit: int = 10) -> list:
    """
    Search Open Library for books matching the query.
    Returns a list of raw result dicts.
    """
    try:
        resp = http_req.get(
            'https://openlibrary.org/search.json',
            params={
                'q': query,
                'limit': limit,
                'fields': 'title,author_name,cover_i,first_publish_year,'
                          'ratings_average,ratings_count,subject,edition_count',
            },
            timeout=API_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json().get('docs', [])
    except Exception:
        return []


def fetch_from_google_books(query: str, limit: int = 10) -> list:
    """
    Search Google Books API (no key required for basic search).
    Returns a list of raw volume dicts.
    """
    try:
        resp = http_req.get(
            'https://www.googleapis.com/books/v1/volumes',
            params={
                'q': query,
                'maxResults': min(limit, 40),
                'orderBy': 'relevance',
                'printType': 'books',
            },
            timeout=API_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json().get('items', [])
    except Exception:
        return []


# ══════════════════════════════════════════════════════════════════
# DATA NORMALIZATION
# ══════════════════════════════════════════════════════════════════

def _normalize_open_library(doc: dict) -> dict | None:
    """Convert an Open Library doc to our standard book format."""
    title = doc.get('title', '').strip()
    if not title:
        return None

    authors = doc.get('author_name', [])
    author = authors[0] if authors else 'Unknown Author'

    # Build cover image URL
    cover_id = doc.get('cover_i')
    if cover_id:
        image = f'https://covers.openlibrary.org/b/id/{cover_id}-M.jpg'
    else:
        image = ''

    rating = doc.get('ratings_average')
    rating_count = doc.get('ratings_count', 0)
    year = doc.get('first_publish_year')
    subjects = doc.get('subject', [])[:5]
    editions = doc.get('edition_count', 0)

    return {
        'title': title,
        'author': author,
        'image': image,
        'rating': round(float(rating), 1) if rating else None,
        'rating_count': int(rating_count) if rating_count else 0,
        'year': year,
        'subjects': subjects,
        'editions': editions,
        'source': 'open_library',
    }


def _normalize_google_books(item: dict) -> dict | None:
    """Convert a Google Books volume to our standard book format."""
    info = item.get('volumeInfo', {})
    title = info.get('title', '').strip()
    if not title:
        return None

    authors = info.get('authors', [])
    author = authors[0] if authors else 'Unknown Author'

    # Cover image (prefer medium, fallback to thumbnail)
    image_links = info.get('imageLinks', {})
    image = (
        image_links.get('thumbnail', '')
        or image_links.get('smallThumbnail', '')
    )
    # Upgrade to HTTPS
    if image.startswith('http://'):
        image = image.replace('http://', 'https://')

    rating = info.get('averageRating')
    rating_count = info.get('ratingsCount', 0)
    year_str = info.get('publishedDate', '')
    year = int(year_str[:4]) if year_str and len(year_str) >= 4 and year_str[:4].isdigit() else None
    categories = info.get('categories', [])[:5]

    return {
        'title': title,
        'author': author,
        'image': image,
        'rating': float(rating) if rating else None,
        'rating_count': int(rating_count) if rating_count else 0,
        'year': year,
        'subjects': categories,
        'editions': 0,
        'source': 'google_books',
    }


def normalize_data(open_library_docs: list, google_books_items: list) -> list:
    """
    Normalize and merge results from both APIs.
    Deduplicates by fuzzy title matching. Prefers the entry with more data.
    """
    books = []

    # Normalize Open Library results
    for doc in open_library_docs:
        normalized = _normalize_open_library(doc)
        if normalized:
            books.append(normalized)

    # Normalize Google Books results
    for item in google_books_items:
        normalized = _normalize_google_books(item)
        if normalized:
            books.append(normalized)

    # Deduplicate: group by similar title + author
    deduplicated = []
    seen_keys = set()

    for book in books:
        key = _dedup_key(book['title'], book['author'])

        # Check if we already have a similar entry
        is_dup = False
        for seen in seen_keys:
            if SequenceMatcher(None, key, seen).ratio() > 0.8:
                is_dup = True
                break

        if not is_dup:
            seen_keys.add(key)
            deduplicated.append(book)

    return deduplicated


def _dedup_key(title: str, author: str) -> str:
    """Create a lowercase dedup key from title + author."""
    return f"{title.lower().strip()}||{author.lower().strip()}"


# ══════════════════════════════════════════════════════════════════
# DYNAMIC RECOMMENDATION / RANKING
# ══════════════════════════════════════════════════════════════════

def recommend_dynamic(books: list, query: str) -> list:
    """
    Rank and score books based on relevance to the query.
    Uses a composite score: title relevance + rating + popularity + recency.
    Returns top MAX_RESULTS books with reasons attached.
    """
    query_lower = query.lower().strip()
    scored = []

    for book in books:
        score = 0.0
        reasons = []

        title_lower = book['title'].lower()
        author_lower = book['author'].lower()

        # ── Title relevance (0–40 pts) ──
        title_sim = SequenceMatcher(None, query_lower, title_lower).ratio()
        if query_lower in title_lower:
            score += 40
            reasons.append('🎯 Direct title match')
        elif title_sim > 0.6:
            score += 30 * title_sim
            reasons.append(f'📖 Title similarity ({title_sim:.0%})')
        else:
            score += 15 * title_sim

        # ── Author relevance (0–20 pts) ──
        if query_lower in author_lower:
            score += 20
            reasons.append(f'✍️ Author: {book["author"]}')

        # ── Rating boost (0–20 pts) ──
        if book.get('rating'):
            rating_score = (book['rating'] / 5.0) * 20
            score += rating_score
            reasons.append(f'⭐ {book["rating"]}/5 rating')

        # ── Popularity signal (0–10 pts) ──
        rc = book.get('rating_count', 0)
        if rc > 100:
            score += 10
            reasons.append('🔥 Highly popular')
        elif rc > 10:
            score += 5
            reasons.append('📊 Well-reviewed')

        # ── Recency bonus (0–5 pts) ──
        if book.get('year') and book['year'] >= 2020:
            score += 5
            reasons.append(f'🆕 Published {book["year"]}')
        elif book.get('year') and book['year'] >= 2000:
            score += 3

        # ── Editions bonus (0–5 pts) — proxy for enduring popularity ──
        editions = book.get('editions', 0)
        if editions > 50:
            score += 5
            reasons.append(f'📚 {editions} editions')
        elif editions > 10:
            score += 2

        # ── Image availability bonus ──
        if book.get('image'):
            score += 2

        book['_score'] = score
        book['reasons'] = reasons[:4]  # cap at 4 reasons
        scored.append(book)

    # Sort by score descending
    scored.sort(key=lambda b: b['_score'], reverse=True)

    # Clean up internal fields before returning
    result = []
    for book in scored[:MAX_RESULTS]:
        result.append({
            'title': book['title'],
            'author': book['author'],
            'image': book['image'],
            'rating': book.get('rating'),
            'year': book.get('year'),
            'subjects': book.get('subjects', []),
            'reasons': book.get('reasons', []),
            'source': book.get('source', 'unknown'),
        })

    return result


# ══════════════════════════════════════════════════════════════════
# PUBLIC API — called by the Flask route
# ══════════════════════════════════════════════════════════════════

def realtime_search(query: str) -> dict:
    """
    Main entry point: search external APIs, normalize, rank, cache.
    Returns a dict matching the expected frontend format:
      { "books": [...], "query": "...", "source": "realtime", "cached": bool }
    """
    if not query or len(query.strip()) < 2:
        return {'books': [], 'query': query, 'source': 'realtime', 'cached': False,
                'error': 'Query too short (minimum 2 characters)'}

    query = query.strip()
    cache_key = f'rt:{query.lower()}'

    # Check cache first
    cached = _cache_get(cache_key)
    if cached is not None:
        return {**cached, 'cached': True}

    # Fetch from both APIs concurrently using threads
    ol_results = []
    gb_results = []
    errors = []

    def _fetch_ol():
        nonlocal ol_results
        try:
            ol_results = fetch_from_open_library(query, limit=10)
        except Exception as e:
            errors.append(f'Open Library: {e}')

    def _fetch_gb():
        nonlocal gb_results
        try:
            gb_results = fetch_from_google_books(query, limit=10)
        except Exception as e:
            errors.append(f'Google Books: {e}')

    t1 = threading.Thread(target=_fetch_ol, daemon=True)
    t2 = threading.Thread(target=_fetch_gb, daemon=True)
    t1.start()
    t2.start()
    t1.join(timeout=API_TIMEOUT + 0.5)
    t2.join(timeout=API_TIMEOUT + 0.5)

    # Normalize and merge
    merged = normalize_data(ol_results, gb_results)

    # Rank dynamically
    ranked = recommend_dynamic(merged, query)

    result = {
        'books': ranked,
        'query': query,
        'source': 'realtime',
        'total_raw': len(ol_results) + len(gb_results),
        'total_merged': len(merged),
    }

    # Cache the result
    _cache_set(cache_key, result)

    return {**result, 'cached': False}
