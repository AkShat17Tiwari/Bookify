"""
BOOKIFY — Book Cover Analyzer
==============================
Analyzes book cover images to predict likely genres based on
dominant color palette extraction using K-Means clustering.

Color palettes are mapped to genre moods:
  - Dark/muted tones    → Mystery/Thriller, Horror
  - Warm/bright colors  → Romance, Cooking, Travel
  - Cool blues/greens   → Science Fiction, Fantasy
  - Pastels/soft shades → Children, Poetry, Self-Help
  - Neutral/earth tones → Literary Fiction, History, Classics
"""

import io
import requests
from PIL import Image
import numpy as np

# ── Genre-Color Mapping ──────────────────────────────────────────
# Each rule: (hue_range, sat_range, val_range) → [(genre, weight)]
# HSV ranges: H [0-360], S [0-1], V [0-1]

GENRE_COLOR_RULES = [
    # Dark / muted → dark themes
    {
        'name': 'dark',
        'condition': lambda h, s, v: v < 0.3,
        'genres': [('Mystery/Thriller', 0.9), ('Horror', 0.8), ('Literary Fiction', 0.4)],
    },
    # Deep reds → passion, danger
    {
        'name': 'deep_red',
        'condition': lambda h, s, v: (h < 20 or h > 340) and s > 0.5 and v > 0.3,
        'genres': [('Romance', 0.9), ('Mystery/Thriller', 0.5), ('Horror', 0.4)],
    },
    # Warm orange/yellow → friendly, appetizing
    {
        'name': 'warm',
        'condition': lambda h, s, v: 20 <= h <= 55 and s > 0.4 and v > 0.4,
        'genres': [('Cooking', 0.8), ('Travel', 0.7), ('Self-Help', 0.5), ('Children', 0.4)],
    },
    # Bright yellow/gold → optimism, adventure
    {
        'name': 'gold',
        'condition': lambda h, s, v: 40 <= h <= 65 and s > 0.3 and v > 0.6,
        'genres': [('Travel', 0.8), ('Fantasy', 0.6), ('Young Adult', 0.5)],
    },
    # Cool green → nature, calm
    {
        'name': 'green',
        'condition': lambda h, s, v: 80 <= h <= 160 and s > 0.3 and v > 0.3,
        'genres': [('Fantasy', 0.7), ('Travel', 0.6), ('Non-Fiction', 0.4)],
    },
    # Cool blue → tech, mystery, calm
    {
        'name': 'blue',
        'condition': lambda h, s, v: 190 <= h <= 260 and s > 0.3 and v > 0.3,
        'genres': [('Science Fiction', 0.9), ('Fantasy', 0.6), ('Mystery/Thriller', 0.4)],
    },
    # Purple / violet → mystical, royal
    {
        'name': 'purple',
        'condition': lambda h, s, v: 260 <= h <= 310 and s > 0.3 and v > 0.3,
        'genres': [('Fantasy', 0.9), ('Romance', 0.5), ('Poetry', 0.5)],
    },
    # Pink → romance, youth
    {
        'name': 'pink',
        'condition': lambda h, s, v: (310 <= h <= 340) and s > 0.3 and v > 0.5,
        'genres': [('Romance', 0.9), ('Young Adult', 0.6), ('Children', 0.5), ('Poetry', 0.4)],
    },
    # Pastel / soft (low saturation, high value)
    {
        'name': 'pastel',
        'condition': lambda h, s, v: s < 0.3 and v > 0.6,
        'genres': [('Children', 0.7), ('Poetry', 0.6), ('Self-Help', 0.5), ('Religious/Spiritual', 0.4)],
    },
    # Earth tones (brownish)
    {
        'name': 'earth',
        'condition': lambda h, s, v: 15 <= h <= 45 and 0.2 < s < 0.6 and 0.2 < v < 0.6,
        'genres': [('History', 0.8), ('Literary Fiction', 0.7), ('Classics', 0.6), ('Biography', 0.5)],
    },
    # Neutral / grey
    {
        'name': 'neutral',
        'condition': lambda h, s, v: s < 0.15 and 0.3 < v < 0.7,
        'genres': [('Literary Fiction', 0.7), ('Classics', 0.6), ('Non-Fiction', 0.5), ('History', 0.4)],
    },
    # Very bright / white
    {
        'name': 'bright',
        'condition': lambda h, s, v: s < 0.15 and v > 0.85,
        'genres': [('Self-Help', 0.6), ('Non-Fiction', 0.5), ('Classics', 0.4)],
    },
]


def extract_palette(image, n_colors=5):
    """Extract dominant colors from an image using simple K-means-like approach."""
    # Resize for speed
    img = image.copy()
    img.thumbnail((150, 150))
    img = img.convert('RGB')

    pixels = np.array(img).reshape(-1, 3).astype(float)

    # Simple K-Means (manual to avoid sklearn dependency at import time)
    np.random.seed(42)
    indices = np.random.choice(len(pixels), size=min(n_colors, len(pixels)), replace=False)
    centers = pixels[indices].copy()

    for _ in range(15):  # iterations
        # Assign each pixel to nearest center
        dists = np.linalg.norm(pixels[:, None] - centers[None, :], axis=2)
        labels = np.argmin(dists, axis=1)

        # Update centers
        new_centers = np.array([
            pixels[labels == k].mean(axis=0) if np.any(labels == k) else centers[k]
            for k in range(n_colors)
        ])

        if np.allclose(centers, new_centers, atol=1):
            break
        centers = new_centers

    # Count pixels per cluster for weighting
    _, counts = np.unique(labels, return_counts=True)
    weights = counts / counts.sum()

    return centers, weights


def rgb_to_hsv(r, g, b):
    """Convert RGB [0-255] to HSV with H [0-360], S [0-1], V [0-1]."""
    r, g, b = r / 255.0, g / 255.0, b / 255.0
    mx = max(r, g, b)
    mn = min(r, g, b)
    diff = mx - mn

    # Hue
    if diff == 0:
        h = 0
    elif mx == r:
        h = (60 * ((g - b) / diff) + 360) % 360
    elif mx == g:
        h = (60 * ((b - r) / diff) + 120) % 360
    else:
        h = (60 * ((r - g) / diff) + 240) % 360

    # Saturation
    s = 0 if mx == 0 else diff / mx

    # Value
    v = mx

    return h, s, v


def analyze_cover(image_url):
    """
    Analyze a book cover from URL and return genre predictions.

    Returns:
        dict with 'genres' (list of (genre, score) tuples, sorted desc),
                   'palette' (list of hex color strings),
                   'error' (str or None)
    """
    try:
        # Download image
        resp = requests.get(image_url, timeout=10, headers={
            'User-Agent': 'BOOKIFY/1.0'
        })
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content))

    except Exception as e:
        return {'genres': [], 'palette': [], 'error': f'Could not load image: {str(e)}'}

    try:
        centers, weights = extract_palette(img, n_colors=5)

        # Convert palette to hex for frontend display
        palette_hex = []
        for c in centers:
            r, g, b = int(c[0]), int(c[1]), int(c[2])
            palette_hex.append(f'#{r:02x}{g:02x}{b:02x}')

        # Score genres based on color rules
        genre_scores = {}
        for i, (center, weight) in enumerate(zip(centers, weights)):
            h, s, v = rgb_to_hsv(center[0], center[1], center[2])

            for rule in GENRE_COLOR_RULES:
                if rule['condition'](h, s, v):
                    for genre, score in rule['genres']:
                        genre_scores[genre] = genre_scores.get(genre, 0) + score * weight

        # Normalize and sort
        if genre_scores:
            max_score = max(genre_scores.values())
            genre_scores = {g: round(s / max_score, 2) for g, s in genre_scores.items()}

        sorted_genres = sorted(genre_scores.items(), key=lambda x: x[1], reverse=True)

        return {
            'genres': sorted_genres[:6],  # Top 6
            'palette': palette_hex,
            'error': None,
        }

    except Exception as e:
        return {'genres': [], 'palette': [], 'error': f'Analysis failed: {str(e)}'}


if __name__ == '__main__':
    # Quick test
    test_url = 'https://images-na.ssl-images-amazon.com/images/I/51Ga5GuElyL._SX308_BO1,204,203,200_.jpg'
    result = analyze_cover(test_url)
    print('Palette:', result['palette'])
    print('Genres:')
    for genre, score in result['genres']:
        print(f'  {genre}: {score}')
