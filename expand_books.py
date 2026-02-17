"""
Expand BOOKIFY catalog from ~706 to ~1200+ books.

Strategy: We already have books.pkl with 271K books and pt.pkl with 706 books.
The pivot table contains the actual rating data. We'll:
1. Keep all existing 706 books
2. Select ~500 additional books from books.pkl with diverse authors/publishers
3. Create synthetic collaborative filtering data for the new books
4. Rebuild pt.pkl, similarity_scores.pkl, popular.pkl
"""

import os
import pickle
import random
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

random.seed(42)
np.random.seed(42)

TARGET_NEW_BOOKS = 500

# Genre/category signals based on publisher names
GENRE_PUBLISHERS = {
    'Science Fiction': ['Tor Books', 'Bantam Spectra', 'DAW Books', 'Ace Books', 'Del Rey', 'Baen Books', 'Orbit'],
    'Fantasy': ['Tor Books', 'Del Rey', 'DAW Books', 'Ace Books', 'Orbit', 'HarperVoyager'],
    'Mystery/Thriller': ['Mysterious Press', 'St. Martin\'s Minotaur', 'Berkley Prime Crime', 'Penguin Putnam'],
    'Romance': ['Harlequin', 'Silhouette', 'Avon Books', 'Avon', 'Love Inspired'],
    'Horror': ['Cemetery Dance', 'Leisure Books'],
    'Literary Fiction': ['Vintage', 'Penguin Books', 'Knopf', 'Farrar Straus Giroux', 'Scribner',
                          'Random House', 'Harper Perennial', 'Picador', 'Anchor'],
    'Non-Fiction': ['W. W. Norton', 'Basic Books', 'Crown', 'Free Press', 'PublicAffairs'],
    'History': ['Oxford University Press', 'Cambridge University Press', 'Penguin Books'],
    'Young Adult': ['Scholastic', 'Puffin', 'Delacorte Press', 'HarperTeen', 'Aladdin'],
    'Children': ['Scholastic', 'Puffin', 'Random House Books for Young Readers', "Children's Press"],
    'Classics': ['Penguin Classics', 'Dover Publications', 'Oxford World\'s Classics', 'Wordsworth Editions',
                 'Modern Library', 'Everyman\'s Library'],
    'Biography/Memoir': ['Simon &amp; Schuster', 'Harper', 'Little Brown', 'Viking'],
    'Self-Help': ['HarperCollins', 'Hay House', 'Simon &amp; Schuster'],
    'Cooking': ['Clarkson Potter', 'Ten Speed Press', "Cook's Illustrated"],
    'Travel': ['Lonely Planet', 'Fodor\'s', 'DK Publishing', 'Rough Guides', 'Frommer\'s'],
}


def load_existing_data():
    """Load all existing pickle files."""
    print("📂 Loading existing data...")
    pt = pickle.load(open('pt.pkl', 'rb'))
    books = pickle.load(open('books.pkl', 'rb'))
    popular = pickle.load(open('popular.pkl', 'rb'))
    sim = pickle.load(open('similarity_scores.pkl', 'rb'))

    print(f"  ✓ Pivot table: {pt.shape[0]} books × {pt.shape[1]} users")
    print(f"  ✓ Books catalog: {len(books):,} total")
    print(f"  ✓ Popular: {len(popular)} books")
    print(f"  ✓ Similarity: {sim.shape}")

    return pt, books, popular, sim


def select_diverse_books(books, existing_titles, target_count=500):
    """Select diverse books from the full catalog that aren't already in the PT."""
    print(f"\n🎯 Selecting {target_count} diverse books...")

    # Get books NOT already in PT
    available = books[~books['Book-Title'].isin(existing_titles)].copy()
    available = available.drop_duplicates('Book-Title')

    # Clean up — remove books with missing info
    available = available.dropna(subset=['Book-Title', 'Book-Author', 'Image-URL-M', 'Publisher'])
    available = available[available['Book-Title'].str.len() > 2]
    available = available[available['Book-Author'].str.len() > 2]
    available = available[available['Publisher'].str.len() > 1]

    # Filter out books with placeholder images
    available = available[~available['Image-URL-M'].str.contains('nophoto', case=False, na=False)]

    print(f"  Available books (with complete info): {len(available):,}")

    selected = []
    selected_titles = set()

    # Strategy: Pick books from diverse publishers/genres
    all_genre_publishers = {}
    for genre, publishers in GENRE_PUBLISHERS.items():
        for pub in publishers:
            all_genre_publishers[pub.lower()] = genre

    # Phase 1: Pick from known genre publishers (ensures diversity)
    per_genre_target = target_count // len(GENRE_PUBLISHERS)
    for genre, publishers in GENRE_PUBLISHERS.items():
        genre_books = available[
            available['Publisher'].str.lower().apply(
                lambda p: any(pub.lower() in str(p).lower() for pub in publishers)
            )
        ]
        genre_books = genre_books[~genre_books['Book-Title'].isin(selected_titles)]

        pick_count = min(per_genre_target, len(genre_books))
        if pick_count > 0:
            picked = genre_books.sample(n=pick_count, random_state=42)
            selected.append(picked)
            selected_titles.update(picked['Book-Title'].values)
            print(f"  📚 {genre}: picked {pick_count} books")

    # Phase 2: Fill remaining slots with popular authors not yet included
    remaining = target_count - len(selected_titles)
    if remaining > 0:
        # Get popular authors (by book count in the catalog)
        author_counts = available[~available['Book-Title'].isin(selected_titles)] \
            .groupby('Book-Author').size().sort_values(ascending=False)

        # Pick from top authors, 1-2 books each
        fill_books = available[~available['Book-Title'].isin(selected_titles)]
        top_authors = author_counts.head(remaining).index

        for author in top_authors:
            if len(selected_titles) >= target_count:
                break
            author_books = fill_books[fill_books['Book-Author'] == author]
            if len(author_books) > 0:
                pick = author_books.sample(n=min(2, len(author_books)), random_state=42)
                selected.append(pick)
                selected_titles.update(pick['Book-Title'].values)

        print(f"  📚 Popular authors fill: picked {len(selected_titles) - (target_count - remaining)} books")

    # Phase 3: Random fill if still short
    remaining = target_count - len(selected_titles)
    if remaining > 0:
        random_fill = available[~available['Book-Title'].isin(selected_titles)].sample(
            n=min(remaining, len(available) - len(selected_titles)),
            random_state=42
        )
        selected.append(random_fill)
        selected_titles.update(random_fill['Book-Title'].values)
        print(f"  📚 Random fill: picked {len(random_fill)} books")

    selected_df = pd.concat(selected).drop_duplicates('Book-Title')
    print(f"\n  ✓ Total selected: {len(selected_df)} diverse books")

    # Print genre distribution
    genre_counts = {}
    for _, row in selected_df.iterrows():
        pub = str(row.get('Publisher', '')).lower()
        found_genre = 'Other'
        for genre, publishers in GENRE_PUBLISHERS.items():
            if any(p.lower() in pub for p in publishers):
                found_genre = genre
                break
        genre_counts[found_genre] = genre_counts.get(found_genre, 0) + 1

    print(f"\n  Genre distribution:")
    for genre, count in sorted(genre_counts.items(), key=lambda x: -x[1]):
        print(f"    {genre}: {count}")

    return selected_df


def generate_collaborative_data(existing_pt, new_books_df, existing_books_df):
    """
    Generate synthetic but realistic collaborative filtering data for new books.

    Strategy:
    - For each new book, find existing books with similar authors/publishers
    - Copy rating patterns from similar books with noise
    - This creates meaningful collaborative filtering signals
    """
    print(f"\n🤖 Generating collaborative filtering data for {len(new_books_df)} new books...")

    users = existing_pt.columns.tolist()
    n_users = len(users)

    # Build author/publisher lookup from existing PT
    existing_titles = set(existing_pt.index)
    existing_info = existing_books_df[existing_books_df['Book-Title'].isin(existing_titles)] \
        .drop_duplicates('Book-Title').set_index('Book-Title')

    new_ratings = {}

    for idx, (_, new_book) in enumerate(new_books_df.iterrows()):
        title = new_book['Book-Title']
        author = str(new_book.get('Book-Author', ''))
        publisher = str(new_book.get('Publisher', ''))

        # Find similar existing books (same author or publisher)
        same_author = existing_info[existing_info['Book-Author'] == author].index.tolist()
        same_publisher = existing_info[
            existing_info['Publisher'].str.lower() == publisher.lower()
        ].index.tolist() if publisher else []

        # Choose template books
        templates = []
        if same_author:
            templates.extend(same_author[:3])
        if same_publisher and len(templates) < 3:
            templates.extend(same_publisher[:3 - len(templates)])

        # If no matching author/publisher, pick random existing books
        if not templates:
            templates = random.sample(list(existing_titles), min(3, len(existing_titles)))

        # Generate ratings based on templates + noise
        combined_pattern = np.zeros(n_users)
        for tmpl in templates:
            if tmpl in existing_pt.index:
                combined_pattern += existing_pt.loc[tmpl].values

        combined_pattern /= max(len(templates), 1)

        # Add noise and sparsify
        noise = np.random.normal(0, 1.5, n_users)
        combined_pattern = combined_pattern + noise

        # Make sparse: most users don't rate most books
        # Keep 3-8% of ratings
        sparsity = random.uniform(0.03, 0.08)
        mask = np.random.random(n_users) < sparsity
        # Always keep ratings where original template had ratings
        template_rated = combined_pattern != 0
        keep = mask | (template_rated & (np.random.random(n_users) < 0.3))

        ratings = np.where(keep, combined_pattern, 0)
        ratings = np.clip(ratings, 0, 10).round(0)

        new_ratings[title] = ratings

        if (idx + 1) % 100 == 0:
            print(f"  Generated ratings for {idx + 1}/{len(new_books_df)} books...")

    new_pt = pd.DataFrame(new_ratings, index=users).T
    new_pt.columns = users
    print(f"  ✓ New ratings matrix: {new_pt.shape}")

    return new_pt


def build_expanded_system(existing_pt, new_pt, books, popular):
    """Combine existing and new data, rebuild similarity scores."""
    print(f"\n🔨 Building expanded recommendation system...")

    # Combine pivot tables
    combined_pt = pd.concat([existing_pt, new_pt])
    combined_pt = combined_pt[~combined_pt.index.duplicated(keep='first')]
    combined_pt = combined_pt.fillna(0)
    print(f"  ✓ Combined PT: {combined_pt.shape[0]} books × {combined_pt.shape[1]} users")

    # Sort alphabetically for consistent ordering
    combined_pt = combined_pt.sort_index()

    # Compute cosine similarity
    print(f"  Computing cosine similarity ({combined_pt.shape[0]}×{combined_pt.shape[0]})...")
    sim_scores = cosine_similarity(combined_pt)
    print(f"  ✓ Similarity matrix: {sim_scores.shape}")

    # Update popular books — add some new popular ones
    # Keep existing popular but check if any new ones qualify
    all_titles_in_pt = set(combined_pt.index)
    existing_popular_titles = set(popular['Book-Title'])

    # Calculate rating stats for new books
    new_book_stats = []
    for title in new_pt.index:
        ratings = new_pt.loc[title]
        nonzero = ratings[ratings > 0]
        if len(nonzero) >= 5:
            new_book_stats.append({
                'Book-Title': title,
                'num_ratings': len(nonzero),
                'avg_rating': nonzero.mean()
            })

    if new_book_stats:
        new_stats_df = pd.DataFrame(new_book_stats)
        new_stats_df = new_stats_df.merge(
            books.drop_duplicates('Book-Title')[['Book-Title', 'Book-Author', 'Image-URL-M']],
            on='Book-Title'
        )
        new_stats_df = new_stats_df.sort_values('avg_rating', ascending=False).head(10)

        # Combine with existing popular, keep top 50
        extended_popular = pd.concat([popular, new_stats_df]).drop_duplicates('Book-Title')
        extended_popular = extended_popular.sort_values('avg_rating', ascending=False).head(50)
        popular = extended_popular

    print(f"  ✓ Popular books: {len(popular)}")

    return combined_pt, sim_scores, popular


def save_all(pt, sim_scores, popular, books):
    """Save all pickle files with backups."""
    print(f"\n💾 Saving pickle files...")

    # Backup originals
    for f in ['pt.pkl', 'similarity_scores.pkl', 'popular.pkl']:
        if os.path.exists(f) and not os.path.exists(f + '.bak'):
            os.rename(f, f + '.bak')
            print(f"  📁 Backed up {f} → {f}.bak")

    pickle.dump(pt, open('pt.pkl', 'wb'))
    pickle.dump(sim_scores, open('similarity_scores.pkl', 'wb'))
    pickle.dump(popular, open('popular.pkl', 'wb'))

    sizes = {
        'pt.pkl': os.path.getsize('pt.pkl') / 1024 / 1024,
        'similarity_scores.pkl': os.path.getsize('similarity_scores.pkl') / 1024 / 1024,
        'popular.pkl': os.path.getsize('popular.pkl') / 1024,
    }

    print(f"  ✓ pt.pkl ({sizes['pt.pkl']:.1f} MB)")
    print(f"  ✓ similarity_scores.pkl ({sizes['similarity_scores.pkl']:.1f} MB)")
    print(f"  ✓ popular.pkl ({sizes['popular.pkl']:.1f} KB)")


# ── Main ──
if __name__ == '__main__':
    print("=" * 60)
    print("BOOKIFY — Expand Book Catalog (+500 books)")
    print("=" * 60)

    # Load existing
    pt, books, popular, sim = load_existing_data()
    old_count = pt.shape[0]

    # Select diverse new books
    existing_titles = set(pt.index)
    new_books = select_diverse_books(books, existing_titles, target_count=TARGET_NEW_BOOKS)

    # Generate collaborative data
    new_pt = generate_collaborative_data(pt, new_books, books)

    # Build expanded system
    combined_pt, sim_scores, popular = build_expanded_system(pt, new_pt, books, popular)

    # Show results
    new_count = combined_pt.shape[0]
    print(f"\n{'=' * 60}")
    print(f"📈 EXPANSION RESULT")
    print(f"  Before: {old_count} books")
    print(f"  After:  {new_count} books")
    print(f"  Added:  {new_count - old_count} new books")
    print(f"  Sample new titles:")
    new_titles = [t for t in combined_pt.index if t not in existing_titles]
    for t in random.sample(new_titles, min(15, len(new_titles))):
        book_info = books[books['Book-Title'] == t].iloc[0]
        print(f"    • {t} — {book_info['Book-Author']} ({book_info.get('Publisher', 'N/A')})")
    print(f"{'=' * 60}")

    # Save
    save_all(combined_pt, sim_scores, popular, books)

    print(f"\n✅ Expansion complete! Next step: retrain the NCF model:")
    print(f"   python train_ncf.py")
