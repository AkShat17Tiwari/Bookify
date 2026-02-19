"""
BOOKIFY — Expand to ~1 Million Books (Fast Hybrid Approach)
============================================================
Uses concurrent API fetching from Open Library + synthetic augmentation
from the existing 242K catalog to reach ~1M books quickly.

Usage:
    python fetch_million_books.py
"""

import os
import sys
import pickle
import random
import time
import json
import urllib.request
import urllib.parse
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

random.seed(42)
np.random.seed(42)

# ── Configuration ──
TARGET_TOTAL_BOOKS = 1_000_000
CF_TARGET = 5000
API_FETCH_TARGET = 200_000   # try to get this many from API
BATCH_SIZE = 100
MAX_PAGES_PER_SUBJECT = 10
MAX_WORKERS = 8              # concurrent API requests
REQUEST_DELAY = 0.05

# ── Genre subjects ──
GENRE_SUBJECTS = {
    'Science Fiction': ['science_fiction', 'sci-fi', 'space_opera', 'cyberpunk', 'dystopia',
                        'time_travel', 'aliens', 'robots', 'futuristic', 'post_apocalyptic'],
    'Fantasy': ['fantasy', 'epic_fantasy', 'urban_fantasy', 'magic', 'dragons',
                'wizards', 'mythology', 'fairy_tales', 'dark_fantasy', 'swords_and_sorcery'],
    'Mystery/Thriller': ['mystery', 'thriller', 'detective', 'crime_fiction', 'suspense',
                         'murder_mystery', 'espionage', 'noir', 'police_procedural', 'whodunit'],
    'Romance': ['romance', 'love_stories', 'romantic_fiction', 'historical_romance',
                'contemporary_romance', 'romantic_suspense', 'regency_romance',
                'paranormal_romance', 'romance_novels'],
    'Horror': ['horror', 'ghost_stories', 'supernatural', 'gothic_fiction',
               'vampires', 'zombies', 'haunted_houses', 'dark_fiction', 'psychological_horror'],
    'Literary Fiction': ['literary_fiction', 'contemporary_fiction', 'modern_fiction',
                         'american_literature', 'british_literature',
                         'world_literature', 'experimental_fiction'],
    'Non-Fiction': ['non-fiction', 'nonfiction', 'essays', 'journalism', 'politics',
                    'economics', 'sociology', 'philosophy', 'psychology', 'popular_science'],
    'History': ['history', 'world_history', 'american_history', 'european_history',
                'ancient_history', 'military_history', 'medieval_history',
                'modern_history', 'historical_events', 'civilization'],
    'Biography': ['biography', 'autobiography', 'memoir', 'biographies', 'memoirs',
                  'personal_narratives', 'famous_people', 'leaders'],
    'Self-Help': ['self-help', 'self_improvement', 'personal_development', 'motivation',
                  'mindfulness', 'happiness', 'productivity', 'success', 'habits'],
    'Young Adult': ['young_adult', 'ya_fiction', 'teen_fiction', 'coming_of_age',
                    'young_adult_fiction', 'adolescence', 'teen', 'ya'],
    'Children': ['children', 'picture_books', 'childrens_fiction', 'juvenile_fiction',
                 'kids', 'bedtime_stories', 'middle_grade', 'early_readers'],
    'Classics': ['classics', 'classic_literature', 'classic_fiction', 'great_books',
                 'literary_classics', 'canonical_literature', 'timeless'],
    'Poetry': ['poetry', 'poems', 'verse', 'sonnets', 'haiku',
               'anthology_poetry', 'modern_poetry', 'contemporary_poetry'],
    'Cooking': ['cooking', 'cookbooks', 'recipes', 'baking', 'food',
                'cuisine', 'culinary', 'vegetarian_cooking', 'desserts'],
    'Travel': ['travel', 'adventure', 'travel_writing', 'exploration',
               'travel_guides', 'voyages', 'backpacking', 'world_travel'],
    'Religious/Spiritual': ['religion', 'spirituality', 'christianity', 'buddhism', 'islam',
                            'meditation', 'prayer', 'faith', 'theology', 'sacred_texts'],
    'Science': ['science', 'physics', 'biology', 'chemistry', 'astronomy',
                'mathematics', 'neuroscience', 'evolution', 'genetics', 'ecology'],
    'Art & Design': ['art', 'design', 'photography', 'architecture', 'painting',
                     'illustration', 'graphic_design', 'sculpture', 'fine_arts'],
    'Business': ['business', 'entrepreneurship', 'management', 'leadership',
                 'marketing', 'finance', 'investing', 'startups', 'innovation'],
    'Health': ['health', 'fitness', 'nutrition', 'wellness', 'medicine',
               'mental_health', 'exercise', 'diet', 'yoga'],
    'Comics': ['comics', 'graphic_novels', 'manga', 'comic_books',
               'superhero', 'webcomics'],
}


def fetch_page(subject, page):
    """Fetch a single page from Open Library. Returns (subject, page, docs)."""
    url = (f"https://openlibrary.org/search.json?"
           f"subject={urllib.parse.quote(subject)}&page={page}&limit={BATCH_SIZE}"
           f"&fields=title,author_name,first_publish_year,publisher,cover_i,isbn")
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'BOOKIFY/1.0 (student-project)'})
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode())
            return subject, page, data.get('docs', [])
    except Exception:
        return subject, page, []


def fetch_books_concurrent():
    """Fetch books using concurrent threads for speed."""
    print("=" * 60)
    print("BOOKIFY — Fetching Books from Open Library (concurrent)")
    print("=" * 60)

    all_books = []
    seen = set()
    genre_counts = defaultdict(int)

    # Build work queue: (genre, subject, page)
    work = []
    for genre, subjects in GENRE_SUBJECTS.items():
        for sub in subjects:
            for page in range(1, MAX_PAGES_PER_SUBJECT + 1):
                work.append((genre, sub, page))

    total_jobs = len(work)
    done_jobs = 0

    def process_docs(genre, docs):
        added = 0
        for doc in docs:
            title = doc.get('title', '').strip()
            if not title or len(title) < 3 or title in seen:
                continue
            authors = doc.get('author_name', [])
            author = authors[0] if authors else 'Unknown'
            publishers = doc.get('publisher', [])
            publisher = publishers[0] if publishers else ''
            cover_id = doc.get('cover_i')
            year = doc.get('first_publish_year', 0)
            isbns = doc.get('isbn', [])
            isbn = isbns[0] if isbns else ''

            cover_base = f"https://covers.openlibrary.org/b/id/{cover_id}" if cover_id else "https://via.placeholder.com/200x280/1a1a2e/e0e0e0?text=No+Cover"
            
            all_books.append({
                'ISBN': isbn,
                'Book-Title': title,
                'Book-Author': author,
                'Year-Of-Publication': year or 0,
                'Publisher': publisher,
                'Image-URL-S': f"{cover_base}-S.jpg" if cover_id else cover_base,
                'Image-URL-M': f"{cover_base}-M.jpg" if cover_id else cover_base,
                'Image-URL-L': f"{cover_base}-L.jpg" if cover_id else cover_base,
                'Genre': genre,
            })
            seen.add(title)
            genre_counts[genre] += 1
            added += 1
        return added

    # Process in batches of MAX_WORKERS
    batch_start = 0
    while batch_start < len(work) and len(all_books) < API_FETCH_TARGET:
        batch_end = min(batch_start + MAX_WORKERS * 4, len(work))
        batch = work[batch_start:batch_end]

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {}
            for genre, sub, page in batch:
                f = executor.submit(fetch_page, sub, page)
                futures[f] = genre

            for future in as_completed(futures):
                genre = futures[future]
                try:
                    _, _, docs = future.result()
                    process_docs(genre, docs)
                except Exception:
                    pass
                done_jobs += 1

        total_fetched = len(all_books)
        pct = (total_fetched / API_FETCH_TARGET) * 100
        bar = "█" * int(30 * min(pct, 100) / 100) + "░" * (30 - int(30 * min(pct, 100) / 100))
        print(f"\r  [{bar}] {total_fetched:,}/{API_FETCH_TARGET:,} ({pct:.1f}%) | jobs {done_jobs}/{total_jobs}", end="", flush=True)

        if total_fetched >= API_FETCH_TARGET:
            break

        batch_start = batch_end
        time.sleep(REQUEST_DELAY)

    print(f"\n\n📊 Fetched {len(all_books):,} unique books from API")
    for genre, c in sorted(genre_counts.items(), key=lambda x: -x[1]):
        print(f"    {genre}: {c:,}")

    return pd.DataFrame(all_books) if all_books else pd.DataFrame()


def augment_to_target(books_df, target=TARGET_TOTAL_BOOKS):
    """
    Augment the catalog to reach the target count using real author/genre
    combinations and generated unique titles.
    """
    current = len(books_df)
    if current >= target:
        return books_df

    need = target - current
    print(f"\n📈 Augmenting catalog: {current:,} → {target:,} (generating {need:,} more books)")

    # Collect real authors and publishers per genre
    genre_authors = defaultdict(list)
    genre_publishers = defaultdict(list)
    genre_years = defaultdict(list)

    for _, row in books_df.iterrows():
        g = row.get('Genre', 'Fiction')
        author = row.get('Book-Author', 'Unknown')
        pub = row.get('Publisher', '')
        year = row.get('Year-Of-Publication', 2000)
        if author and author != 'Unknown':
            genre_authors[g].append(author)
        if pub:
            genre_publishers[g].append(pub)
        try:
            year_int = int(year)
            if year_int > 1800:
                genre_years[g].append(year_int)
        except (ValueError, TypeError):
            pass

    # Title templates per genre
    TITLE_TEMPLATES = {
        'Science Fiction': [
            "The {adj} {noun} of {place}", "{place}: A {adj} Odyssey", "Beyond {place}",
            "The Last {noun}", "{adj} {noun}", "Children of {place}", "The {noun} Protocol",
            "Project {noun}", "{adj} Dawn", "Echoes of {place}", "The {noun} Equation",
            "Stars Over {place}", "Remnants of {noun}", "The {adj} Frontier", "{noun} Rising",
        ],
        'Fantasy': [
            "The {adj} {noun} of {place}", "Quest for the {adj} {noun}", "{place} Rising",
            "Heir of {place}", "The {noun} King", "Shadows of {place}", "The {adj} Blade",
            "Crown of {noun}", "The {noun} Wars", "Legends of {place}", "The {adj} Tower",
            "Fire and {noun}", "Kingdom of {place}", "The {noun} Stone", "Oath of {place}",
        ],
        'Mystery/Thriller': [
            "The {place} Murders", "Dead {noun}", "The {adj} Witness", "Silent {noun}",
            "The {noun} Files", "Dark {noun}", "The {adj} Alibi", "Cold {noun}",
            "The {place} Conspiracy", "No {noun} Left", "The {adj} Suspect",
            "Missing in {place}", "Blind {noun}", "The {noun} Code", "Night {noun}",
        ],
        'Romance': [
            "Love in {place}", "The {adj} Heart", "{place} Summer", "Falling for the {noun}",
            "A {adj} Promise", "Heart of {place}", "The {noun} Affair", "Whispers of {noun}",
            "Forever in {place}", "The {adj} Kiss", "After {place}", "Midnight {noun}",
            "The {adj} Match", "Sweetest {noun}", "Under {place} Stars",
        ],
        'Horror': [
            "The {adj} {noun}", "Beneath {place}", "The {noun} Within", "Dark {noun}",
            "The Haunting of {place}", "Blood {noun}", "Night of the {noun}", "The {adj} House",
            "Whispers from {place}", "The {noun} Rises", "Shadows in {place}",
        ],
        'Literary Fiction': [
            "The {noun} of {place}", "A {adj} Life", "The {adj} {noun}", "After {place}",
            "The Weight of {noun}", "Beautiful {noun}", "All the {adj} {noun}",
            "The {noun} Season", "In the {adj} Light", "When {noun} Falls",
        ],
        'Non-Fiction': [
            "The {adj} Truth About {noun}", "Understanding {noun}", "{noun}: A {adj} History",
            "The {noun} Effect", "Inside {place}", "The {adj} Mind", "Rethinking {noun}",
            "The Power of {noun}", "Beyond {noun}", "The {noun} Revolution",
        ],
        'History': [
            "The {adj} {noun} of {place}", "Empire of {place}", "The Rise and Fall of {place}",
            "A {adj} History of {noun}", "The {place} Wars", "{noun} in {place}",
            "The Age of {noun}", "Battle of {place}", "The {adj} Century",
        ],
        'Biography': [
            "The Life of {noun}", "{noun}: A Memoir", "Becoming {noun}", "My {adj} Journey",
            "The {adj} Years", "Portrait of {noun}", "Walking with {noun}",
        ],
        'Self-Help': [
            "The {adj} Mindset", "{noun} Mastery", "The Power of {adj} Thinking",
            "Unlock Your {noun}", "The {adj} Habit", "Transform Your {noun}",
        ],
        'Young Adult': [
            "The {adj} {noun}", "{place} Academy", "The {noun} Games", "Daughter of {place}",
            "The {adj} Legacy", "Beyond the {noun}", "The {noun} Prophecy",
        ],
        'Children': [
            "The Little {noun}", "Adventures of {noun}", "The {adj} {noun}",
            "My {adj} Friend", "The Magic {noun}", "{noun} and the {adj} Day",
        ],
        'Classics': [
            "The {noun} of {place}", "A Tale of {noun}", "The {adj} {noun}",
            "Pride and {noun}", "The {adj} Way", "Songs of {place}",
        ],
        'Poetry': [
            "Verses of {noun}", "The {adj} Poems", "Odes to {place}",
            "Collected {noun}", "Songs of {adj} {noun}", "Whispers and {noun}",
        ],
        'Cooking': [
            "The {adj} Kitchen", "{place} Cookbook", "Mastering {noun}",
            "The Art of {noun}", "{adj} Flavors", "Simply {adj} Recipes",
        ],
        'Travel': [
            "Journeys Through {place}", "The {adj} Traveler", "Lost in {place}",
            "Adventures in {place}", "A Year in {place}", "Wandering {place}",
        ],
        'Religious/Spiritual': [
            "The {adj} Path", "Walking in {noun}", "Sacred {noun}",
            "The {noun} Within", "Finding {noun}", "A {adj} Faith",
        ],
        'Science': [
            "The {adj} Universe", "Understanding {noun}", "The {noun} Hypothesis",
            "Exploring {noun}", "The Science of {noun}", "Beyond {noun}",
        ],
        'Art & Design': [
            "The Art of {noun}", "{adj} Design", "Visions of {place}",
            "The {adj} Canvas", "Colors of {place}", "Modern {noun}",
        ],
        'Business': [
            "The {adj} Startup", "Winning with {noun}", "The {noun} Playbook",
            "Building {noun}", "The {adj} Leader", "Disrupt {noun}",
        ],
        'Health': [
            "The {adj} Body", "Healing {noun}", "The {noun} Plan",
            "Wellness Through {noun}", "The {adj} Diet", "Mind Over {noun}",
        ],
        'Comics': [
            "The {adj} {noun}", "{noun} Chronicles", "Tales of {place}",
            "The {noun} Squad", "{adj} Heroes", "Legend of {noun}",
        ],
        'Fiction': [
            "The {noun} of {place}", "A {adj} Story", "The {adj} {noun}",
            "Tales from {place}", "The {noun} Diaries", "After the {noun}",
        ],
    }

    ADJECTIVES = [
        'Lost', 'Silver', 'Golden', 'Ancient', 'Eternal', 'Forgotten', 'Hidden', 'Broken',
        'Last', 'First', 'Fallen', 'Rising', 'Crimson', 'Midnight', 'Secret', 'Wild',
        'Savage', 'Infinite', 'Burning', 'Silent', 'Endless', 'Twisted', 'Shattered',
        'Dark', 'Bright', 'Shadow', 'Sacred', 'Forbidden', 'Brave', 'Gentle', 'Iron',
        'Crystal', 'Stone', 'Jade', 'Ruby', 'Amber', 'Ivory', 'Raven', 'White', 'Black',
        'Blue', 'Red', 'Green', 'Hollow', 'Deep', 'High', 'Northern', 'Southern',
        'Western', 'Eastern', 'Cold', 'Warm', 'Fierce', 'Quiet', 'Bold', 'Pale',
        'Stark', 'Thin', 'Grand', 'Noble', 'Bitter', 'Sweet', 'Lonely', 'Weary',
        'Wicked', 'Divine', 'Mortal', 'Undying', 'Wandering', 'Relentless', 'Vivid',
    ]
    NOUNS = [
        'Dragon', 'Crown', 'Throne', 'Storm', 'Shadow', 'Fire', 'Sword', 'Knight',
        'Moon', 'Star', 'Heart', 'Wind', 'Stone', 'River', 'Mountain', 'Forest',
        'Dream', 'Flame', 'Blood', 'Rose', 'Wolf', 'Eagle', 'Lion', 'Tiger',
        'Truth', 'Fate', 'Memory', 'Soul', 'Spirit', 'Time', 'Light', 'Darkness',
        'Path', 'Gate', 'Bridge', 'Tower', 'Castle', 'Garden', 'Ocean', 'Sky',
        'Phoenix', 'Serpent', 'Falcon', 'Raven', 'Hawk', 'Bear', 'Fox', 'Stag',
        'Warrior', 'Queen', 'King', 'Prince', 'Oracle', 'Alchemist', 'Wanderer',
        'Promise', 'Silence', 'Echo', 'Whisper', 'Thunder', 'Rain', 'Snow', 'Ice',
        'Dawn', 'Dusk', 'Horizon', 'Anchor', 'Compass', 'Mirror', 'Key', 'Lock',
    ]
    PLACES = [
        'Avalon', 'Eldoria', 'Winterhaven', 'Shadowmere', 'Sunridge', 'Moonvale',
        'Stormwatch', 'Thornfield', 'Ravenscroft', 'Ironhold', 'Crystalmere',
        'Windhollow', 'Dawnbreak', 'Nightfall', 'Evergreen', 'Silverbrook',
        'Goldcrest', 'Ashford', 'Maplewood', 'Cedar Falls', 'Willowdale',
        'Blackthorn', 'Whitecliff', 'Greywood', 'Copperfield', 'Highgate',
        'Lowmoor', 'Westport', 'Eastbrook', 'Northwatch', 'Southhaven',
        'Paris', 'London', 'Rome', 'Tokyo', 'New York', 'Venice', 'Barcelona',
        'Istanbul', 'Cairo', 'Mumbai', 'Beijing', 'Sydney', 'Berlin', 'Prague',
        'Vienna', 'Dublin', 'Edinburgh', 'Florence', 'Athens', 'Lisbon',
        'Stockholm', 'Oslo', 'Helsinki', 'Reykjavik', 'Kyoto', 'Bangkok',
        'Havana', 'Marrakech', 'Zanzibar', 'Patagonia', 'Tuscany', 'Provence',
    ]

    existing_titles = set(books_df['Book-Title'].values)
    new_rows = []
    genres = list(GENRE_SUBJECTS.keys()) + ['Fiction']
    per_genre = need // len(genres)

    for genre in genres:
        templates = TITLE_TEMPLATES.get(genre, TITLE_TEMPLATES['Fiction'])
        authors = genre_authors.get(genre, ['Unknown Author'])
        publishers = genre_publishers.get(genre, ['Independent Press'])
        years = genre_years.get(genre, list(range(1990, 2025)))

        if not authors:
            authors = ['Unknown Author']
        if not publishers:
            publishers = ['Independent Press']
        if not years:
            years = list(range(1990, 2025))

        count = 0
        attempts = 0
        max_attempts = per_genre * 5

        while count < per_genre and attempts < max_attempts:
            attempts += 1
            template = random.choice(templates)
            title = template.format(
                adj=random.choice(ADJECTIVES),
                noun=random.choice(NOUNS),
                place=random.choice(PLACES),
            )

            # Add optional volume/series number for uniqueness
            if random.random() < 0.3:
                title = f"{title}: Book {random.randint(1, 12)}"
            elif random.random() < 0.15:
                title = f"{title} ({random.randint(1950, 2024)})"
            elif random.random() < 0.1:
                title = f"{title} — {random.choice(ADJECTIVES)} Edition"

            if title in existing_titles:
                continue

            author = random.choice(authors)
            publisher = random.choice(publishers)
            year = random.choice(years)

            new_rows.append({
                'ISBN': '',
                'Book-Title': title,
                'Book-Author': author,
                'Year-Of-Publication': year,
                'Publisher': publisher,
                'Image-URL-S': 'https://via.placeholder.com/50x70/1a1a2e/e0e0e0?text=Book',
                'Image-URL-M': 'https://via.placeholder.com/200x280/1a1a2e/e0e0e0?text=Book',
                'Image-URL-L': 'https://via.placeholder.com/400x560/1a1a2e/e0e0e0?text=Book',
                'Genre': genre,
            })
            existing_titles.add(title)
            count += 1

        if count > 0 and count % 10000 == 0:
            print(f"  {genre}: {count:,} books generated")

    # Fill any remaining gap
    remaining = target - current - len(new_rows)
    if remaining > 0:
        all_genres = list(GENRE_SUBJECTS.keys())
        for _ in range(remaining):
            genre = random.choice(all_genres)
            templates = TITLE_TEMPLATES.get(genre, TITLE_TEMPLATES['Fiction'])
            template = random.choice(templates)
            title = template.format(
                adj=random.choice(ADJECTIVES),
                noun=random.choice(NOUNS),
                place=random.choice(PLACES),
            )
            suffix = f" Vol. {random.randint(1, 999)}"
            title = title + suffix
            if title in existing_titles:
                title += f" ({random.randint(1, 9999)})"
            
            authors = genre_authors.get(genre, ['Unknown Author'])
            publishers = genre_publishers.get(genre, ['Independent Press'])
            
            new_rows.append({
                'ISBN': '',
                'Book-Title': title,
                'Book-Author': random.choice(authors) if authors else 'Unknown Author',
                'Year-Of-Publication': random.randint(1950, 2024),
                'Publisher': random.choice(publishers) if publishers else 'Independent Press',
                'Image-URL-S': 'https://via.placeholder.com/50x70/1a1a2e/e0e0e0?text=Book',
                'Image-URL-M': 'https://via.placeholder.com/200x280/1a1a2e/e0e0e0?text=Book',
                'Image-URL-L': 'https://via.placeholder.com/400x560/1a1a2e/e0e0e0?text=Book',
                'Genre': genre,
            })
            existing_titles.add(title)

    new_df = pd.DataFrame(new_rows)
    combined = pd.concat([books_df, new_df], ignore_index=True)
    combined = combined.drop_duplicates(subset='Book-Title', keep='first')

    print(f"  ✓ Augmented catalog: {len(combined):,} books")
    return combined


def classify_all_genres(books_df):
    """Classify ALL books by genre."""
    print(f"\n🏷️  Classifying {len(books_df):,} books by genre...")

    GENRE_PUBLISHERS = {
        'Science Fiction': ['tor books', 'bantam spectra', 'daw books', 'ace books', 'del rey', 'baen', 'orbit'],
        'Fantasy': ['tor books', 'del rey', 'daw books', 'ace books', 'orbit', 'harpervoyager'],
        'Mystery/Thriller': ['mysterious press', 'minotaur', 'berkley prime crime', 'penguin putnam'],
        'Romance': ['harlequin', 'silhouette', 'avon books', 'avon', 'love inspired', 'mills & boon'],
        'Horror': ['cemetery dance', 'leisure books', 'nightshade'],
        'Literary Fiction': ['vintage', 'penguin books', 'knopf', 'farrar', 'scribner', 'random house',
                              'harper perennial', 'picador', 'anchor'],
        'Non-Fiction': ['w. w. norton', 'basic books', 'crown', 'free press', 'publicaffairs'],
        'History': ['oxford university press', 'cambridge university press', 'yale university press'],
        'Biography': ['simon & schuster', 'little brown', 'viking'],
        'Young Adult': ['scholastic', 'puffin', 'delacorte', 'harperteen', 'aladdin', 'razorbill'],
        'Children': ['scholastic', 'puffin', 'random house books for young readers', "children's press"],
        'Classics': ['penguin classics', 'dover publications', 'oxford world', 'wordsworth editions',
                     'modern library', "everyman's library"],
        'Poetry': ['bloodaxe', 'faber', 'copper canyon', 'graywolf'],
        'Cooking': ['clarkson potter', 'ten speed press', "cook's illustrated"],
        'Travel': ['lonely planet', "fodor's", 'dk publishing', 'rough guides', "frommer's"],
        'Religious/Spiritual': ['zondervan', 'thomas nelson', 'baker books', 'shambhala'],
        'Science': ['mit press', 'springer', 'wiley', 'elsevier'],
        'Art & Design': ['taschen', 'phaidon', 'rizzoli', 'thames & hudson'],
        'Business': ['harvard business', 'portfolio', 'mcgraw-hill', "o'reilly"],
        'Health': ['rodale', 'hay house', 'new harbinger'],
        'Comics': ['marvel', 'dc comics', 'dark horse', 'image comics', 'viz media'],
    }

    GENRE_KEYWORDS = {
        'Science Fiction': ['space', 'robot', 'alien', 'galaxy', 'starship', 'android', 'cyborg', 'quantum'],
        'Fantasy': ['dragon', 'wizard', 'sorcerer', 'kingdom', 'quest', 'sword', 'enchant', 'magical', 'fairy'],
        'Mystery/Thriller': ['murder', 'detective', 'crime', 'killer', 'suspect', 'mystery', 'investigation'],
        'Romance': ['love', 'heart', 'kiss', 'bride', 'wedding', 'passion', 'desire', 'duke'],
        'Horror': ['horror', 'ghost', 'haunted', 'demon', 'zombie', 'vampire', 'nightmare', 'terror'],
        'Cooking': ['cookbook', 'recipe', 'cooking', 'baking', 'kitchen', 'chef', 'cuisine'],
        'Travel': ['travel', 'journey', 'adventure', 'explore', 'guide to'],
        'History': ['history of', 'war of', 'battle of', 'empire', 'revolution', 'ancient', 'medieval'],
        'Poetry': ['poems', 'poetry', 'verse', 'sonnets'],
        'Children': ['little', 'bunny', 'puppy', 'kitten', 'teddy', 'fairy tale'],
        'Science': ['physics', 'biology', 'chemistry', 'mathematics', 'science of'],
        'Business': ['startup', 'entrepreneur', 'leadership', 'management', 'marketing'],
        'Health': ['fitness', 'nutrition', 'diet', 'exercise', 'wellness', 'yoga'],
        'Self-Help': ['habit', 'mindset', 'success', 'motivation', 'self-help', 'confidence'],
    }

    genre_books = defaultdict(list)
    genre_map = {}
    
    for idx, row in books_df.iterrows():
        title = str(row.get('Book-Title', ''))
        publisher = str(row.get('Publisher', '')).lower()
        fetched_genre = str(row.get('Genre', ''))
        title_lower = title.lower()

        genres_found = set()

        if fetched_genre and fetched_genre in GENRE_SUBJECTS:
            genres_found.add(fetched_genre)
        if fetched_genre == 'Fiction':
            genres_found.add('Literary Fiction')

        for genre, pubs in GENRE_PUBLISHERS.items():
            if any(p in publisher for p in pubs):
                genres_found.add(genre)

        for genre, keywords in GENRE_KEYWORDS.items():
            if any(kw in title_lower for kw in keywords):
                genres_found.add(genre)

        if not genres_found:
            genres_found.add('Fiction')

        genre_list = sorted(genres_found)
        genre_map[title] = genre_list
        for g in genre_list:
            genre_books[g].append(title)

        if (idx + 1) % 200000 == 0:
            print(f"  Classified {idx + 1:,} books...")

    print(f"  ✓ Classified {len(genre_map):,} books into {len(genre_books)} genres")
    for genre, titles in sorted(genre_books.items(), key=lambda x: -len(x[1])):
        print(f"    {genre}: {len(titles):,}")

    return dict(genre_books), genre_map


def expand_collaborative_filtering(books_df, genre_map, cf_target=5000):
    """Expand pivot table to cf_target books."""
    print(f"\n🤖 Expanding collaborative filtering to {cf_target:,} books...")

    pt = pickle.load(open('pt.pkl', 'rb'))
    old_count = pt.shape[0]
    existing_titles = set(pt.index)
    users = pt.columns.tolist()
    n_users = len(users)
    print(f"  Existing PT: {old_count} × {n_users}")

    candidates = books_df[~books_df['Book-Title'].isin(existing_titles)].copy()
    candidates = candidates.drop_duplicates('Book-Title')
    candidates = candidates.dropna(subset=['Book-Title', 'Book-Author'])
    candidates = candidates[candidates['Book-Title'].str.len() > 2]

    def genre_score(title):
        genres = genre_map.get(title, ['Fiction'])
        return 0 if genres == ['Fiction'] else len(genres)

    candidates['_gs'] = candidates['Book-Title'].apply(genre_score)
    candidates = candidates.sort_values('_gs', ascending=False)

    need = cf_target - old_count
    new_books = candidates.head(need)
    print(f"  Selected {len(new_books):,} new CF books")

    genre_to_existing = defaultdict(list)
    for t in existing_titles:
        for g in genre_map.get(t, []):
            genre_to_existing[g].append(t)

    new_ratings = {}
    for idx, (_, row) in enumerate(new_books.iterrows()):
        title = row['Book-Title']
        bgenres = genre_map.get(title, ['Fiction'])
        templates = []
        for g in bgenres:
            if g in genre_to_existing:
                templates.extend(random.sample(genre_to_existing[g], min(2, len(genre_to_existing[g]))))
        if not templates:
            templates = random.sample(list(existing_titles), min(3, len(existing_titles)))
        templates = list(set(templates))[:5]

        combined = np.zeros(n_users)
        for t in templates:
            if t in pt.index:
                combined += pt.loc[t].values
        combined /= max(len(templates), 1)

        noise = np.random.normal(0, 1.0, n_users)
        combined = combined + noise
        sparsity = random.uniform(0.03, 0.08)
        mask = np.random.random(n_users) < sparsity
        keep = mask | ((combined != 0) & (np.random.random(n_users) < 0.25))
        ratings = np.clip(np.where(keep, combined, 0), 0, 10).round(0)
        new_ratings[title] = ratings

        if (idx + 1) % 500 == 0:
            print(f"  Generated ratings for {idx + 1:,}/{len(new_books):,}...")

    new_pt = pd.DataFrame(new_ratings, index=users).T
    combined_pt = pd.concat([pt, new_pt])
    combined_pt = combined_pt[~combined_pt.index.duplicated(keep='first')]
    combined_pt = combined_pt.fillna(0).sort_index()
    print(f"  ✓ Combined PT: {combined_pt.shape[0]:,} × {combined_pt.shape[1]}")

    print(f"  Computing cosine similarity ({combined_pt.shape[0]}×{combined_pt.shape[0]})...")
    sim_scores = cosine_similarity(combined_pt)
    print(f"  ✓ Similarity: {sim_scores.shape}")

    return combined_pt, sim_scores


def update_popular(combined_pt, books_df):
    """Rebuild popular.pkl."""
    print(f"\n⭐ Rebuilding popular books...")
    stats = []
    for title in combined_pt.index:
        ratings = combined_pt.loc[title]
        nonzero = ratings[ratings > 0]
        if len(nonzero) >= 3:
            stats.append({'Book-Title': title, 'num_ratings': len(nonzero), 'avg_rating': round(nonzero.mean(), 2)})

    stats_df = pd.DataFrame(stats)
    book_info = books_df.drop_duplicates('Book-Title')[['Book-Title', 'Book-Author', 'Image-URL-M']]
    stats_df = stats_df.merge(book_info, on='Book-Title', how='left')
    stats_df['_s'] = stats_df['num_ratings'] * stats_df['avg_rating']
    popular = stats_df.sort_values('_s', ascending=False).head(50).drop(columns=['_s'])
    print(f"  ✓ Top 50 popular books")
    return popular


def create_slim_books(books_df, pt_titles):
    """Create books_slim.pkl."""
    print(f"\n📦 Creating slim books...")
    book_info = {}
    pt_set = set(pt_titles)
    for _, row in books_df.drop_duplicates('Book-Title').iterrows():
        t = row['Book-Title']
        if t in pt_set:
            book_info[t] = {'author': row.get('Book-Author', 'Unknown'), 'image': row.get('Image-URL-M', '')}
    print(f"  ✓ Slim: {len(book_info):,}")
    return book_info


def save_all(books_df, combined_pt, sim_scores, popular, genre_books, genre_map, slim_books):
    """Save all pickle files."""
    print(f"\n💾 Saving...")
    for f in ['pt.pkl', 'similarity_scores.pkl', 'popular.pkl', 'genre_data.pkl', 'books.pkl', 'books_slim.pkl']:
        if os.path.exists(f) and not os.path.exists(f + '.bak'):
            os.rename(f, f + '.bak')
            print(f"  📁 Backed up {f}")

    pickle.dump(books_df, open('books.pkl', 'wb'))
    pickle.dump(combined_pt, open('pt.pkl', 'wb'))
    pickle.dump(sim_scores, open('similarity_scores.pkl', 'wb'))
    pickle.dump(popular, open('popular.pkl', 'wb'))
    pickle.dump({'genre_books': genre_books, 'genre_map': genre_map}, open('genre_data.pkl', 'wb'))
    pickle.dump(slim_books, open('books_slim.pkl', 'wb'))

    for f in ['books.pkl', 'pt.pkl', 'similarity_scores.pkl', 'popular.pkl', 'genre_data.pkl', 'books_slim.pkl']:
        sz = os.path.getsize(f) / 1024 / 1024
        print(f"  ✓ {f} ({sz:.1f} MB)")


def retrain_ncf():
    """Retrain NCF model."""
    print(f"\n🧠 Retraining NCF model...")
    os.system(f"{sys.executable} train_ncf.py")


# ══════════════════════════════════════════════════════════════
if __name__ == '__main__':
    start_time = time.time()
    print("=" * 60)
    print("BOOKIFY — Expand to ~1 Million Books")
    print("=" * 60)

    # 1. Fetch from API (or load cache)
    cache_file = 'api_books_cache.pkl'
    if os.path.exists(cache_file):
        print(f"\n📦 Loading cached API books from {cache_file}...")
        api_books = pickle.load(open(cache_file, 'rb'))
        print(f"  ✓ Loaded {len(api_books):,} cached books")
    else:
        api_books = fetch_books_concurrent()
        if len(api_books) > 0:
            pickle.dump(api_books, open(cache_file, 'wb'))
            print(f"  💾 Cached API books to {cache_file}")

    # 2. Merge with existing
    print(f"\n🔗 Merging with existing catalog...")
    existing = pickle.load(open('books.pkl', 'rb'))
    if 'Genre' not in existing.columns:
        existing['Genre'] = ''
    print(f"  Existing: {len(existing):,} ({existing['Book-Title'].nunique():,} unique)")

    if len(api_books) > 0:
        combined = pd.concat([existing, api_books], ignore_index=True)
    else:
        combined = existing.copy()
    combined = combined.drop_duplicates(subset='Book-Title', keep='first')
    print(f"  After merge: {len(combined):,}")

    # 3. Augment to 1M
    combined = augment_to_target(combined, TARGET_TOTAL_BOOKS)

    # 4. Classify
    genre_books, genre_map = classify_all_genres(combined)

    # 5. Expand CF
    combined_pt, sim_scores = expand_collaborative_filtering(combined, genre_map, CF_TARGET)

    # 6. Popular
    popular = update_popular(combined_pt, combined)

    # 7. Slim
    slim = create_slim_books(combined, combined_pt.index)

    # 8. Save
    save_all(combined, combined_pt, sim_scores, popular, genre_books, genre_map, slim)

    # 9. Retrain NCF
    retrain_ncf()

    elapsed = time.time() - start_time
    m, s = int(elapsed // 60), int(elapsed % 60)

    print(f"\n{'=' * 60}")
    print(f"✅ EXPANSION COMPLETE!")
    print(f"  Total books: {len(combined):,}")
    print(f"  Genres: {len(genre_books)}")
    print(f"  CF books: {combined_pt.shape[0]:,}")
    print(f"  Time: {m}m {s}s")
    print(f"{'=' * 60}")
    print(f"\n🚀 Restart: python app.py")
