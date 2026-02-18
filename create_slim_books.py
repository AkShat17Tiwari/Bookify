"""
One-time script to create books_slim.pkl — a lightweight dict
containing only the book info needed by the app.

Replaces the 166MB books.pkl DataFrame with a <1MB dict.
"""
import pickle
import sys

# Load the original data
with open('books.pkl', 'rb') as f:
    books = pickle.load(f)
with open('pt.pkl', 'rb') as f:
    pt = pickle.load(f)

# Get only the titles that exist in the pivot table
pt_titles = set(pt.index)

# Build slim lookup: {title: {'author': ..., 'image': ...}}
slim = books[books['Book-Title'].isin(pt_titles)].drop_duplicates('Book-Title')
books_slim = {}
for _, row in slim.iterrows():
    books_slim[row['Book-Title']] = {
        'author': row['Book-Author'],
        'image': row['Image-URL-M'],
    }

# Save
with open('books_slim.pkl', 'wb') as f:
    pickle.dump(books_slim, f)

# Report
orig_size = sys.getsizeof(books.values) + sum(books.memory_usage(deep=True))
slim_size = sys.getsizeof(books_slim)
print(f"Original books.pkl: {len(books)} rows, ~{orig_size / 1024 / 1024:.1f} MB in memory")
print(f"Slim books_slim.pkl: {len(books_slim)} entries, ~{slim_size / 1024 / 1024:.1f} MB in memory")
print("✅ books_slim.pkl created successfully!")
