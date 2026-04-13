import os
from huggingface_hub import HfApi

token = os.environ.get("HF_TOKEN", "")
repo_id = "Akshat200343/bookify"
api = HfApi()

base_dir = "/Users/akshat_tiwari/Documents/brs/book-recommender-system-master"

print("Uploading to Hugging Face...")

# All required backend files
files_to_upload = [
    "app.py",
    "auth.py",
    "security.py",
    "cover_analyzer.py",
    "Dockerfile",
    "requirements.txt",
    "render.yaml",
    ".gitattributes",
    "README.md",
    # Data files
    "popular.pkl",
    "books_slim.pkl",
    "genre_data.pkl",
    "model_accuracy.json",
    "classic_accuracy.json",
]

# Upload individual files
for file in files_to_upload:
    path = os.path.join(base_dir, file)
    if not os.path.exists(path):
        print(f"  SKIP {file} (not found)")
        continue
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"  Uploading {file} ({size_mb:.1f} MB)...")
    try:
        api.upload_file(
            path_or_fileobj=path,
            path_in_repo=file,
            repo_id=repo_id,
            repo_type="space",
            token=token,
            commit_message=f"Update {file}"
        )
    except Exception as e:
        print(f"  ERROR uploading {file}: {e}")

# Upload large model files (Git LFS)
large_files = [
    "pt.pkl",
    "similarity_scores.pkl",
    "ncf_similarity_scores.pkl",
    "ncf_book_embeddings.pkl",
]
for file in large_files:
    path = os.path.join(base_dir, file)
    if not os.path.exists(path):
        print(f"  SKIP {file} (not found)")
        continue
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"  Uploading {file} ({size_mb:.1f} MB) via LFS...")
    try:
        api.upload_file(
            path_or_fileobj=path,
            path_in_repo=file,
            repo_id=repo_id,
            repo_type="space",
            token=token,
            commit_message=f"Update {file} (LFS)"
        )
    except Exception as e:
        print(f"  ERROR uploading {file}: {e}")

# Upload the frontend directory (source + dist)
print("Uploading frontend directory...")
try:
    api.upload_folder(
        folder_path=os.path.join(base_dir, "frontend"),
        path_in_repo="frontend",
        repo_id=repo_id,
        repo_type="space",
        token=token,
        commit_message="Update React frontend",
        ignore_patterns=[
            "node_modules/*",
            ".env",
            ".env.local",
            ".env_staging",
        ]
    )
except Exception as e:
    print(f"  ERROR uploading frontend: {e}")

print("\nUpload complete!")
