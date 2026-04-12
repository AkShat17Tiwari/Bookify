import os
from huggingface_hub import HfApi

token = os.environ.get("HF_TOKEN", "")
repo_id = "Akshat200343/bookify"
api = HfApi()

base_dir = "/Users/akshat_tiwari/Documents/brs/book-recommender-system-master"

print("Uploading to Hugging Face...")

# Upload specific files
files_to_upload = ["app.py", "auth.py", "security.py", "Dockerfile"]
for file in files_to_upload:
    path = os.path.join(base_dir, file)
    print(f"Uploading {file}...")
    api.upload_file(
        path_or_fileobj=path,
        path_in_repo=file,
        repo_id=repo_id,
        repo_type="space",
        token=token,
        commit_message=f"Update {file} with React frontend integration"
    )

# Upload the frontend directory
print("Uploading frontend directory...")
api.upload_folder(
    folder_path=os.path.join(base_dir, "frontend"),
    path_in_repo="frontend",
    repo_id=repo_id,
    repo_type="space",
    token=token,
    commit_message="Add React frontend implementation",
    ignore_patterns=["node_modules/*", "dist/*", ".env_staging", ".env.local"]
)

print("✅ Upload complete!")
