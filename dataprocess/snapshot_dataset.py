from huggingface_hub import snapshot_download
from multiprocessing import cpu_count
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Download Huggingface dataset snapshot with retry.")
    parser.add_argument('--repo_name', type=str, required=True, help='Huggingface dataset repo name')
    parser.add_argument('--local_dir', type=str, required=True, help='Local directory to store the dataset')
    args = parser.parse_args()
    repo_name = args.repo_name
    local_dir = args.local_dir.rstrip('/')  # Remove trailing slash if present
    cache_dir = os.path.join(local_dir, "cache")
    os.makedirs(local_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    while True:
        try:
            print("Attempting to download snapshot...")
            snapshot_download(
                repo_id=repo_name,
                repo_type="dataset",
                cache_dir=cache_dir,
                local_dir=local_dir,
                local_dir_use_symlinks=False,
                force_download=False,
                max_workers=cpu_count(),
            )
            print("Snapshot downloaded successfully.")
            break  # Exit loop after successful download
        except Exception as e:
            print(f"Download interrupted: {e}")
            print("Retrying...")
if __name__ == "__main__":
    main()