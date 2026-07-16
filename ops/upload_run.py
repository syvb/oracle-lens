"""Upload a completed run subtree to the public Hugging Face artifact dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    parser.add_argument("--repo-id", default="syvb/oracle-lens-qwen3-8b-artifacts")
    parser.add_argument("--path-in-repo", default="qwen3-8b-v1")
    args = parser.parse_args()

    api = HfApi()
    api.create_repo(args.repo_id, repo_type="dataset", private=False, exist_ok=True)
    api.update_repo_settings(args.repo_id, repo_type="dataset", private=False)
    result = api.upload_folder(
        repo_id=args.repo_id,
        repo_type="dataset",
        folder_path=args.folder,
        path_in_repo=args.path_in_repo,
        commit_message=f"Upload {args.path_in_repo} artifacts",
    )
    print(result)


if __name__ == "__main__":
    main()
