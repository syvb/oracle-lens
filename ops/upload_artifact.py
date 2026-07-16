"""Upload one artifact file to the public oracle-lens dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("local_path", type=Path)
    parser.add_argument("path_in_repo")
    parser.add_argument("--message", default="Upload oracle-lens artifact")
    args = parser.parse_args()
    result = HfApi().upload_file(
        path_or_fileobj=args.local_path,
        path_in_repo=args.path_in_repo,
        repo_id="syvb/oracle-lens-qwen3-8b-artifacts",
        repo_type="dataset",
        commit_message=args.message,
    )
    print(result)


if __name__ == "__main__":
    main()
