"""Resumably upload a large artifact tree to the public dataset repo."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    api = HfApi()
    api.upload_large_folder(
        repo_id="syvb/oracle-lens-qwen3-8b-artifacts",
        repo_type="dataset",
        folder_path=args.folder,
        private=False,
        num_workers=args.workers,
        print_report=True,
        print_report_every=30,
    )


if __name__ == "__main__":
    main()
