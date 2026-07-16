"""Resumably upload a large artifact tree to the public dataset repo."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi, RepoUrl


class ExistingRepoApi(HfApi):
    """Skip upload_large_folder's redundant repo-create request.

    The target repo is created and anonymously verified during prerequisite
    setup. Hugging Face's large-folder helper otherwise calls create_repo even
    with exist_ok=True, making uploads depend on an unrelated endpoint.
    """

    def create_repo(self, repo_id: str, **kwargs) -> RepoUrl:  # type: ignore[override]
        return RepoUrl(f"https://huggingface.co/datasets/{repo_id}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    api = ExistingRepoApi()
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
