"""Pull the exact public M0 artifacts required by M1."""

from huggingface_hub import snapshot_download


snapshot_download(
    repo_id="syvb/oracle-lens-qwen3-8b-artifacts",
    repo_type="dataset",
    local_dir="/workspace/runs",
    allow_patterns=[
        "qwen3-8b-v1/config.yaml",
        "qwen3-8b-v1/corpus/corpus.parquet",
        "qwen3-8b-v1/corpus/corpus.parquet.meta.yaml",
        "qwen3-8b-v1/corpus/positions.parquet",
    ],
)
