"""Pull the exact public M0/M1 artifacts required by M2 (recon-train/eval)."""

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
        "qwen3-8b-v1/activations/meta.yaml",
        "qwen3-8b-v1/activations/vectors.f16.bin",
        "qwen3-8b-v1/whitening.pt",
    ],
)
