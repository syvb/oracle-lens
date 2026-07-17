"""Pull the artifacts required by M3 (dictionary + teacher): M0 corpus and
positions, M1 activations/whitening, and the M2 v2 reconstructor checkpoint."""

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
snapshot_download(
    repo_id="syvb/oracle-lens-qwen3-8b-recon",
    local_dir="/workspace/runs/qwen3-8b-v1/reconstructor/checkpoint",
)
