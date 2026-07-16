#!/usr/bin/env bash
set -euxo pipefail

curl -LsSf https://astral.sh/uv/install.sh | sh
cd /workspace/oracle-lens
/root/.local/bin/uv venv /workspace/venv --python 3.12
/root/.local/bin/uv pip install \
  --python /workspace/venv/bin/python \
  "torch==2.8.0" \
  -e third_party/natural_language_autoencoders \
  -e third_party/jacobian-lens \
  -e . \
  "transformers==5.14.*" \
  pytest scipy
# Editable dependency resolution can prefer the CPU wheel from PyPI. Reinstall
# torch alone so uv's PyTorch backend selector deterministically takes effect.
/root/.local/bin/uv pip install \
  --python /workspace/venv/bin/python \
  --reinstall \
  "torch==2.8.0" \
  --torch-backend cu126
/workspace/venv/bin/python -c \
  'import torch, transformers; assert torch.cuda.is_available(); assert transformers.__version__.startswith("5.14")'
/workspace/venv/bin/python -m pytest -q
/root/.local/bin/uv pip freeze --python /workspace/venv/bin/python > /workspace/env-m1.txt
