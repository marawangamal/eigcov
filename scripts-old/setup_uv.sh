#!/bin/bash
set -e

rm -f pyproject.toml uv.lock .python-version
rm -rf .venv
echo "Removed .venv and pyproject.toml"

module load python/3.9
uv init --python 3.9 --no-readme

uv add torch torchvision numpy scipy tqdm Pillow \
  "transformers==4.57.6" datasets huggingface-hub peft evaluate trl \
  sentencepiece protobuf \
  open-clip-torch scikit-learn


uv add "promptsource @ git+https://github.com/bigscience-workshop/promptsource.git"