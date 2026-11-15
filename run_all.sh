#!/usr/bin/env bash
# Повний відтворюваний запуск (CPU, ~25 хв на 2 vCPU / 7 ГБ ОЗП). Seed = 42 зафіксовано в src/lib.py.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p data cache results figures
[ -f data/ua_squad_dataset.json ] || curl -L -o data/ua_squad_dataset.json \
  https://github.com/fido-ai/ua-datasets/releases/download/v0.0.1/ua_squad_dataset.json
python src/build_scores.py      | tee results/build_scores.log
python src/run_retrieval.py     | tee results/run_retrieval.log
python src/run_context.py       | tee results/run_context.log
python src/run_errors_figs.py   | tee results/run_errors_figs.log
python src/run_latency.py       | tee results/run_latency.log
