#!/usr/bin/env bash
# SpecReason experiment: k=2, token_budget=32768
# Run this AFTER both vLLM servers are up:
#   Base model (32B)  → port 30000
#   Draft model (1.5B) → port 30001

cd "$(dirname "$0")"

python run_experiment.py \
    --dataset_name aime \
    --k 2 \
    --output_dir ./32k_results \
    --logs_dir  ./32k_logs \
    --score_threshold 7.0 \
    --score_method greedy \
    --token_budget 32768 \
    --first_n_steps_base_model 0 \
    --timeout_per_run 1800
