# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SpecReason is a proof-of-concept implementation of speculative reasoning for fast inference-time compute. It uses a large base model (Qwen/QwQ-32B) and a small draft model (DeepSeek-R1-Distill-Qwen-1.5B) served via vLLM. The small model proposes reasoning steps; the base model scores them and either accepts or regenerates them. Paper: [arXiv:2504.07891](https://arxiv.org/abs/2504.07891).

## Setup

```bash
conda create -n specreason python=3.12 -y
conda activate specreason
pip install vllm==0.8.2
pip install -r requirements.txt
```

**Note:** vLLM 0.8.2 requires manual patches for speculative decoding vocab size mismatches. See `spec_decoding_fix.md` for the exact file edits needed.

## Running vLLM Servers

Both servers must be running before executing `spec_reason.py` or `aime_viewer.py`:

```bash
# Terminal 1: Base model (port 30000)
VLLM_USE_V1=0 vllm serve Qwen/QwQ-32B --dtype auto -tp 2 \
  --max_model_len 8192 --gpu-memory-utilization 0.8 \
  --enable-prefix-caching --port 30000

# Terminal 2: Draft model (port 30001)
VLLM_USE_V1=0 vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --dtype auto -tp 2 \
  --max_model_len 8192 --gpu-memory-utilization 0.1 \
  --enable-prefix-caching --port 30001
```

## Running the Pipeline

```bash
mkdir results
python spec_reason.py \
  --dataset_name aime \
  --problem_id 60 \
  --repeat_id 0 \
  --score_threshold 7.0 \
  --score_method greedy \
  --token_budget 8192 \
  --output_dir ./results
```

Key arguments:
- `--dataset_name`: `aime`, `math`, or `gpqa`
- `--problem_id`: AIME→60-89, MATH-500→0-499, GPQA→0-N
- `--repeat_id`: 0-15 (multiple runs per problem)
- `--score_threshold`: 0-9 float; steps scoring above this are accepted from the small model
- `--score_method`: `greedy` (fastest token) or `average` (weighted over logprobs)
- `--first_n_steps_base_model`: Force the first N steps to use the base model

## Streamlit UI

```bash
streamlit run aime_viewer.py
```

Allows browsing datasets (AIME 2024, MATH-500, GPQA) and running speculative reasoning interactively with per-step visualization.

## Tests

```bash
python extraction_test.py   # Tests \boxed{} answer extraction logic
python smoke_test_setup.py  # Validates installed dependencies
python test_stream.py       # Tests vLLM streaming
```

## Architecture

**`spec_reason.py`** — Core pipeline loop:
1. Starts with the base model for step 0 (or first N steps if `--first_n_steps_base_model` is set)
2. Each subsequent step: small model generates a candidate, base model scores it (0–9)
3. If score ≥ threshold → accept small model output; else → regenerate with base model
4. Extraction: looks for `\boxed{...}` (handles nested braces); when found, base model verifies (YES/NO)
5. Stops when answer is verified or token budget is exhausted

**Scoring** uses vLLM's `logprobs`/`top_logprobs` on the next digit token. Greedy takes the argmax digit; average takes the expectation over digit tokens 0–9.

**`aime_viewer.py`** — Streamlit UI that wraps `spec_reason.py`'s logic inline (not as a subprocess), displaying side-by-side small/base model outputs with color-coded scores per step.

**Output format** — Each run writes to `{output_dir}/{problem_id}/{repeat_id}.pickle` and `.txt`. The pickle contains a list of step metadata dicts with fields: `step_id`, `step_str`, `small_model_step`, `score`, `base_model_step`, timing fields, `stop_reason`, etc.

## HPC / Batch

```bash
sbatch spec_reason_della.sh  # Princeton Della cluster (SLURM)
```
