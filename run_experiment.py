#!/usr/bin/env python3
"""
SpecReason AIME Experiment Runner

Runs all 30 AIME 2024 problems (IDs 60-89) × k repeats each (default k=16).
Designed to be run as a background service. Supports resuming from where it
left off — any run whose pickle already exists is skipped.

Outputs per run:
  {output_dir}/{problem_id}/{repeat_id}.pickle   (from spec_reason.py)
  {output_dir}/{problem_id}/{repeat_id}.txt       (from spec_reason.py)
  {output_dir}/{problem_id}/{repeat_id}.log       (subprocess stdout/stderr)

Outputs per problem (after all k runs):
  {output_dir}/{problem_id}/summary.json

Output for entire experiment (updated after each problem completes):
  {output_dir}/experiment_summary.json

Logs:
  {logs_dir}/experiment_YYYYMMDD_HHMMSS.log   (this invocation, DEBUG+)
  {logs_dir}/experiment_errors.log             (persistent, errors only, append)
"""

import os
import sys
import json
import time
import pickle
import logging
import argparse
import subprocess
import datetime
from pathlib import Path

from recompute_accuracy import recompute_run, load_ground_truth

try:
    import requests as _requests_lib
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False


# ─── Logging ──────────────────────────────────────────────────────────────────

def setup_logging(logs_dir: str) -> str:
    os.makedirs(logs_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_log_path = os.path.join(logs_dir, f"experiment_{timestamp}.log")
    error_log_path = os.path.join(logs_dir, "experiment_errors.log")

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # Per-invocation timestamped log (DEBUG and above)
    fh = logging.FileHandler(run_log_path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    root.addHandler(fh)

    # Persistent error log across all runs (append)
    eh = logging.FileHandler(error_log_path, mode="a")
    eh.setLevel(logging.ERROR)
    eh.setFormatter(fmt)
    root.addHandler(eh)

    # Console (INFO and above)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    # Suppress noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    logging.info(f"Run log:   {run_log_path}")
    logging.info(f"Error log: {error_log_path}")
    return run_log_path


# ─── vLLM Health Check ────────────────────────────────────────────────────────

def check_vllm_health(base_url: str) -> bool:
    if not _HAS_REQUESTS:
        logging.warning("'requests' not installed; skipping vLLM health check.")
        return True
    try:
        resp = _requests_lib.get(f"{base_url}/v1/models", timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


def assert_vllm_healthy(base_url_32b: str, base_url_1_5b: str):
    for url, name in [(base_url_32b, "32b (port 30000)"),
                      (base_url_1_5b, "1.5b (port 30001)")]:
        if not check_vllm_health(url):
            logging.error(f"vLLM server {name} at {url} is not responding. "
                          "Start both servers before running the experiment.")
            sys.exit(1)
    logging.info("Both vLLM servers are healthy.")


# ─── Statistics Extraction ────────────────────────────────────────────────────

def _mean(records: list, key: str):
    """Mean of key across records, ignoring None values."""
    vals = [r[key] for r in records if r.get(key) is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


def compute_stats_from_pickle(pickle_path: str) -> dict:
    """Extract per-run statistics from a spec_reason.py metadata_list pickle."""
    with open(pickle_path, "rb") as f:
        metadata_list = pickle.load(f)

    if not metadata_list:
        return {}

    last = metadata_list[-1]
    stop_reason = last.get("stop_reason", "unknown")

    # is_correct / candidate_answer are stored per-step (added in spec_reason.py)
    # Use the value from the last step that had a verified answer; fall back to
    # the last entry's stored value for backward-compatible pickles.
    is_correct = None
    candidate_answer = None
    for m in metadata_list:
        if m.get("is_correct") is not None:
            is_correct = m["is_correct"]
            candidate_answer = m.get("candidate_answer")

    # Step classification
    # forced_base: first_n steps run directly on base model (no small model tried)
    forced_base = [m for m in metadata_list
                   if m.get("small_model_step") is None
                   and m.get("base_model_step") is not None]
    # accepted: small model was tried, score >= threshold, small output kept
    accepted = [m for m in metadata_list
                if m.get("small_model_step") is not None
                and m.get("base_model_step") is None]
    # rejected: small model was tried, score < threshold, fell back to base
    rejected = [m for m in metadata_list
                if m.get("small_model_step") is not None
                and m.get("base_model_step") is not None]

    total_steps = len(metadata_list)
    forced_base_steps = len(forced_base)
    accepted_steps = len(accepted)
    rejected_steps = len(rejected)
    spec_steps = accepted_steps + rejected_steps
    acceptance_rate = (accepted_steps / spec_steps) if spec_steps > 0 else None

    # Token counts
    total_tokens = sum(m.get("final_num_output_tokens") or 0 for m in metadata_list)
    base_tokens = sum(m.get("num_output_tokens_base") or 0 for m in metadata_list)
    # Tokens spent on accepted small-model steps (used)
    small_tokens_accepted = sum(m.get("num_output_tokens_small") or 0 for m in accepted)
    # Tokens spent on rejected small-model steps (wasted / overhead)
    small_tokens_rejected = sum(m.get("num_output_tokens_small") or 0 for m in rejected)

    # Time
    total_time_s = sum(m.get("step_time") or 0.0 for m in metadata_list)
    base_model_time_s = sum(
        m.get("base_model_time") or 0.0 for m in metadata_list
        if m.get("base_model_time") is not None)
    small_model_time_s = sum(
        m.get("small_model_time") or 0.0 for m in metadata_list
        if m.get("small_model_time") is not None)
    eval_time_s = sum(
        m.get("eval_time") or 0.0 for m in metadata_list
        if m.get("eval_time") is not None)

    # Scores
    all_scores = [m["score"] for m in metadata_list if m.get("score") is not None]
    accepted_scores = [m["score"] for m in accepted if m.get("score") is not None]
    rejected_scores = [m["score"] for m in rejected if m.get("score") is not None]

    avg_score = round(sum(all_scores) / len(all_scores), 4) if all_scores else None
    avg_accepted_score = (round(sum(accepted_scores) / len(accepted_scores), 4)
                          if accepted_scores else None)
    avg_rejected_score = (round(sum(rejected_scores) / len(rejected_scores), 4)
                          if rejected_scores else None)

    # Also read mathverify result if the pickle has already been recomputed
    is_correct_mathverify = None
    for m in metadata_list:
        if m.get("is_correct_mathverify") is not None:
            is_correct_mathverify = m["is_correct_mathverify"]

    return {
        "is_correct": is_correct,
        "is_correct_mathverify": is_correct_mathverify,
        "candidate_answer": candidate_answer,
        "stop_reason": stop_reason,
        "total_steps": total_steps,
        "forced_base_steps": forced_base_steps,
        "accepted_steps": accepted_steps,
        "rejected_steps": rejected_steps,
        "acceptance_rate": acceptance_rate,
        "total_tokens": total_tokens,
        "base_tokens": base_tokens,
        "small_tokens_accepted": small_tokens_accepted,
        "small_tokens_rejected": small_tokens_rejected,
        "total_time_s": round(total_time_s, 3),
        "base_model_time_s": round(base_model_time_s, 3),
        "small_model_time_s": round(small_model_time_s, 3),
        "eval_time_s": round(eval_time_s, 3),
        "avg_score": avg_score,
        "avg_accepted_score": avg_accepted_score,
        "avg_rejected_score": avg_rejected_score,
    }


# ─── Single Run ───────────────────────────────────────────────────────────────

def run_single(args, problem_id: int, repeat_id: int) -> dict:
    """
    Run spec_reason.py for one (problem_id, repeat_id) via subprocess.
    Returns a result dict containing status + all stats.
    Skips (and loads stats from pickle) if the output already exists.
    """
    problem_dir = Path(args.output_dir) / str(problem_id)
    pickle_path = problem_dir / f"{repeat_id}.pickle"
    run_log_path = problem_dir / f"{repeat_id}.log"

    base_result = {"problem_id": problem_id, "repeat_id": repeat_id}

    # Resume: skip if already done
    if pickle_path.exists():
        logging.debug(f"[{problem_id}/{repeat_id}] Pickle exists, skipping.")
        try:
            stats = compute_stats_from_pickle(str(pickle_path))
            return {**base_result, "status": "skipped", **stats}
        except Exception as e:
            logging.error(f"[{problem_id}/{repeat_id}] Could not load existing pickle: {e}")
            return {**base_result, "status": "error", "reason": f"bad_existing_pickle: {e}"}

    problem_dir.mkdir(parents=True, exist_ok=True)

    # Build subprocess command
    script_path = Path(__file__).parent / "spec_reason.py"
    cmd = [
        sys.executable, "-u", str(script_path),
        "--dataset_name", args.dataset_name,
        "--problem_id", str(problem_id),
        "--repeat_id", str(repeat_id),
        "--score_threshold", str(args.score_threshold),
        "--score_method", args.score_method,
        "--token_budget", str(args.token_budget),
        "--first_n_steps_base_model", str(args.first_n_steps_base_model),
        "--output_dir", args.output_dir,
        "--quiet",
    ]

    logging.info(f"[{problem_id}/{repeat_id}] Starting run ...")
    logging.debug(f"[{problem_id}/{repeat_id}] Command: {' '.join(cmd)}")
    t_start = time.monotonic()

    try:
        with open(run_log_path, "w") as log_fh:
            proc = subprocess.run(
                cmd,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                timeout=args.timeout_per_run,
                cwd=str(Path(__file__).parent),
            )
        elapsed = round(time.monotonic() - t_start, 2)

        if proc.returncode != 0:
            logging.error(
                f"[{problem_id}/{repeat_id}] FAILED (rc={proc.returncode}) "
                f"in {elapsed}s — see {run_log_path}")
            return {**base_result, "status": "error",
                    "returncode": proc.returncode, "wall_time_s": elapsed}

        if not pickle_path.exists():
            logging.error(
                f"[{problem_id}/{repeat_id}] Subprocess exited 0 but no pickle "
                f"at {pickle_path}")
            return {**base_result, "status": "error",
                    "reason": "missing_pickle", "wall_time_s": elapsed}

        stats = compute_stats_from_pickle(str(pickle_path))
        logging.info(
            f"[{problem_id}/{repeat_id}] Done in {elapsed}s | "
            f"correct={stats.get('is_correct')} | "
            f"steps={stats.get('total_steps')} | "
            f"accepted={stats.get('accepted_steps')} | "
            f"rejected={stats.get('rejected_steps')} | "
            f"tokens={stats.get('total_tokens')} | "
            f"stop={stats.get('stop_reason')}")
        return {**base_result, "status": "completed", "wall_time_s": elapsed, **stats}

    except subprocess.TimeoutExpired:
        elapsed = round(time.monotonic() - t_start, 2)
        logging.error(
            f"[{problem_id}/{repeat_id}] TIMEOUT after {args.timeout_per_run}s")
        return {**base_result, "status": "timeout",
                "timeout_s": args.timeout_per_run, "wall_time_s": elapsed}

    except Exception as e:
        elapsed = round(time.monotonic() - t_start, 2)
        logging.error(
            f"[{problem_id}/{repeat_id}] Unexpected error: {e}", exc_info=True)
        return {**base_result, "status": "error",
                "reason": str(e), "wall_time_s": elapsed}


# ─── Summaries ────────────────────────────────────────────────────────────────

STAT_KEYS = [
    "total_steps", "forced_base_steps", "accepted_steps", "rejected_steps",
    "acceptance_rate", "total_tokens", "base_tokens", "small_tokens_accepted",
    "small_tokens_rejected", "total_time_s", "base_model_time_s",
    "small_model_time_s", "eval_time_s",
    "avg_score", "avg_accepted_score", "avg_rejected_score",
]


def write_problem_summary(args, problem_id: int, run_results: list) -> dict:
    """After all k runs for a problem, compute and write summary.json."""
    done = [r for r in run_results if r.get("status") in ("completed", "skipped")]
    errors = [r for r in run_results if r.get("status") not in ("completed", "skipped")]

    correct_mv  = [r for r in done if r.get("is_correct_mathverify") is True]
    correct_llm = [r for r in done if r.get("is_correct") is True]
    accuracy_mv  = round(len(correct_mv)  / len(done), 4) if done else None
    accuracy_llm = round(len(correct_llm) / len(done), 4) if done else None

    summary = {
        "problem_id": problem_id,
        "dataset": args.dataset_name,
        "k": args.k,
        "completed_runs": len(done),
        "error_runs": len(errors),
        "correct_count_mathverify": len(correct_mv),
        "correct_count_llm": len(correct_llm),
        "accuracy_mathverify": accuracy_mv,
        "accuracy_llm": accuracy_llm,
        # keep legacy field so old dashboard reads still work
        "accuracy": accuracy_mv,
        "correct_count": len(correct_mv),
        "averages": {key: _mean(done, key) for key in STAT_KEYS},
        "runs": sorted(run_results, key=lambda r: r["repeat_id"]),
    }

    path = Path(args.output_dir) / str(problem_id) / "summary.json"
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)

    logging.info(
        f"[Problem {problem_id}] mathverify={accuracy_mv} ({len(correct_mv)}/{len(done)}) | "
        f"llm={accuracy_llm} ({len(correct_llm)}/{len(done)}) | "
        f"avg_tokens={summary['averages']['total_tokens']} | "
        f"avg_time={summary['averages']['total_time_s']}s")
    return summary


def write_experiment_summary(args, all_problem_summaries: list) -> dict:
    """Write overall experiment_summary.json. Called after each problem."""
    all_runs = [r for s in all_problem_summaries
                for r in s.get("runs", [])
                if r.get("status") in ("completed", "skipped")]

    total_correct_mv  = sum(s.get("correct_count_mathverify", 0) for s in all_problem_summaries)
    total_correct_llm = sum(s.get("correct_count_llm", 0)        for s in all_problem_summaries)
    total_done = sum(s.get("completed_runs", 0) for s in all_problem_summaries)
    overall_accuracy_mv  = round(total_correct_mv  / total_done, 4) if total_done > 0 else None
    overall_accuracy_llm = round(total_correct_llm / total_done, 4) if total_done > 0 else None

    per_problem = {
        s["problem_id"]: {
            "accuracy_mathverify": s.get("accuracy_mathverify"),
            "accuracy_llm": s.get("accuracy_llm"),
            "accuracy": s.get("accuracy_mathverify"),  # legacy
            "correct_count_mathverify": s.get("correct_count_mathverify"),
            "correct_count_llm": s.get("correct_count_llm"),
            "correct_count": s.get("correct_count_mathverify"),  # legacy
            "completed_runs": s.get("completed_runs"),
            "error_runs": s.get("error_runs"),
            "averages": s.get("averages"),
        }
        for s in all_problem_summaries
    }

    summary = {
        "config": {
            "dataset_name": args.dataset_name,
            "problem_ids": args.problem_ids,
            "k": args.k,
            "score_threshold": args.score_threshold,
            "score_method": args.score_method,
            "token_budget": args.token_budget,
            "first_n_steps_base_model": args.first_n_steps_base_model,
            "output_dir": args.output_dir,
        },
        "progress": {
            "problems_completed": len(all_problem_summaries),
            "problems_total": len(args.problem_ids),
            "total_runs_done": total_done,
            "total_runs_expected": len(args.problem_ids) * args.k,
        },
        "overall_accuracy_mathverify": overall_accuracy_mv,
        "overall_accuracy_llm": overall_accuracy_llm,
        "overall_accuracy": overall_accuracy_mv,  # legacy
        "total_correct_mathverify": total_correct_mv,
        "total_correct_llm": total_correct_llm,
        "total_correct": total_correct_mv,  # legacy
        "aggregate": {key: _mean(all_runs, key) for key in STAT_KEYS},
        "per_problem": per_problem,
    }

    path = Path(args.output_dir) / "experiment_summary.json"
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)

    logging.info(
        f"Experiment summary updated: {len(all_problem_summaries)}/{len(args.problem_ids)} "
        f"problems done | mathverify={overall_accuracy_mv} llm={overall_accuracy_llm} "
        f"({total_correct_mv}/{total_done} runs correct)")
    return summary


# ─── Argument Parsing ─────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run all 30 AIME 2024 problems × k repeats with SpecReason.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset_name", type=str, default="aime",
                        choices=["aime", "math", "gpqa"])
    parser.add_argument("--k", type=int, default=16,
                        help="Number of repeats per problem")
    parser.add_argument("--problem_ids", type=int, nargs="+",
                        default=list(range(60, 90)),
                        help="Problem IDs to run (default: 60-89, all 30 AIME problems)")
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="Root directory for all output files")
    parser.add_argument("--logs_dir", type=str, default="./logs",
                        help="Directory for experiment log files")
    parser.add_argument("--score_threshold", type=float, default=7.0)
    parser.add_argument("--score_method", type=str,
                        choices=["greedy", "average"], default="greedy")
    parser.add_argument("--token_budget", type=int, default=8192)
    parser.add_argument("--first_n_steps_base_model", type=int, default=0)
    parser.add_argument("--base_url_32b", type=str,
                        default="http://localhost:30000",
                        help="vLLM endpoint for the 32B base model")
    parser.add_argument("--base_url_1_5b", type=str,
                        default="http://localhost:30001",
                        help="vLLM endpoint for the 1.5B draft model")
    parser.add_argument("--timeout_per_run", type=int, default=1800,
                        help="Per-run subprocess timeout in seconds (default: 30 min)")
    parser.add_argument("--skip_health_check", action="store_true", default=False,
                        help="Skip the vLLM server health check at startup")
    return parser.parse_args()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    run_log = setup_logging(args.logs_dir)

    logging.info("=" * 70)
    logging.info("SpecReason AIME Experiment Runner")
    logging.info("=" * 70)
    logging.info(f"Problems:        {args.problem_ids}")
    logging.info(f"Repeats (k):     {args.k}")
    logging.info(f"Total runs:      {len(args.problem_ids) * args.k}")
    logging.info(f"Score threshold: {args.score_threshold}")
    logging.info(f"Score method:    {args.score_method}")
    logging.info(f"Token budget:    {args.token_budget}")
    logging.info(f"Output dir:      {args.output_dir}")
    logging.info(f"Timeout/run:     {args.timeout_per_run}s")
    logging.info("=" * 70)

    os.makedirs(args.output_dir, exist_ok=True)

    if not args.skip_health_check:
        assert_vllm_healthy(args.base_url_32b, args.base_url_1_5b)

    script_path = Path(__file__).parent / "spec_reason.py"
    if not script_path.exists():
        logging.error(f"spec_reason.py not found at {script_path}")
        sys.exit(1)

    ground_truth = load_ground_truth(args.dataset_name)
    all_problem_summaries = []
    experiment_start = time.monotonic()

    for prob_idx, problem_id in enumerate(args.problem_ids):
        logging.info("")
        logging.info(f"{'─' * 60}")
        logging.info(f"PROBLEM {problem_id}  "
                     f"({prob_idx + 1}/{len(args.problem_ids)})")
        logging.info(f"{'─' * 60}")

        run_results = []
        problem_start = time.monotonic()

        for repeat_id in range(args.k):
            result = run_single(args, problem_id, repeat_id)
            run_results.append(result)

            # After every completed run, recompute accuracy with math_verify
            # and immediately refresh both summaries so the dashboard updates.
            if result.get("status") in ("completed", "skipped"):
                pkl = Path(args.output_dir) / str(problem_id) / f"{repeat_id}.pickle"
                if pkl.exists():
                    try:
                        recompute_run(pkl, ground_truth.get(problem_id, ""), dry_run=False)
                        # Re-read stats with the freshly written mathverify field
                        updated_stats = compute_stats_from_pickle(str(pkl))
                        result.update(updated_stats)
                        run_results[-1] = result
                    except Exception as e:
                        logging.error(f"[{problem_id}/{repeat_id}] recompute failed: {e}")

            # Refresh problem + experiment summary after every run
            problem_summary = write_problem_summary(args, problem_id, run_results)
            current_summaries = [s for s in all_problem_summaries] + [problem_summary]
            write_experiment_summary(args, current_summaries)

        problem_elapsed = round(time.monotonic() - problem_start, 1)
        successes = sum(1 for r in run_results
                        if r.get("status") in ("completed", "skipped"))
        failures = len(run_results) - successes
        logging.info(
            f"Problem {problem_id} finished in {problem_elapsed}s: "
            f"{successes}/{args.k} runs succeeded, {failures} failed/timed out")

        problem_summary = write_problem_summary(args, problem_id, run_results)
        all_problem_summaries.append(problem_summary)
        write_experiment_summary(args, all_problem_summaries)

    total_elapsed = round(time.monotonic() - experiment_start, 1)
    logging.info("")
    logging.info("=" * 70)
    logging.info(f"EXPERIMENT COMPLETE in {total_elapsed}s")
    logging.info("=" * 70)


if __name__ == "__main__":
    main()
