#!/usr/bin/env python3
"""
Post-processing pass: recompute is_correct using math_verify instead of the LLM.

For each run pickle in results_dir, this script:
  1. Loads the ground-truth answer from the dataset
  2. Finds the candidate_answer stored in the pickle
  3. Compares with math_verify (handles 1/5 == 0.2, leading zeros, LaTeX fractions, etc.)
  4. Writes is_correct_mathverify into every step that has a candidate_answer,
     and patches is_correct_mathverify into the last step
  5. Rewrites the .pickle and .txt in-place
  6. Regenerates per-problem summary.json and experiment_summary.json

Adds a new field `is_correct_mathverify` alongside the existing `is_correct`
(LLM-based) so both are preserved. Summaries get both `accuracy_llm` and
`accuracy_mathverify` fields.

Usage:
  python recompute_accuracy.py --results_dir ./results --dataset_name aime
"""

import os
import sys
import json
import pickle
import pprint
import logging
import argparse
import statistics
from pathlib import Path

try:
    from math_verify import verify, parse as mv_parse
    _HAS_MATHVERIFY = True
except ImportError:
    _HAS_MATHVERIFY = False
    logging.warning("math_verify not installed — pip install math_verify")
    sys.exit(1)

from datasets import load_dataset


# ─── Ground truth loader ──────────────────────────────────────────────────────

def load_ground_truth(dataset_name: str) -> dict:
    """Returns {problem_id: answer_str}."""
    logging.info(f"Loading ground truth for dataset: {dataset_name}")
    if dataset_name == "aime":
        ds = load_dataset("HuggingFaceH4/aime_2024")["train"]
        return {idx + 60: str(ds["answer"][idx]) for idx in range(len(ds))}
    elif dataset_name == "math":
        ds = load_dataset("HuggingFaceH4/MATH-500")["test"]
        return {idx: str(ds["answer"][idx]) for idx in range(len(ds))}
    else:
        raise NotImplementedError(f"dataset {dataset_name} not supported")


# ─── Math comparison ──────────────────────────────────────────────────────────

def mathverify_equal(candidate: str, ground_truth: str) -> bool:
    """Compare two answer strings using math_verify; fall back to string match."""
    if not candidate:
        return False
    try:
        result = verify(mv_parse(candidate), mv_parse(ground_truth))
        return bool(result)
    except Exception:
        pass
    # Fallback: normalised string match
    def norm(s):
        return s.strip().lstrip("0") or "0"
    return norm(candidate) == norm(ground_truth)


# ─── Per-run recompute ────────────────────────────────────────────────────────

def recompute_run(pickle_path: Path, ground_truth_answer: str, dry_run: bool) -> dict:
    """
    Load one run's pickle, recompute is_correct_mathverify, optionally save.
    Returns a dict with the key stats for the run.
    """
    with open(pickle_path, "rb") as f:
        metadata_list = pickle.load(f)

    if not metadata_list:
        return {"candidate_answer": None, "is_correct_mathverify": False,
                "is_correct_llm": None}

    # Find the last step that produced a candidate_answer
    final_candidate = None
    for step in reversed(metadata_list):
        ca = step.get("candidate_answer")
        if ca is not None:
            final_candidate = ca
            break

    mathverify_result = mathverify_equal(final_candidate, ground_truth_answer) \
        if final_candidate is not None else False

    # Patch every step that has a candidate_answer
    changed = False
    for step in metadata_list:
        ca = step.get("candidate_answer")
        if ca is not None:
            mv = mathverify_equal(ca, ground_truth_answer)
            if step.get("is_correct_mathverify") != mv:
                step["is_correct_mathverify"] = mv
                changed = True
        else:
            if "is_correct_mathverify" not in step:
                step["is_correct_mathverify"] = None
                changed = True

    # Ensure last step has the final verdict
    last = metadata_list[-1]
    if last.get("is_correct_mathverify") != mathverify_result:
        last["is_correct_mathverify"] = mathverify_result
        changed = True

    llm_correct = last.get("is_correct")

    if changed and not dry_run:
        with open(pickle_path, "wb") as f:
            pickle.dump(metadata_list, f)
        txt_path = pickle_path.with_suffix(".txt")
        with open(txt_path, "w") as f:
            pprint.pprint(metadata_list, stream=f)
        logging.debug(f"  Saved {pickle_path}")

    return {
        "candidate_answer": final_candidate,
        "ground_truth": ground_truth_answer,
        "is_correct_mathverify": mathverify_result,
        "is_correct_llm": llm_correct,
    }


# ─── Summary helpers (mirrors run_experiment.py) ──────────────────────────────

STAT_KEYS = [
    "total_steps", "forced_base_steps", "accepted_steps", "rejected_steps",
    "acceptance_rate", "total_tokens", "base_tokens", "small_tokens_accepted",
    "small_tokens_rejected", "total_time_s", "base_model_time_s",
    "small_model_time_s", "eval_time_s",
    "avg_score", "avg_accepted_score", "avg_rejected_score",
]


def _mean(records, key):
    vals = [r[key] for r in records if r.get(key) is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


def compute_stats_from_pickle(pickle_path: Path) -> dict:
    with open(pickle_path, "rb") as f:
        metadata_list = pickle.load(f)
    if not metadata_list:
        return {}

    last = metadata_list[-1]
    forced_base = [m for m in metadata_list
                   if m.get("small_model_step") is None and m.get("base_model_step") is not None]
    accepted    = [m for m in metadata_list
                   if m.get("small_model_step") is not None and m.get("base_model_step") is None]
    rejected    = [m for m in metadata_list
                   if m.get("small_model_step") is not None and m.get("base_model_step") is not None]

    spec_steps = len(accepted) + len(rejected)
    all_scores      = [m["score"] for m in metadata_list if m.get("score") is not None]
    accepted_scores = [m["score"] for m in accepted      if m.get("score") is not None]
    rejected_scores = [m["score"] for m in rejected      if m.get("score") is not None]

    return {
        "is_correct_llm":        last.get("is_correct"),
        "is_correct_mathverify": last.get("is_correct_mathverify"),
        "candidate_answer":      last.get("candidate_answer"),
        "stop_reason":           last.get("stop_reason", "unknown"),
        "total_steps":           len(metadata_list),
        "forced_base_steps":     len(forced_base),
        "accepted_steps":        len(accepted),
        "rejected_steps":        len(rejected),
        "acceptance_rate":       (len(accepted) / spec_steps) if spec_steps else None,
        "total_tokens":          sum(m.get("final_num_output_tokens") or 0 for m in metadata_list),
        "base_tokens":           sum(m.get("num_output_tokens_base")  or 0 for m in metadata_list),
        "small_tokens_accepted": sum(m.get("num_output_tokens_small") or 0 for m in accepted),
        "small_tokens_rejected": sum(m.get("num_output_tokens_small") or 0 for m in rejected),
        "total_time_s":          round(sum(m.get("step_time")       or 0 for m in metadata_list), 3),
        "base_model_time_s":     round(sum(m.get("base_model_time") or 0 for m in metadata_list
                                           if m.get("base_model_time") is not None), 3),
        "small_model_time_s":    round(sum(m.get("small_model_time") or 0 for m in metadata_list
                                           if m.get("small_model_time") is not None), 3),
        "eval_time_s":           round(sum(m.get("eval_time")       or 0 for m in metadata_list
                                           if m.get("eval_time") is not None), 3),
        "avg_score":             round(sum(all_scores) / len(all_scores), 4)           if all_scores      else None,
        "avg_accepted_score":    round(sum(accepted_scores) / len(accepted_scores), 4) if accepted_scores else None,
        "avg_rejected_score":    round(sum(rejected_scores) / len(rejected_scores), 4) if rejected_scores else None,
    }


def write_problem_summary(problem_id: int, k: int, results_dir: Path,
                          dataset_name: str) -> dict:
    problem_dir = results_dir / str(problem_id)
    runs = []
    for repeat_id in range(k):
        pkl = problem_dir / f"{repeat_id}.pickle"
        if not pkl.exists():
            continue
        try:
            stats = compute_stats_from_pickle(pkl)
            stats["repeat_id"] = repeat_id
            runs.append(stats)
        except Exception as e:
            logging.error(f"  [{problem_id}/{repeat_id}] Could not load: {e}")

    done = runs
    correct_mv  = sum(1 for r in done if r.get("is_correct_mathverify") is True)
    correct_llm = sum(1 for r in done if r.get("is_correct_llm") is True)

    summary = {
        "problem_id":            problem_id,
        "dataset":               dataset_name,
        "k":                     k,
        "completed_runs":        len(done),
        # mathverify accuracy (primary)
        "correct_count_mathverify": correct_mv,
        "accuracy_mathverify":   round(correct_mv  / len(done), 4) if done else None,
        # LLM accuracy (for comparison)
        "correct_count_llm":     correct_llm,
        "accuracy_llm":          round(correct_llm / len(done), 4) if done else None,
        "averages":              {key: _mean(done, key) for key in STAT_KEYS},
        "runs":                  sorted(done, key=lambda r: r["repeat_id"]),
    }

    path = problem_dir / "summary.json"
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def write_experiment_summary(all_summaries: list, results_dir: Path,
                             config: dict) -> dict:
    all_runs = [r for s in all_summaries for r in s.get("runs", [])]

    total_mv  = sum(s.get("correct_count_mathverify", 0) for s in all_summaries)
    total_llm = sum(s.get("correct_count_llm", 0)        for s in all_summaries)
    total_done = sum(s.get("completed_runs", 0)           for s in all_summaries)

    per_problem = {}
    for s in all_summaries:
        per_problem[s["problem_id"]] = {
            "accuracy_mathverify": s.get("accuracy_mathverify"),
            "accuracy_llm":        s.get("accuracy_llm"),
            "correct_count_mathverify": s.get("correct_count_mathverify"),
            "correct_count_llm":   s.get("correct_count_llm"),
            "completed_runs":      s.get("completed_runs"),
            "averages":            s.get("averages"),
        }

    summary = {
        "config":                  config,
        "total_runs":              total_done,
        "overall_accuracy_mathverify": round(total_mv  / total_done, 4) if total_done else None,
        "overall_accuracy_llm":        round(total_llm / total_done, 4) if total_done else None,
        "total_correct_mathverify":    total_mv,
        "total_correct_llm":           total_llm,
        "aggregate":               {key: _mean(all_runs, key) for key in STAT_KEYS},
        "per_problem":             per_problem,
    }

    path = results_dir / "experiment_summary.json"
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    logging.info(f"Experiment summary written to {path}")
    return summary


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    parser = argparse.ArgumentParser(
        description="Recompute is_correct using math_verify and regenerate summaries.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--results_dir",   type=str, default="./results")
    parser.add_argument("--dataset_name",  type=str, default="aime",
                        choices=["aime", "math"])
    parser.add_argument("--k",             type=int, default=16)
    parser.add_argument("--problem_ids",   type=int, nargs="+",
                        default=list(range(60, 90)))
    parser.add_argument("--dry_run",       action="store_true",
                        help="Compute and log results without writing any files")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    ground_truth = load_ground_truth(args.dataset_name)

    logging.info(f"Results dir:  {results_dir}")
    logging.info(f"Dataset:      {args.dataset_name}")
    logging.info(f"Problems:     {args.problem_ids}")
    logging.info(f"k:            {args.k}")
    logging.info(f"Dry run:      {args.dry_run}")
    logging.info("")

    # ── Per-run recompute ────────────────────────────────────────────────────
    total_runs = 0
    agreement  = 0   # LLM and math_verify agree
    disagree   = 0   # they disagree (interesting cases)

    for problem_id in args.problem_ids:
        gt = ground_truth.get(problem_id)
        if gt is None:
            logging.warning(f"No ground truth for problem {problem_id}, skipping")
            continue

        problem_dir = results_dir / str(problem_id)
        if not problem_dir.exists():
            logging.debug(f"[{problem_id}] No results directory yet, skipping")
            continue

        prob_mv_correct = 0
        prob_total = 0

        for repeat_id in range(args.k):
            pkl = problem_dir / f"{repeat_id}.pickle"
            if not pkl.exists():
                continue

            result = recompute_run(pkl, gt, dry_run=args.dry_run)
            total_runs += 1
            prob_total  += 1

            mv  = result["is_correct_mathverify"]
            llm = result["is_correct_llm"]

            if mv:
                prob_mv_correct += 1

            if mv == bool(llm):
                agreement += 1
            else:
                disagree  += 1
                logging.info(
                    f"  [{problem_id}/{repeat_id}] DISAGREEMENT — "
                    f"mathverify={mv}, llm={llm}, "
                    f"candidate='{result['candidate_answer']}', "
                    f"truth='{gt}'")

        if prob_total:
            logging.info(
                f"[{problem_id}] gt={gt!r}  "
                f"math_verify: {prob_mv_correct}/{prob_total}  "
                f"({prob_mv_correct/prob_total:.1%})")

    logging.info("")
    logging.info(f"Total runs processed: {total_runs}")
    if total_runs:
        logging.info(f"LLM vs math_verify agreement: "
                     f"{agreement}/{total_runs} ({agreement/total_runs:.1%})")
        if disagree:
            logging.info(f"Disagreements: {disagree} — see lines above for details")

    # ── Regenerate summaries ─────────────────────────────────────────────────
    if not args.dry_run:
        logging.info("")
        logging.info("Regenerating summaries …")
        all_summaries = []
        for problem_id in args.problem_ids:
            if not (results_dir / str(problem_id)).exists():
                continue
            summary = write_problem_summary(
                problem_id, args.k, results_dir, args.dataset_name)
            all_summaries.append(summary)
            logging.info(
                f"  [{problem_id}] "
                f"mathverify={summary.get('accuracy_mathverify')} "
                f"({summary.get('correct_count_mathverify')}/{summary.get('completed_runs')})  "
                f"llm={summary.get('accuracy_llm')} "
                f"({summary.get('correct_count_llm')}/{summary.get('completed_runs')})")

        if all_summaries:
            config = {
                "dataset_name":  args.dataset_name,
                "problem_ids":   args.problem_ids,
                "k":             args.k,
                "results_dir":   str(results_dir),
                "recomputed_with": "math_verify",
            }
            write_experiment_summary(all_summaries, results_dir, config)

    logging.info("Done.")


if __name__ == "__main__":
    main()
