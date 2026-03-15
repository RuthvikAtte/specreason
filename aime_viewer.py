import streamlit as st
from datasets import load_dataset, load_from_disk
import os
import re
import json
import pickle
import subprocess
import datetime
from pathlib import Path

st.set_page_config(page_title="SpecReason Viewer", layout="wide")
st.title("SpecReason Viewer")

# ─── Top-level navigation ────────────────────────────────────────────────────
tab_explorer, tab_dashboard = st.tabs(["Problem Explorer", "Experiment Dashboard"])


# ─── Shared helpers ──────────────────────────────────────────────────────────
@st.cache_data(show_spinner=True)
def load_dataset_records(dataset_choice):
    records = []
    if dataset_choice == "AIME 2024":
        ds = load_dataset("HuggingFaceH4/aime_2024")["train"]
        for idx, item in enumerate(ds):
            records.append({
                "index": idx,
                "problem_id": idx + 60,
                "problem": item.get("problem", ""),
                "answer": str(item.get("answer", "")),
            })
    elif dataset_choice == "MATH-500":
        ds = load_dataset("HuggingFaceH4/MATH-500")["test"]
        for idx, item in enumerate(ds):
            records.append({
                "index": idx,
                "problem_id": idx,
                "problem": item.get("problem", ""),
                "answer": str(item.get("solution", item.get("answer", ""))),
            })
    elif dataset_choice == "GPQA (Disk)":
        disk_path = "/scratch/gpfs/rp2773/hf_cache/datasets/gpqa"
        if not os.path.exists(disk_path):
            st.error(f"Directory {disk_path} not found on this machine.")
            return []
        ds = load_from_disk(disk_path)
        for idx, item in enumerate(ds):
            records.append({
                "index": idx,
                "problem_id": idx,
                "problem": item.get("Question", ""),
                "answer": (f"Correct: {item.get('Correct Answer', '')}\n"
                           f"Incorrect 1: {item.get('Incorrect Answer 1', '')}\n"
                           f"Incorrect 2: {item.get('Incorrect Answer 2', '')}\n"
                           f"Incorrect 3: {item.get('Incorrect Answer 3', '')}"),
            })
    elif dataset_choice == "GPQA Diamond":
        try:
            ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond")["train"]
            for idx, item in enumerate(ds):
                records.append({
                    "index": idx,
                    "problem_id": idx,
                    "problem": item.get("Question", ""),
                    "answer": (f"Correct: {item.get('Correct Answer', '')}\n"
                               f"Incorrect 1: {item.get('Incorrect Answer 1', '')}\n"
                               f"Incorrect 2: {item.get('Incorrect Answer 2', '')}\n"
                               f"Incorrect 3: {item.get('Incorrect Answer 3', '')}"),
                })
        except Exception as e:
            st.error(f"Failed to load GPQA Diamond: {e}")
    return records


def get_arg_dataset_name(ds_choice):
    if ds_choice == "AIME 2024":  return "aime"
    if ds_choice == "MATH-500":   return "math"
    return "gpqa"


def extract_boxed(text):
    if not text:
        return None
    marker = r'\boxed{'
    idx = text.rfind(marker)
    if idx == -1:
        return None
    start = idx + len(marker)
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if   text[i] == '{': depth += 1
        elif text[i] == '}': depth -= 1
        i += 1
    return text[start:i - 1].strip() if depth == 0 else None


def normalize_math(s):
    if not s:
        return ""
    s = str(s).strip()
    s = s.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")
    s = s.replace(r"\left", "").replace(r"\right", "")
    return s.replace(" ", "")


def render_text_box(text, bg_color):
    if text:
        safe = text.replace('<', '&lt;').replace('>', '&gt;')
        st.markdown(
            f"<div style='background:{bg_color};padding:12px;border-radius:5px;"
            f"white-space:pre-wrap;font-family:monospace;font-size:0.82em'>{safe}</div>",
            unsafe_allow_html=True)
    else:
        st.write("N/A")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Problem Explorer
# ═══════════════════════════════════════════════════════════════════════════════
with tab_explorer:
    st.sidebar.header("Dataset")
    dataset_choice = st.sidebar.selectbox(
        "Choose Dataset",
        ["AIME 2024", "MATH-500", "GPQA Diamond", "GPQA (Disk)"],
        key="ds_select",
    )

    records = load_dataset_records(dataset_choice)

    st.sidebar.header("Filters")
    query = st.sidebar.text_input("Search text in question", value="", key="search")
    filtered = records
    if query.strip():
        q = query.strip().lower()
        filtered = [r for r in records if q in r["problem"].lower()]

    if not filtered:
        st.warning("No questions matched your search.")
        st.stop()

    if "current_idx" not in st.session_state:
        st.session_state.current_idx = 0
    if st.session_state.current_idx >= len(filtered):
        st.session_state.current_idx = 0

    st.sidebar.markdown("---")
    st.sidebar.write(f"**Problem {st.session_state.current_idx + 1} of {len(filtered)}**")
    nav1, nav2, nav3 = st.sidebar.columns([1, 1, 1])
    with nav1:
        if st.button("⬅️", key="prev"):
            if st.session_state.current_idx > 0:
                st.session_state.current_idx -= 1
                st.rerun()
    with nav3:
        if st.button("➡️", key="nxt"):
            if st.session_state.current_idx < len(filtered) - 1:
                st.session_state.current_idx += 1
                st.rerun()

    selected = filtered[st.session_state.current_idx]

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader(f"Problem {selected['problem_id']} (index {selected['index']})")
        st.write(selected["problem"])
    with col2:
        with st.expander("Show answer"):
            st.code(selected["answer"])

    st.markdown("---")

    # ── Trace source picker ─────────────────────────────────────────────────
    st.subheader("View Existing Traces")

    src_col, rep_col = st.columns(2)
    with src_col:
        trace_dir = st.selectbox(
            "Results directory",
            ["playground_ui", "results", "results_fixed", "Custom…"],
            key="trace_dir_select",
        )
        if trace_dir == "Custom…":
            trace_dir = st.text_input("Enter path", value="./results", key="trace_custom")

    # Detect available repeats for this problem in the chosen dir
    prob_dir = Path(trace_dir) / str(selected["problem_id"])
    available_repeats = sorted([
        int(p.stem) for p in prob_dir.glob("*.pickle")
    ]) if prob_dir.exists() else []

    with rep_col:
        if available_repeats:
            repeat_id = st.selectbox(
                f"Repeat ID ({len(available_repeats)} available)",
                available_repeats,
                key="repeat_sel",
            )
        else:
            repeat_id = 0
            st.info("No traces found in this directory for this problem.")

    # ── Run new trace ───────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Run Speculative Reasoning")
    run_dir = "playground_ui"

    col_r1, col_r2 = st.columns(2)
    with col_r1:
        n_base_steps = st.number_input(
            "Force first N steps to base model",
            min_value=0, max_value=20, value=3, key="nbase",
            help="Ensures the problem is correctly set up before the small model takes over.")
    with col_r2:
        acceptance_threshold = st.number_input(
            "Acceptance threshold", min_value=0.0, max_value=9.0,
            value=7.0, step=0.5, key="thresh",
            help="Score out of 9 required for a small-model step to be accepted.")

    if st.button(f"Run Speculative Reasoning for Problem {selected['problem_id']}"):
        os.makedirs(run_dir, exist_ok=True)
        d_name = get_arg_dataset_name(dataset_choice)
        with st.spinner("Running spec_reason.py… (requires models on localhost 30000/30001)"):
            st.write("Live output:")
            out_ph = st.empty()
            proc = subprocess.Popen(
                ["python3", "-u", "spec_reason.py",
                 "--dataset_name", d_name,
                 "--problem_id", str(selected["problem_id"]),
                 "--output_dir", run_dir,
                 "--repeat_id", "0",
                 "--first_n_steps_base_model", str(n_base_steps),
                 "--score_threshold", str(acceptance_threshold)],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
            )
            full_out = ""
            for line in iter(proc.stdout.readline, ""):
                full_out += line
                out_ph.code(full_out[-4000:], language="text")
            proc.wait()
            if proc.returncode != 0:
                st.error(f"spec_reason.py exited with code {proc.returncode}")
            else:
                st.success("Done!")
        # After run, point the trace viewer at playground_ui repeat 0
        available_repeats = [0]
        trace_dir = run_dir
        repeat_id = 0

    # ── Render trace ─────────────────────────────────────────────────────────
    st.markdown("---")
    pickle_path = Path(trace_dir) / str(selected["problem_id"]) / f"{repeat_id}.pickle"

    if pickle_path.exists():
        st.info(f"Loaded trace: `{pickle_path}`")
        with open(pickle_path, "rb") as f:
            metadata_list = pickle.load(f)

        total_small_toks = sum(s.get("num_output_tokens_small") or 0 for s in metadata_list)
        total_base_toks  = sum(s.get("num_output_tokens_base")  or 0 for s in metadata_list)
        accepted_steps = sum(
            1 for s in metadata_list
            if s.get("score") is not None and float(s.get("score")) >= acceptance_threshold)
        rejected_steps = sum(
            1 for s in metadata_list
            if s.get("score") is not None and float(s.get("score")) < acceptance_threshold)

        m1, m2, m3 = st.columns(3)
        m1.metric("Total Steps", len(metadata_list))
        m2.metric("Total Output Tokens", total_small_toks + total_base_toks,
                  help=f"Small: {total_small_toks} | Base: {total_base_toks}")
        m3.metric("Accepted / Rejected", f"{accepted_steps} / {rejected_steps}")

        # Answer correctness
        extracted_answer = None
        for step in reversed(metadata_list):
            out = step.get("base_model_step") or step.get("small_model_step") or ""
            ans = extract_boxed(out)
            if ans is not None:
                extracted_answer = ans
                break

        last_step = metadata_list[-1]

        if extracted_answer is not None:
            st.markdown(f"### Result: Model answered `{extracted_answer}`")
            official_full = str(selected["answer"]).strip()
            official_boxed = extract_boxed(official_full)
            target = official_boxed if official_boxed is not None else official_full

            clean_ext = normalize_math(extracted_answer)
            clean_tgt = normalize_math(target)
            is_correct = False
            try:
                from math_verify import verify, parse
                is_correct = bool(verify(parse(clean_ext), parse(clean_tgt)))
            except Exception:
                pass
            if not is_correct:
                is_correct = (clean_ext == clean_tgt)

            if is_correct:
                st.success(f"✅ Matches official answer: `{target}`")
            else:
                st.error(f"❌ `{extracted_answer}` does not match official answer `{target}`")

            if last_step.get("stop_reason") == "budget":
                st.warning("Token budget exceeded — evaluated against last `\\boxed{}` found.")

        elif get_arg_dataset_name(dataset_choice) == "aime" and last_step.get("stop_reason") == "budget":
            last_str = last_step.get("base_model_step") or last_step.get("small_model_step") or ""
            numbers = re.findall(r'\b\d+\b', last_str)
            if numbers:
                h_ans = numbers[-1]
                st.markdown(f"### Result: Heuristic answer `{h_ans}`")
                st.warning("Token budget exceeded — no `\\boxed{}`, extracted last number heuristically.")
                official = str(selected["answer"]).strip()
                if h_ans == official:
                    st.success(f"✅ Heuristic answer matches official: `{official}`")
                else:
                    st.error(f"❌ Heuristic `{h_ans}` ≠ official `{official}`")
            else:
                st.warning("⚠️ Token budget exceeded and no numbers found heuristically.")
        elif last_step.get("stop_reason") == "budget":
            st.warning("⚠️ Token budget exceeded — no `\\boxed{}` generated.")
        else:
            st.warning("⚠️ No `\\boxed{}` found in model output.")

        # Per-step breakdown
        st.markdown("---")
        st.subheader("Step-by-step Trace")
        for step in metadata_list:
            score_val = step.get("score")
            if score_val is not None:
                if float(score_val) >= acceptance_threshold:
                    score_str = f"🟢 Score: {score_val}"
                    bg = "rgba(0,200,0,0.08)"
                else:
                    score_str = f"🔴 Score: {score_val}"
                    bg = "rgba(220,0,0,0.08)"
            else:
                score_str = "⚪ Score: N/A"
                bg = "rgba(128,128,128,0.08)"

            st.markdown("---")
            st.subheader(f"Step {step['step_id']}  ({score_str})")
            small_t = step.get("num_output_tokens_small") or 0
            base_t  = step.get("num_output_tokens_base")  or 0
            st.markdown(
                f"**Tokens:** {small_t + base_t} &mdash; *(Small: {small_t} | Base: {base_t})*  "
                f"&nbsp;&nbsp; **Time:** {step.get('step_time', 0):.2f}s")

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### Small Model Output")
                render_text_box(step.get("small_model_step"), bg)
            with c2:
                st.markdown("#### Base Model Output")
                render_text_box(step.get("base_model_step"), bg)

            if step.get("justification"):
                st.markdown("#### Evaluation Justification")
                render_text_box(step["justification"], "rgba(128,128,128,0.08)")
    else:
        st.write("No trace found for this problem / repeat. Run one above, or check the directory.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Experiment Dashboard
# ═══════════════════════════════════════════════════════════════════════════════
with tab_dashboard:
    st.subheader("Experiment Dashboard")

    dash_col1, dash_col2 = st.columns([3, 1])
    with dash_col1:
        results_dir = st.text_input("Results directory", value="./results", key="dash_dir")
    with dash_col2:
        st.write("")
        st.write("")
        if st.button("🔄 Refresh", key="refresh_btn"):
            st.rerun()

    results_path = Path(results_dir)
    summary_path = results_path / "experiment_summary.json"
    logs_dir = Path("./logs")

    # ── Service status ────────────────────────────────────────────────────────
    st.markdown("#### Service Status")
    try:
        res = subprocess.run(
            ["systemctl", "--user", "is-active", "specreason-experiment"],
            capture_output=True, text=True, timeout=3)
        svc_status = res.stdout.strip()
    except Exception:
        svc_status = "unknown"

    status_color = {"active": "🟢", "inactive": "⚫", "failed": "🔴"}.get(svc_status, "🟡")
    st.markdown(f"{status_color} `specreason-experiment` service is **{svc_status}**")

    if summary_path.exists():
        mtime = datetime.datetime.fromtimestamp(summary_path.stat().st_mtime)
        age_s = (datetime.datetime.now() - mtime).total_seconds()
        age_str = (f"{int(age_s)}s ago" if age_s < 120
                   else f"{int(age_s//60)}m ago" if age_s < 3600
                   else f"{age_s/3600:.1f}h ago")
        st.caption(f"Summary last updated: {mtime.strftime('%Y-%m-%d %H:%M:%S')} ({age_str})")

    # ── Experiment summary ────────────────────────────────────────────────────
    if summary_path.exists():
        with open(summary_path) as f:
            exp = json.load(f)

        prog = exp.get("progress", {})
        done_runs  = prog.get("total_runs_done", 0)
        total_runs = prog.get("total_runs_expected", 480)
        done_probs = prog.get("problems_completed", 0)
        total_probs = prog.get("problems_total", 30)

        cfg = exp.get("config", {})
        st.markdown("#### Configuration")
        cfg_cols = st.columns(5)
        cfg_cols[0].metric("k (repeats)", cfg.get("k", "—"))
        cfg_cols[1].metric("Score threshold", cfg.get("score_threshold", "—"))
        cfg_cols[2].metric("Score method", cfg.get("score_method", "—"))
        cfg_cols[3].metric("Token budget", cfg.get("token_budget", "—"))
        cfg_cols[4].metric("Forced base steps", cfg.get("first_n_steps_base_model", "—"))

        st.markdown("#### Progress")
        st.progress(done_runs / total_runs if total_runs else 0,
                    text=f"{done_runs} / {total_runs} runs  ({done_probs}/{total_probs} problems)")

        # Overall accuracy
        overall_acc = exp.get("overall_accuracy")
        total_correct = exp.get("total_correct", 0)
        agg = exp.get("aggregate", {})

        ov_cols = st.columns(5)
        ov_cols[0].metric("Overall Accuracy",
                           f"{overall_acc:.1%}" if overall_acc is not None else "—",
                           help=f"{total_correct} correct out of {done_runs} completed runs")
        ov_cols[1].metric("Avg Tokens / run",
                           f"{agg.get('total_tokens', '—'):.0f}" if agg.get('total_tokens') else "—")
        ov_cols[2].metric("Avg Time / run",
                           f"{agg.get('total_time_s', '—'):.1f}s" if agg.get('total_time_s') else "—")
        ov_cols[3].metric("Avg Acceptance Rate",
                           f"{agg.get('acceptance_rate', 0):.1%}" if agg.get('acceptance_rate') is not None else "—")
        ov_cols[4].metric("Avg Steps / run",
                           f"{agg.get('total_steps', '—'):.1f}" if agg.get('total_steps') else "—")

        # ── Per-problem table ─────────────────────────────────────────────────
        st.markdown("#### Per-Problem Results")

        per_prob = exp.get("per_problem", {})
        rows = []
        for pid_str, pdata in sorted(per_prob.items(), key=lambda x: int(x[0])):
            avgs = pdata.get("averages") or {}
            rows.append({
                "Problem": int(pid_str),
                "Runs Done": pdata.get("completed_runs", 0),
                "Correct": pdata.get("correct_count", 0),
                "Accuracy": pdata.get("accuracy"),
                "Avg Tokens": avgs.get("total_tokens"),
                "Avg Steps": avgs.get("total_steps"),
                "Avg Accepted": avgs.get("accepted_steps"),
                "Avg Rejected": avgs.get("rejected_steps"),
                "Avg Accept Rate": avgs.get("acceptance_rate"),
                "Avg Time (s)": avgs.get("total_time_s"),
                "Errors": pdata.get("error_runs", 0),
            })

        if rows:
            import pandas as pd

            df = pd.DataFrame(rows)

            def fmt_pct(v):
                return f"{v:.1%}" if v is not None else "—"
            def fmt_f1(v):
                return f"{v:.1f}" if v is not None else "—"

            styled = (df.style
                .format({
                    "Accuracy":       fmt_pct,
                    "Avg Accept Rate": fmt_pct,
                    "Avg Tokens":     fmt_f1,
                    "Avg Steps":      fmt_f1,
                    "Avg Accepted":   fmt_f1,
                    "Avg Rejected":   fmt_f1,
                    "Avg Time (s)":   fmt_f1,
                })
                .background_gradient(subset=["Accuracy"], cmap="RdYlGn", vmin=0, vmax=1)
                .background_gradient(subset=["Avg Accept Rate"], cmap="Blues", vmin=0, vmax=1)
            )
            st.dataframe(styled, use_container_width=True, hide_index=True)

        # ── Aggregate token breakdown bar chart ───────────────────────────────
        if rows and any(r["Avg Tokens"] for r in rows):
            try:
                import pandas as pd
                import altair as alt

                chart_df = pd.DataFrame([{
                    "Problem": r["Problem"],
                    "Base tokens": (agg.get("base_tokens") or 0),
                    "Small accepted": (agg.get("small_tokens_accepted") or 0),
                    "Small rejected": (agg.get("small_tokens_rejected") or 0),
                } for r in rows if r["Avg Tokens"]])

                # Per-problem token breakdown from individual summaries
                ptoken_rows = []
                for pid_str, pdata in sorted(per_prob.items(), key=lambda x: int(x[0])):
                    avgs = pdata.get("averages") or {}
                    if avgs.get("total_tokens"):
                        ptoken_rows.append({
                            "Problem": int(pid_str),
                            "Base tokens": avgs.get("base_tokens") or 0,
                            "Small accepted": avgs.get("small_tokens_accepted") or 0,
                            "Small rejected": avgs.get("small_tokens_rejected") or 0,
                        })

                if ptoken_rows:
                    pt_df = pd.DataFrame(ptoken_rows)
                    pt_melt = pt_df.melt("Problem", var_name="Type", value_name="Tokens")
                    chart = (alt.Chart(pt_melt)
                        .mark_bar()
                        .encode(
                            x=alt.X("Problem:O", title="Problem ID"),
                            y=alt.Y("Tokens:Q"),
                            color=alt.Color("Type:N",
                                scale=alt.Scale(
                                    domain=["Base tokens", "Small accepted", "Small rejected"],
                                    range=["#e45756", "#54a24b", "#f58518"])),
                            tooltip=["Problem", "Type", "Tokens"],
                        ).properties(title="Avg Token Breakdown per Problem", height=280))
                    st.altair_chart(chart, use_container_width=True)
            except Exception:
                pass  # chart is optional; skip if altair/pandas unavailable

    else:
        st.info(f"No `experiment_summary.json` found in `{results_dir}`. "
                "Start the experiment to see results here.")

    # ── Log viewer ────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Logs")

    log_col1, log_col2 = st.columns([3, 1])
    with log_col2:
        log_lines = st.number_input("Lines to show", min_value=20, max_value=500,
                                    value=80, step=20, key="log_lines")
    with log_col1:
        log_type = st.radio("Log file", ["Latest run log", "Errors only"],
                            horizontal=True, key="log_type")

    if log_type == "Errors only":
        log_path = logs_dir / "experiment_errors.log"
    else:
        # Find most recent timestamped log
        log_path = None
        if logs_dir.exists():
            candidates = sorted(logs_dir.glob("experiment_2*.log"), reverse=True)
            if candidates:
                log_path = candidates[0]

    if log_path and log_path.exists():
        try:
            with open(log_path) as f:
                lines = f.readlines()
            tail = "".join(lines[-int(log_lines):])
            st.caption(f"File: `{log_path}`  ({len(lines)} total lines)")
            st.code(tail, language="text")
        except Exception as e:
            st.error(f"Could not read log: {e}")
    else:
        st.info("No log files found yet. Logs appear in `./logs/` once the experiment starts.")

    # ── Per-run log drill-down ────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Per-Run Log Drill-down")
    dr_col1, dr_col2 = st.columns(2)
    with dr_col1:
        drill_prob = st.number_input("Problem ID", min_value=60, max_value=89,
                                     value=60, key="drill_prob")
    with dr_col2:
        drill_rep = st.number_input("Repeat ID", min_value=0, max_value=15,
                                    value=0, key="drill_rep")

    run_log_path = results_path / str(drill_prob) / f"{drill_rep}.log"
    if run_log_path.exists():
        with open(run_log_path) as f:
            run_log_lines = f.readlines()
        st.caption(f"`{run_log_path}`  ({len(run_log_lines)} lines)")
        tail_lines = st.slider("Lines from end", 20, min(500, len(run_log_lines)),
                               min(100, len(run_log_lines)), key="drill_tail")
        st.code("".join(run_log_lines[-tail_lines:]), language="text")
    else:
        st.info(f"No log found at `{run_log_path}`")
