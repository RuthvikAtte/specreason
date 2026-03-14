import streamlit as st
from datasets import load_dataset, load_from_disk
import os
import pickle
import subprocess

st.set_page_config(page_title="Dataset Viewer", layout="wide")
st.title("Dataset & Spec Reasoning Viewer")
st.caption("Browse problems and run speculative reasoning")

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
            problem_text = item.get("Question", "")
            ans = f"Correct: {item.get('Correct Answer', '')}\nIncorrect 1: {item.get('Incorrect Answer 1', '')}\nIncorrect 2: {item.get('Incorrect Answer 2', '')}\nIncorrect 3: {item.get('Incorrect Answer 3', '')}"
            records.append({
                "index": idx,
                "problem_id": idx,
                "problem": problem_text,
                "answer": ans,
            })
    elif dataset_choice == "GPQA Diamond":
        try:
            ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond")["train"]
            for idx, item in enumerate(ds):
                problem_text = item.get("Question", "")
                ans = f"Correct: {item.get('Correct Answer', '')}\nIncorrect 1: {item.get('Incorrect Answer 1', '')}\nIncorrect 2: {item.get('Incorrect Answer 2', '')}\nIncorrect 3: {item.get('Incorrect Answer 3', '')}"
                records.append({
                    "index": idx,
                    "problem_id": idx,
                    "problem": problem_text,
                    "answer": ans,
                })
        except Exception as e:
            st.error(f"Failed to load GPQA Diamond dataset. The dataset might be behind a gate. Details: {e}")
            
    return records

def get_arg_dataset_name(ds_choice):
    if ds_choice == "AIME 2024": return "aime"
    if ds_choice == "MATH-500": return "math"
    return "gpqa"

st.sidebar.header("Dataset Selection")
dataset_choice = st.sidebar.selectbox(
    "Choose Dataset", 
    ["AIME 2024", "MATH-500", "GPQA Diamond", "GPQA (Disk)"]
)

records = load_dataset_records(dataset_choice)

st.sidebar.header("Filters")
query = st.sidebar.text_input("Search text in question", value="")

filtered = records
if query.strip():
    q = query.strip().lower()
    filtered = [r for r in records if q in r["problem"].lower()]

if not filtered:
    st.warning("No questions matched your search.")
    st.stop()

# Initialize session state for problem index if it doesn't exist
if "current_idx" not in st.session_state:
    st.session_state.current_idx = 0

# Bound check in case filters change
if st.session_state.current_idx >= len(filtered):
    st.session_state.current_idx = 0

st.sidebar.markdown("---")
st.sidebar.write(f"**Showing problem {st.session_state.current_idx + 1} of {len(filtered)}**")

nav_col1, nav_col2, nav_col3 = st.sidebar.columns([1, 1, 1])
with nav_col1:
    if st.button("⬅️"):
        if st.session_state.current_idx > 0:
            st.session_state.current_idx -= 1
            st.rerun()

with nav_col3:
    if st.button("➡️"):
        if st.session_state.current_idx < len(filtered) - 1:
            st.session_state.current_idx += 1
            st.rerun()

# Select the actual problem using the index
selected = filtered[st.session_state.current_idx]

col1, col2 = st.columns([2, 1])
with col1:
    st.subheader(f"Problem {selected['problem_id']} (index {selected['index']})")
    st.write(selected["problem"])

with col2:
    with st.expander("Show answer"):
        st.code(selected["answer"])

st.markdown("---")
st.subheader("Speculative Reasoning Runner")
output_dir = "playground_ui"

col_runner_1, col_runner_2 = st.columns(2)
with col_runner_1:
    n_base_steps = st.number_input("Force first N steps to base model", min_value=0, max_value=20, value=3, help="Ensures the problem is correctly set up before the small model takes over.")
with col_runner_2:
    acceptance_threshold = st.number_input("Acceptance threshold", min_value=0.0, max_value=9.0, value=7.0, step=0.5, help="Score required out of 9 for the small model's step to form. Higher means stricter.")

if st.button("Run Speculative Reasoning for Problem " + str(selected['problem_id'])):
    os.makedirs(output_dir, exist_ok=True)
    d_name = get_arg_dataset_name(dataset_choice)
    with st.spinner("Running spec_reason.py... (Requires models on localhost 30000/30001)"):
        # Run subprocess
        result = subprocess.run([
            "python3", "spec_reason.py",
            "--dataset_name", d_name,
            "--problem_id", str(selected["problem_id"]),
            "--output_dir", output_dir,
            "--repeat_id", "0",
            "--first_n_steps_base_model", str(n_base_steps),
            "--score_threshold", str(acceptance_threshold)
        ], capture_output=True, text=True)
        if result.returncode != 0:
            st.error(f"Error running spec_reason.py:\n{result.stderr}")
        else:
            st.success("Successfully finished reasoning!")

pickle_path = os.path.join(output_dir, str(selected['problem_id']), "0.pickle")
if os.path.exists(pickle_path):
    st.info("Loaded trace from previous run for this problem")
    with open(pickle_path, "rb") as f:
        metadata_list = pickle.load(f)
    
    total_small_tokens = sum(step.get('num_output_tokens_small') or 0 for step in metadata_list)
    total_base_tokens = sum(step.get('num_output_tokens_base') or 0 for step in metadata_list)
    
    st.write(f"**Total Steps Taken:** {len(metadata_list)}")
    st.write(f"**Total Output Tokens:** {total_small_tokens + total_base_tokens} (Small: {total_small_tokens} | Base: {total_base_tokens})")
    
    def render_full_text(text, bg_color):
        if text:
            # Escape HTML brackets so things like <think> show up properly
            safe_text = text.replace('<', '&lt;').replace('>', '&gt;')
            st.markdown(
                f"<div style='background-color: {bg_color}; padding: 15px; border-radius: 5px; white-space: pre-wrap; font-family: monospace;'>{safe_text}</div>", 
                unsafe_allow_html=True
            )
        else:
            st.write("N/A")

    for step in metadata_list:
        score_val = step.get('score')
        
        # Color coding logic based on threshold (7.0 is default in spec_reason.py)
        if score_val is not None:
            if float(score_val) >= 7.0:
                score_str = f"🟢 Score: {score_val}"
                step_bg_color = "rgba(0, 255, 0, 0.1)" # Light transparent green
            else:
                score_str = f"🔴 Score: {score_val}"
                step_bg_color = "rgba(255, 0, 0, 0.1)" # Light transparent red
        else:
            score_str = "⚪ Score: N/A"
            step_bg_color = "rgba(128, 128, 128, 0.1)" # Transparent grey
            
        with st.expander(f"Step {step['step_id']} ({score_str})"):
            small_toks = step.get('num_output_tokens_small') or 0
            base_toks = step.get('num_output_tokens_base') or 0
            st.markdown(f"**Output Tokens (Total):** {small_toks + base_toks} &mdash; *(Small: {small_toks} | Base: {base_toks})*")
            st.markdown(f"**Step Time:** {step.get('step_time', 0):.2f}s")
            
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("### Small Model Output")
                render_full_text(step.get('small_model_step'), step_bg_color)
            with c2:
                st.markdown("### Base Model Output")
                render_full_text(step.get('base_model_step'), step_bg_color)
            
            if step.get('justification'):
                st.markdown("#### Evaluation Justification")
                render_full_text(step['justification'], "rgba(128, 128, 128, 0.1)")
else:
    st.write("No reasoning trace found for this problem yet. Click 'Run' above.")
