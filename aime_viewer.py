import streamlit as st
from datasets import load_dataset, load_from_disk
import os

st.set_page_config(page_title="Dataset Viewer", layout="wide")
st.title("Dataset Viewer")
st.caption("Browse and search problems from AIME, MATH-500, and GPQA")

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
            
    return records

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

problem_ids = [r["problem_id"] for r in filtered]
selected_id = st.sidebar.selectbox("Select problem_id", problem_ids, index=0)
selected = next(r for r in filtered if r["problem_id"] == selected_id)

col1, col2 = st.columns([2, 1])
with col1:
    st.subheader(f"Problem {selected['problem_id']} (index {selected['index']})")
    st.write(selected["problem"])

with col2:
    with st.expander("Show answer"):
        st.code(selected["answer"])

st.markdown("---")
st.subheader("Matching questions")
st.write(f"Showing {len(filtered)} of {len(records)} total")

for r in filtered:
    with st.expander(f"Problem {r['problem_id']}"):
        st.write(r["problem"])
