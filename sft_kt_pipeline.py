#!/usr/bin/env python3
# sft_kt_pipeline.py
# Supervised fine-tuning pipeline for knowledge tracing on o4-mini.
# - Prepares JSONL (messages format) with your prompt
# - Starts fine-tune on o4-mini
# - Evaluates on _test split (exact match with student_option_id)

import os
import re
import json
import ast
import time
import math
import argparse
import random
from dataclasses import dataclass
from collections import deque, defaultdict
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd
from tqdm import tqdm

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ----------------------------
# Configuration defaults
# ----------------------------

DEFAULT_MODEL = "o4-mini"
HISTORY_K = 5  # number of prior examples to include
RANDOM_SEED = 42
VAL_FRACTION = 0.08  # validation split by student_id
MAX_OUTPUT_TOKENS = 4  # at inference time; index-only
TEMPERATURE = 0.0

# ----------------------------
# Prompt templates (adjusted per your specs)
# ----------------------------

SYSTEM_PROMPT = (
    "You are a student working on an exam on databases, containing multiple choice questions. "
    "You are shown a set of questions that you answered earlier in the exam, together with the correct answers and your student answers. "
    "Analyze your responses to identify possible misconceptions that led to incorrect answers. "
    "Inspect the new question and think how you would answer it as such a student. "
    "You may answer incorrectly if that is what the student is likely to do for this question. "
    "Important: Respond with ONLY the integer index (1-based) of the chosen multiple choice option. Do not include any other text."
)

HUMAN1_HEADER = "Question-answer records:"
HUMAN2_PREFIX = "New multiple choice question:\n"

# ----------------------------
# Utilities
# ----------------------------

def parse_option_texts(x: Any) -> List[str]:
    """Parse the option_texts column which is a string representation of a Python list."""
    if isinstance(x, list):
        return x
    if pd.isna(x):
        return []
    try:
        parsed = ast.literal_eval(x)
        if isinstance(parsed, list):
            return [str(o) for o in parsed]
        return []
    except Exception:
        # Fallback: try to split by '||' or semicolon
        return [p.strip() for p in str(x).split("||") if p.strip()]

def safe_int(x: Any, default: int = None) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        return default

def fmt_options(opts: List[str]) -> str:
    lines = []
    for i, opt in enumerate(opts, start=1):
        lines.append(f"{i}) {opt}")
    return "\n".join(lines)

def fmt_history_record(qid: Any,
                       q_text: str,
                       options: List[str],
                       correct_idx: int,
                       student_idx: int,
                       correct_flag: Optional[bool]) -> str:
    verdict = None
    if correct_flag is True:
        verdict = "Correct"
    elif correct_flag is False:
        verdict = "Incorrect"
    verdict_str = f" ({verdict})" if verdict is not None else ""
    block = [
        f"Q{qid}: {q_text}",
        "Options:",
        fmt_options(options),
        f"Correct answer: {correct_idx}",
        f"Your answer: {student_idx}{verdict_str}",
        "---",
    ]
    return "\n".join(block)

def fmt_new_question(qid: Any, q_text: str, options: List[str]) -> str:
    block = [
        f"Q{qid}: {q_text}",
        "Options:",
        fmt_options(options),
    ]
    return "\n".join(block)

def coerce_bool(x: Any) -> Optional[bool]:
    if isinstance(x, bool):
        return x
    if pd.isna(x):
        return None
    s = str(x).strip().lower()
    if s in {"true", "t", "1", "yes", "y"}:
        return True
    if s in {"false", "f", "0", "no", "n"}:
        return False
    return None

def ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)

# ----------------------------
# Data loading & preparation
# ----------------------------

@dataclass
class Question:
    question_id: Any
    text: str
    options: List[str]
    correct_idx: int
    num_options: int
    difficulty: Any

def load_questions(questions_csv: str) -> Dict[Any, Question]:
    dfq = pd.read_csv(questions_csv)
    required_cols = {"question_id", "q_text", "num_answer_options", "option_texts", "correct_option_id"}
    missing = required_cols - set(dfq.columns)
    if missing:
        raise ValueError(f"Questions CSV missing columns: {missing}")
    out: Dict[Any, Question] = {}
    for row in dfq.itertuples(index=False):
        opt_list = parse_option_texts(getattr(row, "option_texts"))
        num_opts = safe_int(getattr(row, "num_answer_options"), default=len(opt_list)) or len(opt_list)
        # pad/truncate options to match num_opts
        if len(opt_list) < num_opts:
            opt_list = opt_list + [f"(Option {i})" for i in range(len(opt_list)+1, num_opts+1)]
        elif len(opt_list) > num_opts:
            opt_list = opt_list[:num_opts]
        correct_idx = safe_int(getattr(row, "correct_option_id"))
        if correct_idx is None or not (1 <= correct_idx <= num_opts):
            # If invalid, try to recover by clamping
            correct_idx = max(1, min(num_opts, correct_idx or 1))
        out[getattr(row, "question_id")] = Question(
            question_id=getattr(row, "question_id"),
            text=str(getattr(row, "q_text")),
            options=[str(x) for x in opt_list],
            correct_idx=correct_idx,
            num_options=num_opts,
            difficulty=getattr(row, "q_difficulty") if "q_difficulty" in dfq.columns else None,
        )
    return out

@dataclass
class Interaction:
    row_idx: int
    interact_id: Any
    student_id: Any
    question_id: Any
    student_idx: int
    correct_flag: Optional[bool]
    time: pd.Timestamp

def load_interactions(csv_path: str) -> List[Interaction]:
    dfi = pd.read_csv(csv_path)
    required = {"interact_id", "student_id", "question_id", "student_option_id", "student_option_correct", "time"}
    missing = required - set(dfi.columns)
    if missing:
        raise ValueError(f"Interactions CSV missing columns: {missing}")
    # parse datetime and coerce fields
    dfi["time"] = pd.to_datetime(dfi["time"], errors="coerce")
    dfi["student_option_id"] = dfi["student_option_id"].apply(lambda x: safe_int(x, default=None))
    dfi["student_option_correct"] = dfi["student_option_correct"].apply(coerce_bool)

    interactions: List[Interaction] = []
    for i, row in enumerate(dfi.itertuples(index=False)):
        interactions.append(
            Interaction(
                row_idx=i,
                interact_id=getattr(row, "interact_id"),
                student_id=getattr(row, "student_id"),
                question_id=getattr(row, "question_id"),
                student_idx=safe_int(getattr(row, "student_option_id"), default=None),
                correct_flag=coerce_bool(getattr(row, "student_option_correct")),
                time=getattr(row, "time"),
            )
        )
    return interactions

# ----------------------------
# JSONL builders
# ----------------------------

def build_example_messages(history_blocks: str, new_question_block: str) -> List[Dict[str, str]]:
    # We combine the two human parts into one user message for simplicity,
    # keeping the same semantics as your original prompt.
    user_content = f"{HUMAN1_HEADER}\n{history_blocks}\n\n{HUMAN2_PREFIX}{new_question_block}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

def format_history(
    hist: List[Tuple[Interaction, Question]]
) -> str:
    parts = []
    for inter, q in hist:
        parts.append(
            fmt_history_record(
                qid=inter.question_id,
                q_text=q.text,
                options=q.options,
                correct_idx=q.correct_idx,
                student_idx=inter.student_idx if inter.student_idx is not None else "NA",
                correct_flag=inter.correct_flag,
            )
        )
    return "\n".join(parts) if parts else "(No prior records)\n---"

def build_jsonl_for_split(
    interactions: List[Interaction],
    questions: Dict[Any, Question],
    history_k: int,
    students_for_split: Optional[set] = None,
    restrict_to_students: Optional[set] = None,
    drop_missing_questions: bool = True
) -> List[Dict[str, Any]]:
    """
    Build JSONL lines for a split (train or val).
    - students_for_split: if provided, include only those students
    - restrict_to_students: if provided, also restrict to those students (used for val build)
    """
    # group interactions by student and sort chronologically
    by_student: Dict[Any, List[Interaction]] = defaultdict(list)
    for inter in interactions:
        if students_for_split and inter.student_id not in students_for_split:
            continue
        by_student[inter.student_id].append(inter)

    for sid in by_student:
        by_student[sid].sort(key=lambda x: (x.time.value if pd.notna(x.time) else -1, safe_int(x.interact_id, 0)))

    output: List[Dict[str, Any]] = []
    for sid, rows in by_student.items():
        window: deque = deque([], maxlen=history_k)
        for inter in rows:
            q = questions.get(inter.question_id)
            if drop_missing_questions and q is None:
                # skip if we lack question content
                continue
            # build history from prior items in deque
            hist_blocks = format_history([(h_inter, questions.get(h_inter.question_id)) for h_inter in window if questions.get(h_inter.question_id) is not None])

            new_block = fmt_new_question(qid=inter.question_id, q_text=q.text, options=q.options)
            messages = build_example_messages(hist_blocks, new_block)

            # target is the student's chosen index (1-based)
            target = inter.student_idx
            if target is None:
                # skip unlabeled
                window.append(inter)
                continue

            line = {
                "messages": messages + [{"role": "assistant", "content": str(target)}],
                "meta": {
                    "student_id": sid,
                    "question_id": inter.question_id,
                    "difficulty": q.difficulty,
                    "num_options": q.num_options,
                    "time": inter.time.isoformat() if pd.notna(inter.time) else None,
                    "correct_option_id": q.correct_idx,
                    "student_option_correct": inter.correct_flag,
                },
            }
            output.append(line)
            # slide window to include the just-answered interaction
            window.append(inter)
    return output

def build_eval_prompts(
    interactions: List[Interaction],
    questions: Dict[Any, Question],
    history_k: int
) -> List[Dict[str, Any]]:
    """Create eval prompts (messages) plus gold labels for the test split."""
    by_student: Dict[Any, List[Interaction]] = defaultdict(list)
    for inter in interactions:
        by_student[inter.student_id].append(inter)
    for sid in by_student:
        by_student[sid].sort(key=lambda x: (x.time.value if pd.notna(x.time) else -1, safe_int(x.interact_id, 0)))

    prompts: List[Dict[str, Any]] = []
    for sid, rows in by_student.items():
        window: deque = deque([], maxlen=history_k)
        for inter in rows:
            q = questions.get(inter.question_id)
            if q is None or inter.student_idx is None:
                window.append(inter)
                continue
            hist_blocks = format_history([(h_inter, questions.get(h_inter.question_id)) for h_inter in window if questions.get(h_inter.question_id) is not None])
            new_block = fmt_new_question(qid=inter.question_id, q_text=q.text, options=q.options)
            messages = build_example_messages(hist_blocks, new_block)
            prompts.append({
                "messages": messages,
                "gold": {
                    "student_option_id": inter.student_idx,   # 1-based
                    "question_id": inter.question_id,
                    "student_id": sid,
                    "difficulty": q.difficulty,
                    "num_options": q.num_options,
                }
            })
            window.append(inter)
    return prompts

# ----------------------------
# File IO helpers
# ----------------------------

def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out

# ----------------------------
# OpenAI FT + inference
# ----------------------------

def upload_file(client, path: str) -> str:
    with open(path, "rb") as f:
        up = client.files.create(file=f, purpose="fine-tune")
    return up.id

def start_fine_tune(client, model: str, training_file_id: str, validation_file_id: Optional[str] = None, suffix: Optional[str] = None) -> str:
    payload = {
        "model": model,
        "training_file": training_file_id,
        "hyperparameters": {"n_epochs": "auto"},
    }
    if validation_file_id:
        payload["validation_file"] = validation_file_id
    if suffix:
        payload["suffix"] = suffix[:40]
    job = client.fine_tuning.jobs.create(**payload)
    return job.id

def wait_for_job(client, job_id: str, poll_secs: float = 10.0) -> Tuple[str, Optional[str]]:
    """Polls the job until completion; returns (status, fine_tuned_model)."""
    last_status = None
    while True:
        job = client.fine_tuning.jobs.retrieve(job_id)
        status = job.status
        if status != last_status:
            print(f"[fine-tune] status = {status}")
            last_status = status
        if status in {"succeeded", "failed", "cancelled"}:
            ft_model = getattr(job, "fine_tuned_model", None)
            return status, ft_model
        time.sleep(poll_secs)

def parse_index(text: str, num_options: int) -> Optional[int]:
    """
    Extract the first 1..num_options integer from the model output.
    The model is instructed to output only the index, but we parse defensively.
    """
    if text is None:
        return None
    # remove non-digit separators, keep negatives for safety (not expected)
    m = re.search(r"\b(\d{1,2})\b", text)
    if not m:
        return None
    idx = int(m.group(1))
    if 1 <= idx <= max(1, num_options):
        return idx
    return None

def evaluate_model_on_prompts(client, model: str, eval_prompts: List[Dict[str, Any]], max_items: Optional[int] = None) -> Dict[str, Any]:
    """
    Calls the fine-tuned model and computes accuracy against the student's chosen index.
    Uses Chat Completions for role-based parity with SFT messages.
    """
    from openai import OpenAI
    acc_numer = 0
    acc_denom = 0
    per_diff = defaultdict(lambda: {"correct": 0, "count": 0})
    results = []

    total = len(eval_prompts if max_items is None else eval_prompts[:max_items])
    print(f"Evaluating {total} examples...")

    for ex in tqdm(eval_prompts[:max_items] if max_items is not None else eval_prompts):
        messages = ex["messages"]
        gold = ex["gold"]
        num_opts = int(gold.get("num_options", 4))
        # Use Chat Completions for messages (o4-mini supports it as well)
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_OUTPUT_TOKENS,
        )
        raw_out = resp.choices[0].message.content if (resp and resp.choices and resp.choices[0].message) else None
        pred = parse_index(raw_out, num_opts)
        gold_idx = int(gold["student_option_id"])
        correct = (pred == gold_idx)
        acc_numer += int(correct)
        acc_denom += 1
        d = gold.get("difficulty", None)
        per_diff[d]["count"] += 1
        per_diff[d]["correct"] += int(correct)
        results.append({
            "question_id": gold["question_id"],
            "student_id": gold["student_id"],
            "difficulty": d,
            "gold": gold_idx,
            "pred": pred,
            "raw": raw_out,
            "num_options": num_opts,
            "correct": correct,
        })

    overall_acc = acc_numer / acc_denom if acc_denom else 0.0
    per_diff_acc = {k: (v["correct"] / v["count"] if v["count"] else None) for k, v in per_diff.items()}
    return {
        "overall_accuracy": overall_acc,
        "n": acc_denom,
        "by_difficulty": per_diff_acc,
        "details": results,
    }

# ----------------------------
# Main
# ----------------------------

def main():
    parser = argparse.ArgumentParser(description="SFT pipeline for knowledge tracing (o4-mini).")
    parser.add_argument("--data-dir", type=str, default=".", help="Directory containing CSVs.")
    parser.add_argument("--questions-csv", type=str, default=None, help="Path to questions CSV (defaults to *_questions.csv in data dir).")
    parser.add_argument("--train-csv", type=str, default=None, help="Path to interactions train CSV (defaults to *_interactions_train.csv).")
    parser.add_argument("--test-csv", type=str, default=None, help="Path to interactions test CSV (defaults to *_interactions_test.csv).")
    parser.add_argument("--out-dir", type=str, default="./outputs_sft", help="Where to write JSONL and reports.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Base model to fine-tune (default: o4-mini).")
    parser.add_argument("--suffix", type=str, default="kt_o4mini", help="Suffix/name for the fine-tuned model.")
    parser.add_argument("--val-frac", type=float, default=VAL_FRACTION, help="Fraction of students for validation.")
    parser.add_argument("--hist-k", type=int, default=HISTORY_K, help="Number of prior interactions to include in context.")
    parser.add_argument("--eval-max", type=int, default=None, help="Max eval items (for quick runs).")
    parser.add_argument("--prepare-only", action="store_true", help="Only prepare JSONL, do not start fine-tune.")
    parser.add_argument("--skip-eval", action="store_true", help="Skip evaluation after fine-tuning.")
    args = parser.parse_args()

    random.seed(RANDOM_SEED)

    ensure_dir(args.out_dir)

    # Resolve file paths
    def find_one(glob_suffix: str) -> Optional[str]:
        cands = [os.path.join(args.data_dir, f) for f in os.listdir(args.data_dir) if f.endswith(glob_suffix)]
        return cands[0] if cands else None

    q_csv = args.questions_csv or find_one("_questions.csv")
    train_csv = args.train_csv or find_one("_interactions_train.csv")
    test_csv = args.test_csv or find_one("_interactions_test.csv")

    if not q_csv or not os.path.exists(q_csv):
        raise FileNotFoundError("Questions CSV not found. Use --questions-csv to specify it.")
    if not train_csv or not os.path.exists(train_csv):
        raise FileNotFoundError("Train interactions CSV not found. Use --train-csv to specify it.")
    if test_csv and not os.path.exists(test_csv):
        test_csv = None  # ignore if bad path

    print(f"[load] questions: {q_csv}")
    print(f"[load] train:     {train_csv}")
    if test_csv:
        print(f"[load] test:      {test_csv}")
    else:
        print("[load] test:      (not found; evaluation will be skipped unless --skip-eval)")

    # Load data
    questions = load_questions(q_csv)
    train_inters = load_interactions(train_csv)
    test_inters = load_interactions(test_csv) if test_csv else []

    # Report basic stats
    n_q = len(questions)
    n_train = len(train_inters)
    n_test = len(test_inters)
    uniq_students_train = len(set(i.student_id for i in train_inters))
    print(f"[stats] #questions={n_q}  #train_interactions={n_train}  #train_students={uniq_students_train}  #test_interactions={n_test}")

    # Split by student_id for train/val to avoid leakage
    train_students = sorted(set(i.student_id for i in train_inters))
    random.shuffle(train_students)
    val_cut = max(1, int(len(train_students) * args.val_frac))
    val_students = set(train_students[:val_cut])
    tr_students = set(train_students[val_cut:])

    # Build JSONL lines
    print("[build] constructing train/val JSONL...")
    train_lines = build_jsonl_for_split(train_inters, questions, history_k=args.hist_k, students_for_split=tr_students)
    val_lines = build_jsonl_for_split(train_inters, questions, history_k=args.hist_k, students_for_split=val_students)

    train_jsonl_path = os.path.join(args.out_dir, "train.jsonl")
    val_jsonl_path = os.path.join(args.out_dir, "val.jsonl")
    write_jsonl(train_jsonl_path, train_lines)
    write_jsonl(val_jsonl_path, val_lines)
    print(f"[write] train.jsonl -> {train_jsonl_path}  ({len(train_lines)} examples)")
    print(f"[write] val.jsonl   -> {val_jsonl_path}    ({len(val_lines)} examples)")

    # Prepare eval prompts if test is available
    eval_jsonl_path = None
    if test_inters:
        print("[build] constructing eval prompts from test...")
        eval_prompts = build_eval_prompts(test_inters, questions, history_k=args.hist_k)
        eval_jsonl_path = os.path.join(args.out_dir, "eval_prompts.jsonl")
        write_jsonl(eval_jsonl_path, eval_prompts)
        print(f"[write] eval_prompts.jsonl -> {eval_jsonl_path}  ({len(eval_prompts)} examples)")

    if args.prepare_only:
        print("[done] Prepared JSONL files only (--prepare-only).")
        return
    input()

    # ----------------------------
    # Start fine-tuning
    # ----------------------------
    from openai import OpenAI
    client = OpenAI()

    print("[upload] uploading training/validation files to OpenAI...")
    train_file_id = upload_file(client, train_jsonl_path)
    val_file_id = upload_file(client, val_jsonl_path)
    print(f"[upload] training_file_id:   {train_file_id}")
    print(f"[upload] validation_file_id: {val_file_id}")

    print(f"[finetune] starting fine-tuning job on model={args.model}...")
    job_id = start_fine_tune(client, model=args.model, training_file_id=train_file_id, validation_file_id=val_file_id, suffix=args.suffix)
    print(f"[finetune] job_id: {job_id}")

    status, ft_model = wait_for_job(client, job_id)
    if status != "succeeded" or not ft_model:
        print(f"[finetune] Job finished with status={status}.")
        return

    print(f"[finetune] Succeeded. fine_tuned_model = {ft_model}")

    # ----------------------------
    # Evaluation
    # ----------------------------
    if args.skip_eval or not eval_jsonl_path:
        print("[eval] Skipped (no test file or --skip-eval).")
        return

    eval_prompts = read_jsonl(eval_jsonl_path)
    report = evaluate_model_on_prompts(client, ft_model, eval_prompts, max_items=args.eval_max)

    # Write report
    report_path = os.path.join(args.out_dir, "evaluation_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"[eval] overall_accuracy={report['overall_accuracy']:.4f} on n={report['n']}")
    print(f"[eval] by_difficulty={report['by_difficulty']}")
    print(f"[eval] report saved -> {report_path}")

if __name__ == "__main__":
    main()
