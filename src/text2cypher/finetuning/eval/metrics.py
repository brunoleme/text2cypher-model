from evaluate import load
import time
import pandas as pd
from typing import Dict, Callable, List
import numpy as np
import os
import re

from torch.utils.data import DataLoader

# Load evaluation metrics
rouge_metric = load("rouge")
bleu_metric = load("bleu")
bertscore_metric = load("bertscore")

# --- Basic metrics ---
def calculate_exact_match(predictions: List[str], references: List[str], _):
    try:
        scores = [1.0 if p.strip() == r.strip() else 0.0 for p, r in zip(predictions, references)]
        return float(np.mean(scores)) if scores else 0.0
    except Exception:
        return 0.0

# --- Classical metrics ---
def calculate_rouge(predictions, references, _):
    try:
        scores = rouge_metric.compute(predictions=predictions, references=references)
        return sum(scores.values()) / len(scores)
    except Exception:
        return 0

def calculate_bleu(predictions, references, _):
    try:
        return bleu_metric.compute(predictions=predictions, references=references)['bleu']
    except Exception:
        return 0

def calculate_bertscore(predictions, references, _):
    try:
        return np.mean(bertscore_metric.compute(predictions=predictions, references=references, lang="en")['f1'])
    except Exception:
        return 0

def calculate_cypher_lint_rate(predictions: List[str], _, __):
    """Simple Cypher lint: balanced parentheses/brackets and known starting keywords."""
    allowed_starts = ("MATCH", "RETURN", "WITH", "CALL", "CREATE", "MERGE", "UNWIND", "DELETE", "SET")
    def ok(q: str) -> bool:
        s = q.strip()
        if not s:
            return False
        up = s.upper()
        if not any(up.startswith(k) for k in allowed_starts):
            return False
        # balance () and []
        return s.count("(") == s.count(")") and s.count("[") == s.count("]")
    if not predictions:
        return 0.0
    return float(np.mean([1.0 if ok(p) else 0.0 for p in predictions]))

# --- Group Evaluation Entry Point ---
def compute_group_metrics_from_rows(model, rows: List[Dict], max_length: int, metrics_list: Dict):
    # Generate predictions
    preds = []
    refs = []
    instr = []
    for r in rows:
        q = r.get("question", "")
        sch = r.get("schema", None)
        ref = r.get("cypher", "")
        try:
            pred = model.generate_cypher(question=q, schema=sch, max_length=max_length)
        except Exception:
            pred = ""
        preds.append(pred)
        refs.append(ref)
        instr.append(q)

    result = {
        metric_name: metric_fn(preds, refs, instr)
        for metric_name, metric_fn in metrics_list.items()
    }
    return pd.DataFrame(result, index=[0])

# --- Misc Profiling Helpers ---
def calculate_model_size(model_ckpt_path: str) -> float:
    total = 0
    for dirpath, _, filenames in os.walk(model_ckpt_path):
        for f in filenames:
            total += os.path.getsize(os.path.join(dirpath, f))
    return round(total / (1024 * 1024), 2)

def calculate_model_size_in_params(model) -> int:
    return sum(p.numel() for p in model.model.parameters())

def calculate_average_latency(model, rows: List[Dict], max_length: int) -> float:
    latencies = []
    warm = rows[:1]
    for r in warm:
        _ = model.generate_cypher(question=r.get("question", ""), schema=r.get("schema", None), max_length=max_length)
    for r in rows:
        start = time.time()
        _ = model.generate_cypher(question=r.get("question", ""), schema=r.get("schema", None), max_length=max_length)
        latencies.append(time.time() - start)
    return round(sum(latencies) / len(latencies), 4) if latencies else None
