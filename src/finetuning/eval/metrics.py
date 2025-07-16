from evaluation import load
import time
import pandas as pd
from typing import Dict
import numpy as np
import os

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from eval.prompts import (
    factual_consistency_prompt,
    relevance_prompt,
    completeness_prompt,
    conciseness_prompt,
    clarity_prompt,
)
from torch.utils.data import DataLoader
from loguru import logger

# Load evaluation metrics
rouge_metric = load("rouge")
bleu_metric = load("bleu")
bertscore_metric = load("bertscore")

def calculate_rouge(predictions, references, instructions):
    rouge_scores = rouge_metric.compute(predictions=predictions, references=references)
    return sum(rouge_scores.values()) / len(rouge_scores) if rouge_scores else 0

def calculate_bleu(predictions, references, instructions):
    return bleu_metric.compute(predictions=predictions, references=references)['bleu']

def calculate_bertscore(predictions, references, instructions):
    return np.mean(bertscore_metric.compute(predictions=predictions, references=references, lang="en")['f1'])

def calculate_relevance(predictions, references, instructions):
    api_key = os.getenv("OPENAI_API_KEY")
    llm_oai = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key=api_key).bind_tools([ResponseFormatter])
    relevance_grader = relevance_prompt | llm_oai
    grades = []

    for prediction, reference, instruction in zip(predictions, references, instructions):
        prompt_input = {
            "instruction": instruction,
            "ground_truth_note": reference,
            "generated_note": prediction,
        }
        response = relevance_grader.invoke(prompt_input)
        try:
            score = float(response.tool_calls[0]["args"]["score"])
            if 1 <= score <= 5:
                grades.append(score)
        finally:
            continue
    return np.mean(grades)

def calculate_factual_consistency(predictions, references, instructions):
    api_key = os.getenv("OPENAI_API_KEY")
    llm_oai = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key=api_key).bind_tools([ResponseFormatter])
    grader = factual_consistency_prompt | llm_oai
    grades = []

    for prediction, instruction in zip(predictions, instructions):
        prompt_input = {
            "instruction": instruction,
            "generated_note": prediction,
        }
        response = grader.invoke(prompt_input)
        try:
            score = float(response.tool_calls[0]["args"]["score"])
            if 1 <= score <= 5:
                grades.append(score)
        finally:
            continue
    return np.mean(grades)

def calculate_completeness(predictions, references, instructions):
    api_key = os.getenv("OPENAI_API_KEY")
    llm_oai = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key=api_key).bind_tools([ResponseFormatter])
    grader = completeness_prompt | llm_oai
    grades = []

    for prediction, instruction in zip(predictions, instructions):
        prompt_input = {
            "instruction": instruction,
            "generated_note": prediction,
        }
        response = grader.invoke(prompt_input)
        try:
            score = float(response.tool_calls[0]["args"]["score"])
            if 1 <= score <= 5:
                grades.append(score)
        finally:
            continue
    return np.mean(grades)

def calculate_clarity(predictions, references, instructions):
    api_key = os.getenv("OPENAI_API_KEY")
    llm_oai = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key=api_key).bind_tools([ResponseFormatter])
    grader = clarity_prompt | llm_oai
    grades = []

    for prediction, instruction in zip(predictions, instructions):
        prompt_input = {
            "instruction": instruction,
            "generated_note": prediction,
        }
        response = grader.invoke(prompt_input)
        try:
            score = float(response.tool_calls[0]["args"]["score"])
            if 1 <= score <= 5:
                grades.append(score)
        finally:
            continue
    return np.mean(grades)

def calculate_conciseness(predictions, references, instructions):
    api_key = os.getenv("OPENAI_API_KEY")
    llm_oai = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key=api_key).bind_tools([ResponseFormatter])
    grader = conciseness_prompt | llm_oai
    grades = []

    for prediction, instruction in zip(predictions, instructions):
        prompt_input = {
            "instruction": instruction,
            "generated_note": prediction,
        }
        response = grader.invoke(prompt_input)
        try:
            score = float(response.tool_calls[0]["args"]["score"])
            if 1 <= score <= 5:
                grades.append(score)
        finally:
            continue
    return np.mean(grades)

def compute_group_metrics(model, dataloader: DataLoader, device: str, max_length: int, metrics_list: Dict):
    tokenizer = model.tokenizer
    predictions = model.generate_notes(dataloader, max_length=max_length)

    all_references = []
    all_instructions = []

    for batch in dataloader:
        inputs = {key: value.to(device) for key, value in batch.items()}
        batch_references = [
            tokenizer.decode(reference.tolist(), skip_special_tokens=False)
            for reference in inputs["labels"]
            if -100 not in reference.tolist()
        ]
        all_references.extend(batch_references)

        batch_instructions = [
            tokenizer.decode(input_id, skip_special_tokens=False)
            for input_id in inputs["input_ids"]
        ]
        all_instructions.extend(batch_instructions)

    result = {
        metric_name: metric_fn(predictions, all_references, all_instructions)
        for metric_name, metric_fn in metrics_list.items()
    }
    result_df = pd.DataFrame(result, index=[0])
    return result_df

class ResponseFormatter(BaseModel):
    """Always use this tool to structure your response to the user."""
    score: int = Field(description="The score measured by the evaluation model")

def calculate_model_size(model_ckpt_path: str) -> float:
    total = 0
    for dirpath, _, filenames in os.walk(model_ckpt_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total += os.path.getsize(fp)
    return round(total / (1024 * 1024), 2)

def calculate_model_size_in_params(model) -> int:
    return sum(p.numel() for p in model.model.parameters())

def calculate_average_latency(model, dataloader, max_length: int) -> float:
    latencies = []

    for batch in dataloader:
        instructions = [
            model.tokenizer.decode(input_id, skip_special_tokens=True)
            for input_id in batch["input_ids"]
        ]
        _ = model.generate_note(conversation=instructions[0], max_length=max_length)

        for instruction in instructions:
            start = time.time()
            _ = model.generate_note(conversation=instruction, max_length=max_length)
            end = time.time()
            latencies.append(end - start)

    return round(sum(latencies) / len(latencies), 4) if latencies else None
