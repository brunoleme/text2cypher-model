# inference.py
import json
import os
import re
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import time

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def format_conversation(conversation: str) -> str:
    turns = conversation.split("\n")
    formatted_turns = []
    for turn in turns:
        if turn.startswith("Doctor:"):
            formatted_turns.append(f"<speaker>Doctor:</speaker>{turn[7:]}")
        elif turn.startswith("Patient:"):
            formatted_turns.append(f"<speaker>Patient:</speaker>{turn[8:]}")
    return f"summarize: <conversation>{' '.join(formatted_turns)}</conversation>"

def clean_conversation(text: str) -> str:
    """
    Clean and normalize a clinical conversation string.
    - Removes control characters (except newline/tab)
    - Normalizes quotes
    - Replaces excessive whitespace
    """
    text = re.sub(r"[\x00-\x09\x0B-\x1F\x7F]+", "", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = (
        text.replace("“", '"')
            .replace("”", '"')
            .replace("‘", "'")
            .replace("’", "'")
    )
    text = re.sub(r"\s+", " ", text).strip()
    return text

def model_fn(model_dir):
    """
    Load HF model and tokenizer from the model_dir.
    """
    print(f"[INFO] Loading model on device: {DEVICE}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to(DEVICE)
    return {"tokenizer": tokenizer, "model": model}


def input_fn(request_body, content_type="application/json"):
    """
    Parse and preprocess input request.
    """
    if content_type != "application/json":
        raise ValueError(f"Unsupported content type: {content_type}")
    
    data = json.loads(request_body)
    conversation = data.get("conversation")
    max_length = int(data.get("max_length", 512))

    if not conversation:
        raise ValueError("Missing or empty 'conversation' in request body")

    # Preprocess input
    formatted = format_conversation(conversation)
    cleaned = clean_conversation(formatted)

    return {"conversation": cleaned, "max_length": max_length}

def predict_fn(inputs, model_artifacts):
    """
    Run note generation.
    """
    tokenizer = model_artifacts["tokenizer"]
    model = model_artifacts["model"]
    conversation = inputs["conversation"]
    max_length = inputs["max_length"]

    print(f"[INFO] data on device: {DEVICE}")

    encoded = tokenizer(
        conversation,
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
        padding=True,
    ).to(DEVICE)

    model.eval()
    with torch.no_grad():
        start = time.time()
        outputs = model.generate(
            **encoded,
            max_length=max_length + 2,
            num_beams=4,
            length_penalty=2.0,
            early_stopping=True,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
        )
        print(f"⏱ Inference took {time.time() - start:.2f} seconds")

    note = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"clinical_note": note}

def output_fn(prediction, accept="application/json"):
    """
    Return model output as JSON.
    """
    return json.dumps(prediction), accept
