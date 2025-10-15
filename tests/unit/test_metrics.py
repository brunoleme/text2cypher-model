import pytest
from text2cypher.finetuning.eval import metrics
import torch

@pytest.fixture
def dummy_data():
    predictions = ["The patient has a mild fever."]
    references = ["The patient has a mild fever."]
    instructions = ["Summarize the conversation about the patient's symptoms."]
    return predictions, references, instructions

@pytest.mark.parametrize("metric_fn", [
    metrics.calculate_rouge,
    metrics.calculate_bleu,
    metrics.calculate_bertscore
])
def test_classical_metrics(metric_fn, dummy_data):
    pred, ref, instr = dummy_data
    score = metric_fn(pred, ref, instr)
    assert isinstance(score, float)
    assert 0 <= score <= 1

def test_cypher_lint_and_exact_match(dummy_data):
    predictions = [
        "MATCH (n) RETURN n",  # valid
        "return n",            # invalid start
        "MATCH (n RETURN n"    # unbalanced
    ]
    references = [
        "MATCH (n) RETURN n",
        "MATCH (n) RETURN n",
        "MATCH (n) RETURN n"
    ]
    instr = ["q1", "q2", "q3"]
    em = metrics.calculate_exact_match(predictions, references, instr)
    assert 0 <= em <= 1
    lint = metrics.calculate_cypher_lint_rate(predictions, references, instr)
    assert 0 <= lint <= 1

def test_calculate_model_size_in_params():
    class Dummy:
        model = type("Model", (), {"parameters": lambda self: [torch.nn.Parameter(torch.randn(10, 10))]})()
    assert isinstance(metrics.calculate_model_size_in_params(Dummy()), int)
