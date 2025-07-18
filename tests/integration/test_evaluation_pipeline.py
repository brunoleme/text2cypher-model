import pytest
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from omegaconf import OmegaConf
from unittest.mock import patch, MagicMock
from fakes import (
    FakeT5NoteGenerationModel,
    FakeNoteChatDataModule,
)
from text2cypher.finetuning import evaluate_model


@pytest.fixture
def dummy_cfg():
    return OmegaConf.create({
        "model": {
            "name": "t5-small",
            "type": "t5",
            "peft_method": None,
            "max_length": 128
        },
        "training": {
            "batch_size": 1,
            "num_workers": 0
        },
        "evaluation_model": {
            "checkpoint": "checkpoints/mock.ckpt",
            "display_name": "mock-model"
        },
        "evaluation": {
            "test_samples_lexical_metrics": 1,
            "test_samples_semantic_metrics": 1,
            "test_samples_ai_as_judge_metrics": 1,
        },
        "data": {
            "shuffle": False,
            "shuffle_seed": 42
        }
    })


@patch("text2cypher.finetuning.evaluate_model.wandb.Table")
@patch("text2cypher.finetuning.evaluate_model.wandb.log")
@patch("text2cypher.finetuning.evaluate_model.wandb.init")
@patch("text2cypher.finetuning.evaluate_model.load_model")
@patch("text2cypher.finetuning.evaluate_model.setup_dataloader")
@patch("text2cypher.finetuning.evaluate_model.compute_group_metrics")
@patch("text2cypher.finetuning.evaluate_model.calculate_average_latency", return_value=0.123)
@patch("text2cypher.finetuning.evaluate_model.calculate_model_size_in_params", return_value=123456)
def test_evaluate_model_pipeline(
    mock_model_size, mock_latency, mock_compute_metrics, mock_dataloader,
    mock_load_model, mock_wandb_init, mock_wandb_log, mock_wandb_table, dummy_cfg
):
    mock_model = MagicMock()
    mock_load_model.return_value = mock_model
    mock_dataloader.return_value = DataLoader(TensorDataset(torch.tensor([[1, 2, 3]])))
    mock_compute_metrics.return_value = pd.DataFrame({"mock_metric": [0.9]})

    run_mock = MagicMock()
    mock_wandb_init.return_value.__enter__.return_value = run_mock
    mock_wandb_init.return_value.__exit__.return_value = None

    evaluate_model.evaluate_model(dummy_cfg)

    mock_load_model.assert_called_once()
    assert mock_compute_metrics.call_count == 3
    mock_latency.assert_called_once()
    mock_model_size.assert_called_once()
    mock_wandb_log.assert_called_once()
    mock_wandb_table.assert_called_once()
    run_mock.finish.assert_called_once()


@patch("text2cypher.finetuning.evaluate_model.load_model")
@patch("text2cypher.finetuning.evaluate_model.NoteChatDataModule", FakeNoteChatDataModule)
@patch("wandb.init")
@patch("wandb.log")
def test_evaluate_model_runs(mock_log, mock_wandb_init, mock_load_model):
    mock_load_model.return_value = FakeT5NoteGenerationModel()

    # Simulate Hydra config
    class DummyCfg:
        class Model:
            name = "t5-small"
            type = "t5"
            max_length = 32
            peft_method = "lora"

        class Training:
            batch_size = 2
            num_workers = 0

        class Data:
            shuffle = False
            shuffle_seed = 42
            input_data_uri = "tests/resources/notechat_sample_dataset.csv"

        class Evaluation:
            test_samples_lexical_metrics = 2
            test_samples_semantic_metrics = 2
            test_samples_ai_as_judge_metrics = 1

        class EvaluationModel:
            checkpoint = "checkpoints/mock.ckpt"
            display_name = "mock-t5"

        model = Model()
        training = Training()
        data = Data()
        evaluation = Evaluation()
        evaluation_model = EvaluationModel()

    evaluate_model.evaluate_model(DummyCfg())
    mock_log.assert_called()
