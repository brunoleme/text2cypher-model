import os
from unittest.mock import patch, MagicMock
import pandas as pd

from text2cypher.finetuning.evaluate_model import evaluate_model
from tests.utils import load_config_with_overrides, run_preprocessing_for_tests


def test_evaluation_pipeline(temp_output_dirs):
    run_preprocessing_for_tests(temp_output_dirs["preprocessed_input_data_folder"])

    # Create dummy checkpoint file
    model_artifacts_dir = temp_output_dirs["preprocessed_input_data_folder"] / "model-artifacts-dev"
    model_artifacts_dir.mkdir(exist_ok=True)
    dummy_checkpoint = model_artifacts_dir / "pipeline_id_best_model.ckpt"
    dummy_checkpoint.touch()

    # Prepare config
    cfg = load_config_with_overrides(
        data={
            "preprocessed_input_data_folder": "tests/resources/preprocessed",
            "source_data_folder": "tests/resources",
            "source_data_path": "source_data/notechat_sample_dataset.csv",
        },
        training={
            "model_artifact_dir": str(model_artifacts_dir),
        },
        evaluation={
            "reports_dir": str(temp_output_dirs["reports_dir"]),
            "test_samples_lexical_metrics": 2,
            "test_samples_semantic_metrics": 2,
            "test_samples_ai_as_judge_metrics": 2,
        }
    )

    os.environ["ENV"] = "dev"
    os.environ["PIPELINE_RUN_ID"] = "pipeline_id"

    with (
        patch("text2cypher.finetuning.evaluate_model.load_model") as mock_load_model,
        patch("text2cypher.finetuning.evaluate_model.compute_group_metrics") as mock_compute_metrics,
        patch("text2cypher.finetuning.evaluate_model.calculate_average_latency", return_value=0.1),
        patch("text2cypher.finetuning.evaluate_model.calculate_model_size_in_params", return_value=1234567)
    ):
        # Mock model and metric result
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model
        mock_compute_metrics.side_effect = [
            pd.DataFrame({"rouge_score": [0.0], "bleu_score": [0.0]}),
            pd.DataFrame({"bert_score": [0.847899]}),
            pd.DataFrame({
                "factual_consistency": [3.0],
                "relevance": [5.0],
                "completeness": [3.0],
                "conciseness": [3.666667],
                "clarity": [3.0],
            })
        ]

        evaluate_model(cfg)

    report_file = temp_output_dirs["reports_dir"] / "pipeline_id_eval_metrics.json"
    assert report_file.exists()
    assert report_file.read_text()
