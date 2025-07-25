import os
from text2cypher.finetuning.evaluate_model import evaluate_model
from tests.utils import load_config_with_overrides, run_preprocessing_for_tests


def test_evaluation_pipeline(temp_output_dirs):
    run_preprocessing_for_tests(temp_output_dirs["preprocessed_input_data_folder"])

    (temp_output_dirs["preprocessed_input_data_folder"] / "model-artifacts-dev").mkdir(exist_ok=True)
    dummy_checkpoint = temp_output_dirs["preprocessed_input_data_folder"] / "model-artifacts-dev/dummy_best_model.ckpt"
    dummy_checkpoint.touch()

    cfg = load_config_with_overrides(
        data={"preprocessed_input_data_folder": str(temp_output_dirs["preprocessed_input_data_folder"])},
        evaluation={"test_samples_lexical_metrics": 2},
    )

    os.environ["ENV"] = "dev"

    evaluate_model(cfg)

    report_file = temp_output_dirs["reports_dir"] / "eval_metrics.json"
    assert report_file.exists()
    assert report_file.read_text()
