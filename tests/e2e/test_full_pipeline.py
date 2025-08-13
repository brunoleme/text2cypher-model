import os
from text2cypher.finetuning.preprocessing import preprocessing
from text2cypher.finetuning.train import train
from text2cypher.finetuning.evaluate_model import evaluate_model
from tests.utils import load_config_with_overrides

def test_full_pipeline(temp_output_dirs):
    preproc_root = temp_output_dirs["preprocessed_output_data_folder"]
    cfg = load_config_with_overrides(
        data={
            "preprocessed_output_data_folder": str(preproc_root),
            "source_data_folder": "tests/resources",
            "source_data_path": "source_data/notechat_sample_dataset.csv",
            "preprocessed_input_data_folder": str(preproc_root),
        },
        training={"model_artifact_dir": "tests/resources/artifacts"},
        evaluation={
            "reports_dir": str(temp_output_dirs["reports_dir"]),
            "test_samples_lexical_metrics": 2,
            "test_samples_semantic_metrics": 2,
            "test_samples_ai_as_judge_metrics": 2,
            "model_artifact_dir": "tests/resources/artifacts",
        }
    )

    env_folder = "dev"
    os.environ["ENV"] = env_folder
    os.environ["PIPELINE_RUN_ID"] = "pipeline_id"

    preproc_root.mkdir(parents=True, exist_ok=True)
    (temp_output_dirs["reports_dir"]).mkdir(parents=True, exist_ok=True)

    preprocessing(cfg)
    train(cfg)
    evaluate_model(cfg)

    report_file = temp_output_dirs["reports_dir"] / "pipeline_id_eval_metrics.json"
    assert report_file.exists()
    assert report_file.read_text()