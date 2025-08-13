import os
from pathlib import Path
from text2cypher.finetuning.train import train
from tests.utils import load_config_with_overrides, run_preprocessing_for_tests


def test_training_pipeline(temp_output_dirs):
    run_preprocessing_for_tests(temp_output_dirs["preprocessed_input_data_folder"])

    cfg = load_config_with_overrides(
        data={
            "preprocessed_input_data_folder": "tests/resources/preprocessed",
            "source_data_folder": "tests/resources",
            "source_data_path": "source_data/notechat_sample_dataset.csv",
            },
        training={"model_artifact_dir": "tests/resources/artifacts"}
    )

    env_folder = "dev"
    os.environ["ENV"] = env_folder
    os.environ["PIPELINE_RUN_ID"] = "pipeline_id"
    train(cfg)

    ckpts = list(Path(cfg.training.model_artifact_dir).glob("pipeline_id_best_model.ckpt"))
    assert len(ckpts) == 1
    assert ckpts[0].stat().st_size > 0
