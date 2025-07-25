from text2cypher.finetuning.train import train
from tests.utils import load_config_with_overrides, run_preprocessing_for_tests


def test_training_pipeline(temp_output_dirs):
    run_preprocessing_for_tests(temp_output_dirs["preprocessed_input_data_folder"])

    cfg = load_config_with_overrides(
        data={"preprocessed_input_data_folder": str(temp_output_dirs["preprocessed_input_data_folder"])},
        training={"model_artifact_dir": str(temp_output_dirs["model_artifact_dir"])}
    )

    train(cfg)

    ckpts = list(temp_output_dirs["model_artifact_dir"].glob("*best_model.ckpt"))
    assert len(ckpts) == 1
