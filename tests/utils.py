from omegaconf import OmegaConf
import os
from pathlib import Path

from text2cypher.finetuning.data.notechat_preprocessing import NoteChatDataPreprocessingModule


def load_config_with_overrides(**overrides):
    config_path = os.path.abspath("src/text2cypher/finetuning/config/config.dev.yaml")
    cfg = OmegaConf.load(config_path)

    for key, val in overrides.items():
        OmegaConf.update(cfg, key, val)
    return cfg

def run_preprocessing_for_tests(output_dir: Path):
    source_data_folder = "tests/resources"
    preprocessed_input_data_folder = str(output_dir)
    source_data_path = "source_data/notechat_sample_dataset.csv"
    env_folder = "dev"

    Path(preprocessed_input_data_folder).mkdir(parents=True, exist_ok=True)
    (Path(preprocessed_input_data_folder) / "preprocessed").mkdir(parents=True, exist_ok=True)

    preprocessingmodule = NoteChatDataPreprocessingModule(
        model_name="t5-small",
        source_data_folder=source_data_folder,
        source_data_path=source_data_path,
        preprocessed_output_data_folder=preprocessed_input_data_folder,
        env_folder=env_folder,
        max_length=128,
    )

    preprocessingmodule.run()

