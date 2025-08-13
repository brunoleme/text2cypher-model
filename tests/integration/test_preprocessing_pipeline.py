import os
from text2cypher.finetuning.preprocessing import preprocessing
from tests.utils import load_config_with_overrides


def test_preprocessing_pipeline(temp_output_dirs):
    cfg = load_config_with_overrides(
        data={
            "preprocessed_output_data_folder": str(temp_output_dirs["preprocessed_output_data_folder"]),
            "source_data_folder": "tests/resources",
            "source_data_path": "source_data/notechat_sample_dataset.csv",
        }
    )

    env_folder = "dev"
    os.environ["ENV"] = env_folder
    (temp_output_dirs["preprocessed_output_data_folder"]).mkdir(parents=True, exist_ok=True)
    
    preprocessing(cfg)

    expected_files = [
+        "preprocessed/notechat_sample_dataset_train.parquet",
+        "preprocessed/notechat_sample_dataset_val.parquet",
+        "preprocessed/notechat_sample_dataset_test.parquet",
    ]
    for file in expected_files:
        assert (temp_output_dirs["preprocessed_output_data_folder"] / file).exists()
