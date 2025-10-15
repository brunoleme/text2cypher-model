import os
from pathlib import Path

from text2cypher.finetuning.data.text2cypher_preprocessing import Text2CypherDataPreprocessingModule

def run_preprocessing_for_tests():
    source_data_folder = "tests/resources"
    preprocessed_input_data_folder = "tests/resources"
    # Use separate train/test local files to mirror S3 layout
    source_train_data_path = "text2cypher_train.parquet"
    source_test_data_path = "text2cypher_test.parquet"
    env_folder = "dev"

    preprocessingmodule = Text2CypherDataPreprocessingModule(
        model_name="meta-llama/Llama-3.2-1B-Instruct",
        source_data_folder=source_data_folder,
        source_data_path="text2cypher.parquet",  # base name for outputs
        source_train_data_path=source_train_data_path,
        source_test_data_path=source_test_data_path,
        preprocessed_output_data_folder=preprocessed_input_data_folder,
        env_folder=env_folder,
        max_length=128,
        val_from_test_ratio=0.5,
        # Relax quality filters for tiny test datasets
        apply_quality_filters=True,
        min_question_length=1,
        min_cypher_length=1,
        max_length_ratio=10.0,
        min_length_ratio=0.0,
    )

    preprocessingmodule.run()

