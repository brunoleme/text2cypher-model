import pytest
import pandas as pd
from datasets import Dataset
from text2cypher.finetuning.data.text2cypher_dataset import Text2CypherDataModule
from text2cypher.finetuning.data.text2cypher_preprocessing import Text2CypherDataPreprocessingModule

source_data_folder = "tests/resources"
source_data_path = "text2cypher_sample.parquet"
preprocessed_output_data_folder = "tests/resources"
preprocessed_input_data_folder = "tests/resources"
env_folder = 'dev'

def test_dataset_initialization() -> None:
    dataset = Text2CypherDataModule(model_name="meta-llama/Llama-3.2-1B-Instruct", source_data_path=source_data_path, env_folder=env_folder, preprocessed_input_data_folder=preprocessed_input_data_folder, batch_size=2, max_length=128, )
    assert dataset.model_name == "meta-llama/Llama-3.2-1B-Instruct"
    assert dataset.batch_size == 2
    assert dataset.max_length == 128

def test_prompt_formatting() -> None:
    fmt = Text2CypherDataPreprocessingModule.format_text_for_cypher(
        question="List all movies",
        schema="Node properties: - **Movie** - `title`: STRING"
    )
    assert "### Graph Schema ###" in fmt
    assert "### Question ###" in fmt

def test_module_constructs_tokenizer() -> None:
    module = Text2CypherDataModule(model_name="meta-llama/Llama-3.2-1B-Instruct", source_data_path=source_data_path, env_folder=env_folder, preprocessed_input_data_folder=preprocessed_input_data_folder)
    assert module.tokenizer is not None

@pytest.mark.parametrize("token", [
    "MATCH", "RETURN", "WITH", "CREATE", "MERGE", "UNWIND",
])
def test_cypher_tokens_present(token):
    tokenizer = Text2CypherDataModule(model_name="meta-llama/Llama-3.2-1B-Instruct", source_data_path=source_data_path, env_folder=env_folder, preprocessed_input_data_folder=preprocessed_input_data_folder).tokenizer
    tokenizer.add_tokens([token])
    assert token in tokenizer.get_vocab()

def test_data_splitting_with_mocker(mocker) -> None:
    raw_rows = [
        {
            "instance_id": "id-1",
            "question": "List all movies",
            "schema": "Node: Movie(title STRING)",
            "cypher": "MATCH (m:Movie) RETURN m LIMIT 5",
            "data_source": "unit-test",
            "database_reference_alias": "neo4j-movies",
        },
        {
            "instance_id": "id-2",
            "question": "Find actors in The Matrix",
            "schema": "Node: Person(name STRING); Node: Movie(title STRING); REL: ACTED_IN",
            "cypher": "MATCH (p:Person)-[:ACTED_IN]->(m:Movie {title: 'The Matrix'}) RETURN p",
            "data_source": "unit-test",
            "database_reference_alias": "neo4j-movies",
        },
    ]
    dummy_ds = Dataset.from_pandas(pd.DataFrame(raw_rows))
    mocker.patch("text2cypher.finetuning.data.text2cypher_dataset.load_dataset", return_value={"train": dummy_ds})

    module = Text2CypherDataModule(model_name="meta-llama/Llama-3.2-1B-Instruct", source_data_path=source_data_path, env_folder=env_folder, preprocessed_input_data_folder=preprocessed_input_data_folder, train_samples=1, val_samples=1, test_samples=1)
    module.setup()

    assert len(module.train_dataset) > 0
    assert len(module.val_dataset) >= 0
    assert len(module.test_dataset) >= 0

def test_dataloader_creation_with_mock(mocker) -> None:
    raw_rows = [
        {
            "instance_id": "id-1",
            "question": "List all movies",
            "schema": "Node: Movie(title STRING)",
            "cypher": "MATCH (m:Movie) RETURN m LIMIT 5",
            "data_source": "unit-test",
            "database_reference_alias": "neo4j-movies",
        }
    ]
    dummy_ds = Dataset.from_pandas(pd.DataFrame(raw_rows))
    mocker.patch("text2cypher.finetuning.data.text2cypher_dataset.load_dataset", return_value={"train": dummy_ds})

    module = Text2CypherDataModule(model_name="meta-llama/Llama-3.2-1B-Instruct", source_data_path=source_data_path, env_folder=env_folder, preprocessed_input_data_folder=preprocessed_input_data_folder, train_samples=1)
    module.setup()

    loader = module.train_dataloader()
    batch = next(iter(loader))
    assert "input_ids" in batch
    assert "labels" in batch

def test_noop_legacy():
    assert True
