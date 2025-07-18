import pytest
import pandas as pd
from datasets import Dataset
from text2cypher.finetuning.data.notechat_dataset import NoteChatDataModule

source_data_path = "tests/resources/notechat_sample_dataset.csv"

def test_dataset_initialization() -> None:
    dataset = NoteChatDataModule(model_name="t5-small", source_data_path=source_data_path, batch_size=2, max_length=128, )
    assert dataset.model_name == "t5-small"
    assert dataset.batch_size == 2
    assert dataset.max_length == 128

def test_conversation_formatting() -> None:
    dataset = NoteChatDataModule(model_name="t5-small", source_data_path=source_data_path)
    conversation = "Doctor: Hello\nPatient: Hi"
    formatted = dataset.format_conversation(conversation)
    expected = "<conversation><speaker>Doctor:</speaker>Hello <speaker>Patient:</speaker>Hi</conversation>"
    assert formatted.replace(" ", "") == expected.replace(" ", "")

def test_preprocess_function() -> None:
    dataset = NoteChatDataModule(model_name="t5-small", source_data_path=source_data_path)
    examples = {
        "conversation": ["Doctor: Hello\nPatient: Hi"],
        "data": ["Patient visited for checkup"],
    }
    result = dataset.preprocess_function(
        examples,
        dataset.tokenizer,
        max_source_length=128,
        max_target_length=128,
    )
    assert "input_ids" in result
    assert "labels" in result

@pytest.mark.parametrize("token", [
    "<conversation>", "</conversation>",
    "<note>", "</note>",
    "<speaker>Doctor:</speaker>", "<speaker>Patient:</speaker>",
])
def test_special_tokens_exist(token):
    tokenizer = NoteChatDataModule("t5-small", source_data_path=source_data_path).tokenizer
    assert tokenizer.convert_tokens_to_ids(token) != tokenizer.unk_token_id

def test_data_splitting_with_mocker(mocker) -> None:
    dummy_data = {
        "conversation": ["Doctor: Hi\nPatient: Hello"] * 100,
        "data": ["Note"] * 100,
    }
    dummy_df = pd.DataFrame(dummy_data)
    mocker.patch("pandas.read_csv", return_value=dummy_df)

    module = NoteChatDataModule("t5-small", source_data_path=source_data_path, train_samples=10, val_samples=5, test_samples=5)
    module.prepare_data()
    module.setup()

    assert len(module.train_dataset) == 10
    assert len(module.val_dataset) == 5
    assert len(module.test_dataset) == 5

def test_dataloader_creation_with_mock(mocker) -> None:
    dummy_data = Dataset.from_dict({
        "conversation": ["Doctor: Hi\nPatient: Hello"] * 10,
        "data": ["Note"] * 10,
    })
    dummy_df = pd.DataFrame(dummy_data)
    mocker.patch("pandas.read_csv", return_value=dummy_df)

    module = NoteChatDataModule("t5-small", source_data_path=source_data_path, train_samples=5)
    module.setup()

    loader = module.train_dataloader()
    batch = next(iter(loader))
    assert "input_ids" in batch
    assert "labels" in batch

def test_formatting_with_unknown_prefix():
    conversation = "Nurse: Hello"
    formatted = NoteChatDataModule.format_conversation(conversation)
    assert formatted == "<conversation></conversation>"
