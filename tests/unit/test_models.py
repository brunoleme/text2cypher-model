import pytest
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from text2cypher.finetuning.models.llama_model import LlamaText2CypherModel


@pytest.fixture
def default_llama_model() -> LlamaText2CypherModel:
    return LlamaText2CypherModel(
        model_name="meta-llama/Llama-3.2-1B-Instruct",
        model_type="llama",
        use_quantization=False,
    )

def test_llama_model_initialization():
    model = LlamaText2CypherModel(
        model_name="meta-llama/Llama-3.2-1B-Instruct",
        model_type="llama",
        use_quantization=False,
    )
    assert model.model_name == "meta-llama/Llama-3.2-1B-Instruct"
    assert model.model_type == "llama"
    assert hasattr(model, "model")
    assert hasattr(model, "tokenizer")



def test_model_generate_cypher(default_llama_model) -> None:
    question = "List all movies"
    cypher = default_llama_model.generate_cypher(question, max_length=50)
    assert isinstance(cypher, str)

class DummyDataset(Dataset):
    def __init__(self):
        self.samples = [
            {"question": "List all movies"},
            {"question": "Find top rated movies"},
        ]
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        return self.samples[idx]

def test_generate_cyphers_batch(default_llama_model) -> None:
    dataset = DummyDataset()
    dataloader = DataLoader(dataset, batch_size=1)

    cyphers = default_llama_model.generate_cyphers(dataloader, max_length=20)
    assert isinstance(cyphers, list)
    assert all(isinstance(n, str) for n in cyphers)

def test_is_loaded(default_llama_model):
    assert isinstance(default_llama_model.is_loaded(), bool)

def test_generate_cypher_with_empty_input(default_llama_model):
    cypher = default_llama_model.generate_cypher("", max_length=50)
    assert isinstance(cypher, str)

@pytest.mark.parametrize("peft_method", ["lora"])  # prompt_tuning disabled in tests due to cache handling
def test_llama_model_with_peft(peft_method):
    model = LlamaText2CypherModel(
        model_name="meta-llama/Llama-3.2-1B-Instruct",
        model_type="llama",
        peft_method=peft_method,
        use_quantization=False
    )

    assert model.peft_method == peft_method
    assert model.peft_config is not None
    assert hasattr(model.model, "base_model")

    cypher = model.generate_cypher("List all movies", max_length=50)
    assert isinstance(cypher, str)
