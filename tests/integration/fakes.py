# tests/integration/fakes.py

import torch
from torch.utils.data import DataLoader, Dataset
from types import SimpleNamespace


# === 1. Fake Tokenizer ===
class FakeTokenizer:
    def decode(self, tokens, skip_special_tokens=True):
        return "decoded input" if isinstance(tokens, list) else "decoded label"


# === 2. Fake T5NoteGenerationModel ===
class FakeT5NoteGenerationModel:
    def __init__(self):
        self.tokenizer = FakeTokenizer()
        self.model = SimpleNamespace()
        self.model.parameters = lambda: [torch.nn.Parameter(torch.randn(10, 10))]

    def generate_notes(self, dataloader, max_length: int):
        return ["Generated note"] * len(dataloader)

    def generate_note(self, conversation: str, max_length: int):
        return "Generated note"


# === 3. Fake Dataset (used in DataLoader) ===
class FakeDataset(Dataset):
    def __init__(self, size=3):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor([1, 2, 3, 4]),
            "attention_mask": torch.tensor([1, 1, 1, 1]),
            "labels": torch.tensor([5, 6, 7, 8]),
        }


# === 4. Fake NoteChatDataModule ===
class FakeNoteChatDataModule:
    def __init__(self, *args, **kwargs):
        self.dataset = FakeDataset()

    def setup(self):
        pass

    def test_dataloader(self):
        return DataLoader(self.dataset, batch_size=1)
