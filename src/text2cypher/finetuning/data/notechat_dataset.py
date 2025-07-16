import os
from functools import partial
from typing import Dict, Optional

import pytorch_lightning as pl
from datasets import load_dataset
from loguru import logger
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorForSeq2Seq

from text2cypher.finetuning.utils.text_utils import replace_first_dash


class NoteChatDataModule(pl.LightningDataModule):
    def __init__(
        self,
        model_name: str,
        batch_size: int = 8,
        max_length: int = 128,
        max_source_length: int = 384,
        max_target_length: int = 128,
        train_samples: int = -1,
        val_samples: int = -1,
        test_samples: int = -1,
        train_split: float = 0.8,
        val_split: float = 0.1,
        test_split: float = 0.1,
        num_workers: int = 4,
        shuffle: bool = True,
        shuffle_seed: int = 42,
    ):
        super().__init__()
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.train_samples = train_samples
        self.val_samples = val_samples
        self.test_samples = test_samples
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.shuffle_seed = shuffle_seed

        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        logger.info(f"Initializing tokenizer with model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(replace_first_dash(model_name))

        special_tokens = {
            "additional_special_tokens": [
                "<conversation>", "</conversation>",
                "<note>", "</note>",
                "<speaker>Doctor:</speaker>",
                "<speaker>Patient:</speaker>",
            ]
        }
        self.tokenizer.add_special_tokens(special_tokens)

        self.data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=model_name,
            padding="longest",
            max_length=self.max_length,
            pad_to_multiple_of=8,
            return_tensors="pt"
        )

    def prepare_data(self):
        logger.info("Downloading NoteChat dataset if needed")
        load_dataset("akemiH/NoteChat")

    def setup(self, stage: Optional[str] = None):
        logger.info(f"Setting up dataset for stage: {stage}")
        full_dataset = load_dataset("akemiH/NoteChat")["train"]
        logger.debug(f"Dataset columns: {full_dataset.column_names}")

        n = len(full_dataset)
        assert abs(self.train_split + self.val_split + self.test_split - 1.0) < 1e-6, \
            "Train/val/test splits must sum to 1.0"

        train_end = int(n * self.train_split)
        val_end = train_end + int(n * self.val_split)

        self.train_dataset = full_dataset.select(range(0, train_end))
        self.val_dataset = full_dataset.select(range(train_end, val_end))
        self.test_dataset = full_dataset.select(range(val_end, n))

        if self.train_samples > 0:
            self.train_dataset = self.train_dataset.select(
                range(min(self.train_samples, len(self.train_dataset)))
            )

        if self.val_samples > 0:
            self.val_dataset = self.val_dataset.select(
                range(min(self.val_samples, len(self.val_dataset)))
            )

        if self.test_samples > 0:
            self.test_dataset = self.test_dataset.select(
                range(min(self.test_samples, len(self.test_dataset)))
            )

        self._log_dataset_statistics(self.train_dataset, "train")
        self._log_dataset_statistics(self.val_dataset, "validation")
        self._log_dataset_statistics(self.test_dataset, "test")

        logger.info(
            f"Train size: {len(self.train_dataset)}, "
            f"Validation size: {len(self.val_dataset)}, "
            f"Test size: {len(self.test_dataset)}"
        )

    def _log_dataset_statistics(self, dataset, split_name: str) -> None:
        conv_lengths = [len(x.split()) for x in dataset["conversation"]]
        note_lengths = [len(x.split()) for x in dataset["data"]]

        logger.info(f"{split_name.capitalize()} set statistics:")
        logger.info(
            f"Conversation lengths - Min: {min(conv_lengths)}, "
            f"Max: {max(conv_lengths)}, "
            f"Avg: {sum(conv_lengths)/len(conv_lengths):.2f}"
        )
        logger.info(
            f"Note lengths - Min: {min(note_lengths)}, "
            f"Max: {max(note_lengths)}, "
            f"Avg: {sum(note_lengths)/len(note_lengths):.2f}"
        )

    @staticmethod
    def format_conversation(conversation: str) -> str:
        turns = conversation.split("\n")
        formatted_turns = []
        for turn in turns:
            if turn.startswith("Doctor:"):
                formatted_turns.append(f"<speaker>Doctor:</speaker>{turn[7:]}")
            elif turn.startswith("Patient:"):
                formatted_turns.append(f"<speaker>Patient:</speaker>{turn[8:]}")
        return f"<conversation>{' '.join(formatted_turns)}</conversation>"

    @staticmethod
    def preprocess_function(
        examples: Dict,
        tokenizer: AutoTokenizer,
        max_source_length: int,
        max_target_length: int
    ) -> Dict:
        conversations = examples["conversation"]
        clinical_notes = examples["data"]

        formatted_conversations = [
            f"summarize: {NoteChatDataModule.format_conversation(conv)}"
            for conv in conversations
        ]

        model_inputs = tokenizer(
            formatted_conversations,
            padding=False,
            max_length=max_source_length,
            truncation=True,
        )

        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                clinical_notes,
                padding=False,
                max_length=max_target_length,
                truncation=True,
            )

        model_inputs["labels"] = labels["input_ids"]
        model_inputs["source_lengths"] = [len(x) for x in model_inputs["input_ids"]]
        model_inputs["target_lengths"] = [len(x) for x in labels["input_ids"]]
        return model_inputs

    def _create_dataloader(self, dataset, shuffle: bool = False) -> DataLoader:
        preprocess_fn = partial(
            self.preprocess_function,
            tokenizer=self.tokenizer,
            max_source_length=self.max_source_length,
            max_target_length=self.max_target_length,
        )

        processed_dataset = dataset.map(
            preprocess_fn,
            remove_columns=dataset.column_names,
            desc="Processing data...",
            batched=True,
            batch_size=1000,
            num_proc=None if os.environ.get("TOKENIZERS_PARALLELISM") == "false" else 4,
        )

        if "source_lengths" in processed_dataset.features:
            processed_dataset = processed_dataset.sort("source_lengths")
            processed_dataset = processed_dataset.remove_columns(["source_lengths", "target_lengths"])

        return DataLoader(
            processed_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=shuffle,
            collate_fn=self.data_collator,
            persistent_workers=self.num_workers > 0,
            pin_memory=False,
        )

    def train_dataloader(self) -> DataLoader:
        logger.info("Creating train dataloader")
        return self._create_dataloader(self.train_dataset, shuffle=self.shuffle)

    def val_dataloader(self) -> DataLoader:
        logger.info("Creating validation dataloader")
        return self._create_dataloader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        logger.info("Creating test dataloader")
        return self._create_dataloader(self.test_dataset, shuffle=False)
