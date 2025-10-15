import os
from functools import partial
from typing import Dict, Any, Optional
import re

import pandas as pd
from datasets import Dataset
from loguru import logger

from text2cypher.finetuning.utils.tokenization_utils import normalize_cypher_query


class Text2CypherDataPreprocessingModule:
    def __init__(
        self,
        model_name: str,
        source_data_folder: str,
        source_data_path: str,
        preprocessed_output_data_folder: str,
        env_folder: str,
        source_train_data_path: Optional[str] = None,
        source_test_data_path: Optional[str] = None,
        max_length: int = 1536,
        max_source_length: int = 1024,  # For question + schema
        max_target_length: int = 512,   # For Cypher query
        train_samples: int = -1,
        val_samples: int = -1,
        test_samples: int = -1,
        train_split: float = 0.7,
        val_split: float = 0.15,
        test_split: float = 0.15,
        val_from_test_ratio: float = 0.2,
        shuffle: bool = True,
        shuffle_seed: int = 42,
        # Data quality filtering parameters
        apply_quality_filters: bool = True,
        min_question_length: int = 20,
        min_cypher_length: int = 10,
        max_length_ratio: float = 0.8,
        min_length_ratio: float = 0.01,
    ):
        self.model_name = model_name
        self.source_data_folder = source_data_folder
        self.source_data_path = source_data_path
        self.preprocessed_output_data_folder = preprocessed_output_data_folder
        self.env_folder = env_folder
        self.max_length = max_length
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.train_samples = train_samples
        self.val_samples = val_samples
        self.test_samples = test_samples
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.val_from_test_ratio = val_from_test_ratio
        self.source_train_data_path = source_train_data_path
        self.source_test_data_path = source_test_data_path
        self.shuffle = shuffle
        self.shuffle_seed = shuffle_seed
        # Data quality filtering parameters
        self.apply_quality_filters = apply_quality_filters
        self.min_question_length = min_question_length
        self.min_cypher_length = min_cypher_length
        self.max_length_ratio = max_length_ratio
        self.min_length_ratio = min_length_ratio

    def _filter_low_quality_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply quality filters to remove low-quality training examples.
        
        Filters applied:
        1. Minimum question length (in words)
        2. Minimum Cypher query length (in words)
        3. Length ratio constraints (Cypher shouldn't be too long/short relative to question)
        4. Remove corrupted/placeholder data
        5. Remove duplicates
        """
        if not self.apply_quality_filters:
            logger.info("Quality filtering disabled, returning original data")
            return df

        initial_count = len(df)
        logger.info(f"Applying quality filters to {initial_count} samples")
        
        # Filter 1: Minimum question length
        question_word_counts = df['question'].str.split().str.len()
        df = df[question_word_counts >= self.min_question_length]
        after_min_question = len(df)
        logger.info(f"After min question length filter ({self.min_question_length} words): {after_min_question} samples")
        
        # Filter 2: Minimum Cypher query length
        cypher_word_counts = df['cypher'].str.split().str.len()
        df = df[cypher_word_counts >= self.min_cypher_length]
        after_min_cypher = len(df)
        logger.info(f"After min Cypher length filter ({self.min_cypher_length} words): {after_min_cypher} samples")
        
        # Filter 3: Length ratio constraints
        question_lengths = df['question'].str.len()
        cypher_lengths = df['cypher'].str.len()
        length_ratios = cypher_lengths / question_lengths
        
        # Remove samples where Cypher is too long relative to question
        df = df[length_ratios <= self.max_length_ratio]
        after_max_ratio = len(df)
        logger.info(f"After max length ratio filter ({self.max_length_ratio}): {after_max_ratio} samples")
        
        # Remove samples where Cypher is too short relative to question
        df = df[length_ratios >= self.min_length_ratio]
        after_min_ratio = len(df)
        logger.info(f"After min length ratio filter ({self.min_length_ratio}): {after_min_ratio} samples")
        
        # Filter 4: Remove corrupted/placeholder data
        corrupted_patterns = [
            r'\[No meaningful.*found\]',
            r'^(MATCH|RETURN|WHERE|CREATE)?\s*$',  # Empty or single-keyword queries
            r'^\.+$',  # Just dots
            r'^N/A$|^n/a$|^NULL$|^null$',  # Placeholder values
        ]
        
        for pattern in corrupted_patterns:
            initial_size = len(df)
            df = df[~df['cypher'].str.contains(pattern, regex=True, case=False, na=False)]
            final_size = len(df)
            if initial_size != final_size:
                logger.info(f"Removed {initial_size - final_size} samples matching pattern: {pattern}")
        
        after_corruption_filter = len(df)
        
        # Filter 5: Remove duplicates based on question text
        df = df.drop_duplicates(subset=['question'], keep='first')
        after_dedup = len(df)
        logger.info(f"After deduplication: {after_dedup} samples (removed {after_corruption_filter - after_dedup} duplicates)")
        
        # Summary
        removed_count = initial_count - after_dedup
        removal_percentage = (removed_count / initial_count) * 100
        logger.info(f"Quality filtering summary:")
        logger.info(f"  Initial samples: {initial_count}")
        logger.info(f"  Final samples: {after_dedup}")
        logger.info(f"  Removed samples: {removed_count} ({removal_percentage:.1f}%)")
        
        return df

    def load_data(self) -> None:
        expected_columns = {"question", "schema", "cypher"}
        
        if self.source_train_data_path and self.source_test_data_path:
            # Load separate train and test parquet files
            train_path = f"{self.source_data_folder}/{self.source_train_data_path}"
            test_path = f"{self.source_data_folder}/{self.source_test_data_path}"
            logger.info(f"Reading train parquet: {train_path}")
            logger.info(f"Reading test parquet:  {test_path}")

            df_train = pd.read_parquet(train_path)
            df_test = pd.read_parquet(test_path)

            if not expected_columns.issubset(df_train.columns):
                raise ValueError("Train parquet missing required columns: question, schema, cypher")
            if not expected_columns.issubset(df_test.columns):
                raise ValueError("Test parquet missing required columns: question, schema, cypher")

            # Drop NAs and apply quality filters
            df_train = df_train.dropna(subset=["question", "cypher"]) \
                               .assign()
            df_test = df_test.dropna(subset=["question", "cypher"]) \
                             .assign()

            df_train = self._filter_low_quality_data(df_train)
            df_test = self._filter_low_quality_data(df_test)

            # Derive validation from a portion of the test set
            n_test = len(df_test)
            n_val = int(n_test * self.val_from_test_ratio)
            df_val = df_test.iloc[:n_val].reset_index(drop=True)
            df_test_final = df_test.iloc[n_val:].reset_index(drop=True)

            self.train_dataset = Dataset.from_pandas(df_train.reset_index(drop=True))
            self.val_dataset = Dataset.from_pandas(df_val)
            self.test_dataset = Dataset.from_pandas(df_test_final)

            logger.info(
                f"Loaded train/val/test rows: {len(self.train_dataset)}/{len(self.val_dataset)}/{len(self.test_dataset)}"
            )
        else:
            # Single parquet, split into train/val/test by ratios
            source_data_path = f"{self.source_data_folder}/{self.source_data_path}"
            logger.info(f"Reading parquet from S3 in the mounted path: {source_data_path}")
            df = pd.read_parquet(source_data_path)
            available_columns = set(df.columns)
            if not expected_columns.issubset(available_columns):
                missing_cols = expected_columns - available_columns
                logger.error(f"Missing required columns: {missing_cols}")
                logger.info(f"Available columns: {available_columns}")
                raise ValueError(f"Parquet file must contain columns: {expected_columns}")

            initial_count = len(df)
            df = df.dropna(subset=["question", "cypher"]) \
                   .reset_index(drop=True)
            cleaned_count = len(df)
            if initial_count != cleaned_count:
                logger.warning(
                    f"Removed {initial_count - cleaned_count} rows with None values in question or cypher"
                )
            df = self._filter_low_quality_data(df)
            self.dataset = Dataset.from_pandas(df)
            logger.info(f"Loaded {len(self.dataset)} rows from S3 after quality filtering.")

    def setup(self) -> None:
        logger.info(f"Setting up dataset...")

        self.load_data()

        # If separate train/test files were provided, load_data already created splits
        if hasattr(self, 'train_dataset') and hasattr(self, 'test_dataset'):
            # Optionally downselect counts for tests
            if self.train_samples > 0:
                self.train_dataset = self.train_dataset.select(
                    range(min(self.train_samples, len(self.train_dataset)))
                )
            if self.val_samples > 0 and hasattr(self, 'val_dataset') and len(self.val_dataset) > 0:
                self.val_dataset = self.val_dataset.select(
                    range(min(self.val_samples, len(self.val_dataset)))
                )
            if self.test_samples > 0:
                self.test_dataset = self.test_dataset.select(
                    range(min(self.test_samples, len(self.test_dataset)))
                )
        else:
            # Single-parquet flow: split here
            full_dataset = self.dataset.shuffle(seed=self.shuffle_seed)
            logger.debug(f"Dataset columns: {full_dataset.column_names}")

            n = len(full_dataset)
            assert abs(self.train_split + self.val_split + self.test_split - 1.0) < 1e-6, \
                "Train/val/test splits must sum to 1.0"

            train_end = int(n * self.train_split)
            val_end = train_end + int(n * self.val_split)

            self.train_dataset = full_dataset.select(range(0, train_end))
            self.val_dataset = full_dataset.select(range(train_end, val_end))
            self.test_dataset = full_dataset.select(range(val_end, n))

        # Note: downselection handled above for both flows

        self._log_dataset_statistics(self.train_dataset, "train")
        self._log_dataset_statistics(self.val_dataset, "validation")
        self._log_dataset_statistics(self.test_dataset, "test")

        logger.info(
            f"Train size: {len(self.train_dataset)}, "
            f"Validation size: {len(self.val_dataset)}, "
            f"Test size: {len(self.test_dataset)}"
        )

    def _log_dataset_statistics(self, dataset, split_name: str) -> None:
        if len(dataset) == 0:
            logger.info(f"{split_name.capitalize()} set is empty; skipping statistics.")
            return

        questions = dataset["question"]
        cypher_queries = dataset["cypher"]
        
        question_lengths = [len(x.split()) for x in questions]
        cypher_lengths = [len(x.split()) for x in cypher_queries]
        
        # Character-based statistics
        question_char_lengths = [len(x) for x in questions]
        cypher_char_lengths = [len(x) for x in cypher_queries]

        logger.info(f"{split_name.capitalize()} set statistics:")
        logger.info(
            f"Question lengths (words) - Min: {min(question_lengths)}, "
            f"Max: {max(question_lengths)}, "
            f"Avg: {sum(question_lengths)/len(question_lengths):.2f}"
        )
        logger.info(
            f"Cypher lengths (words) - Min: {min(cypher_lengths)}, "
            f"Max: {max(cypher_lengths)}, "
            f"Avg: {sum(cypher_lengths)/len(cypher_lengths):.2f}"
        )
        logger.info(
            f"Question lengths (chars) - Min: {min(question_char_lengths)}, "
            f"Max: {max(question_char_lengths)}, "
            f"Avg: {sum(question_char_lengths)/len(question_char_lengths):.1f}"
        )
        logger.info(
            f"Cypher lengths (chars) - Min: {min(cypher_char_lengths)}, "
            f"Max: {max(cypher_char_lengths)}, "
            f"Avg: {sum(cypher_char_lengths)/len(cypher_char_lengths):.1f}"
        )

    def export_cleaned_data(self) -> None:
        # Determine a stable base name for output files
        if self.source_train_data_path and self.source_test_data_path:
            stem = os.path.splitext(os.path.basename(self.source_train_data_path))[0]
            stem = re.sub(r"_(train|test)$", "", stem)
        else:
            stem = os.path.splitext(os.path.basename(self.source_data_path))[0]

        train_filename = f"{stem}_train.parquet"
        val_filename = f"{stem}_val.parquet"
        test_filename = f"{stem}_test.parquet"

        output_folder = f"{self.preprocessed_output_data_folder}/preprocessed"
        os.makedirs(output_folder, exist_ok=True)

        train_data_dest_path = f"{output_folder}/{train_filename}"
        val_data_dest_path = f"{output_folder}/{val_filename}"
        test_data_dest_path = f"{output_folder}/{test_filename}"

        logger.info(f"Saving preprocessed datasets:")
        logger.info(f"  Train: {train_data_dest_path}")
        logger.info(f"  Val: {val_data_dest_path}")
        logger.info(f"  Test: {test_data_dest_path}")

        # When using single parquet flow, these datasets were created in setup()
        if hasattr(self, "train_dataset") and hasattr(self, "val_dataset") and hasattr(self, "test_dataset"):
            # Ensure non-empty splits for downstream loaders
            train_ds = self.train_dataset
            val_ds = self.val_dataset
            test_ds = self.test_dataset

            if len(val_ds) == 0:
                if len(test_ds) > 0:
                    val_ds = test_ds.select([0])
                elif len(train_ds) > 0:
                    val_ds = train_ds.select([0])
            if len(train_ds) == 0 and len(test_ds) > 0:
                train_ds = test_ds.select([0])
            if len(test_ds) == 0 and len(train_ds) > 0:
                test_ds = train_ds.select([0])

            train_ds.to_parquet(train_data_dest_path)
            val_ds.to_parquet(val_data_dest_path)
            test_ds.to_parquet(test_data_dest_path)
        else:
            # Single parquet path: split self.dataset
            full_dataset = self.dataset
            n = len(full_dataset)
            train_end = int(n * self.train_split)
            val_end = train_end + int(n * self.val_split)
            full_dataset.select(range(0, train_end)).to_parquet(train_data_dest_path)
            full_dataset.select(range(train_end, val_end)).to_parquet(val_data_dest_path)
            full_dataset.select(range(val_end, n)).to_parquet(test_data_dest_path)

        logger.info("Data export completed successfully")

    def clean_output_folder(self):
        output_folder = f"{self.preprocessed_output_data_folder}/preprocessed"
        if os.path.exists(output_folder):
            for filename in os.listdir(output_folder):
                file_path = os.path.join(output_folder, filename)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                        logger.debug(f"Deleted: {file_path}")
                except Exception as e:
                    logger.warning(f"Failed to delete {file_path}. Reason: {e}")
        else:
            os.makedirs(output_folder, exist_ok=True)
            logger.info(f"Created output folder: {output_folder}")

    def run(self) -> None:
        """Main preprocessing pipeline."""
        logger.info("Starting Text2Cypher preprocessing pipeline")
        
        self.clean_output_folder()
        self.setup()
        self.export_cleaned_data()
        
        logger.info("Text2Cypher preprocessing pipeline completed successfully")

    @staticmethod
    def format_text_for_cypher(question: str, schema: str = None) -> str:
        """
        Format question and schema into a structured prompt for Cypher generation.
        This is used by the API for consistent formatting.
        """
        if schema:
            return (
                "You are a Cypher Query Expert. Convert the following natural language question into a Cypher query using the provided graph schema.\n\n"
                f"### Graph Schema ###\n{schema}\n\n"
                f"### Question ###\n{question}\n\n"
                f"### Cypher Query ###\n"
            )
        else:
            return (
                "You are a Cypher Query Expert. Convert the following natural language question into a Cypher query.\n\n"
                f"### Question ###\n{question}\n\n"
                f"### Cypher Query ###\n"
            )

