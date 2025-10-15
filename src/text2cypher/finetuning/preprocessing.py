from loguru import logger
from omegaconf import DictConfig
import os


from text2cypher.finetuning.data.text2cypher_preprocessing import Text2CypherDataPreprocessingModule
from text2cypher.finetuning.utils.logger import setup_logger


def preprocessing(cfg: DictConfig):
    setup_logger(cfg.logging.log_path)
    logger.info(f"Starting preprocessing pipeline")

    env_folder = os.getenv("ENV", "no-env")
    preprocessingmodule = Text2CypherDataPreprocessingModule(
        model_name=cfg.model.name,
        source_data_folder=cfg.data.source_data_folder,
        preprocessed_output_data_folder=cfg.data.preprocessed_output_data_folder,
        env_folder=env_folder,
        source_data_path=cfg.data.source_data_path,
        source_train_data_path=getattr(cfg.data, 'source_train_data_path', None),
        source_test_data_path=getattr(cfg.data, 'source_test_data_path', None),
        max_length=cfg.model.max_length,
        train_samples=cfg.data.train_samples,
        val_samples=cfg.data.val_samples,
        test_samples=cfg.data.test_samples,
        val_from_test_ratio=getattr(cfg.data, 'val_from_test_ratio', 0.2),
        shuffle=cfg.data.shuffle,
        shuffle_seed=cfg.data.shuffle_seed,
        # Data quality filtering parameters
        apply_quality_filters=cfg.data.apply_quality_filters,
        min_question_length=cfg.data.min_text_length,
        min_cypher_length=cfg.data.min_cypher_length,
        max_length_ratio=cfg.data.max_length_ratio,
        min_length_ratio=cfg.data.min_length_ratio,
    )

    preprocessingmodule.run()
    logger.info(f"Data preprocessing finished")


if __name__ == "__main__":
    preprocessing()
