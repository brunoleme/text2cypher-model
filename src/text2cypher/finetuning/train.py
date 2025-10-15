import os

import hydra
from loguru import logger
from omegaconf import DictConfig
import wandb

from text2cypher.finetuning.data.text2cypher_dataset import Text2CypherDataModule
from text2cypher.finetuning.models.trainer import Text2CypherModelTrainer
from text2cypher.finetuning.utils.logger import setup_logger

MODEL_CLASSES = {
    "llama": "text2cypher.finetuning.models.llama_model.LlamaText2CypherModel",
}

def train(cfg: DictConfig):
    setup_logger(cfg.logging.log_path)
    logger.info(f"Starting training pipeline (pure PyTorch)")

    env_folder = os.getenv("ENV", "no-env")
    pipeline_run_id = os.getenv("PIPELINE_RUN_ID", "no-pipeline-id")

    # Build datasets and dataloaders
    datamodule = Text2CypherDataModule(
        model_name=cfg.model.name,
        source_data_path=cfg.data.source_data_path,
        preprocessed_input_data_folder=cfg.data.preprocessed_input_data_folder,
        env_folder=env_folder,
        batch_size=cfg.training.batch_size,
        max_length=cfg.model.max_length,
        num_workers=cfg.training.num_workers,
        shuffle=cfg.data.shuffle,
        shuffle_seed=cfg.data.shuffle_seed,
        apply_quality_filters=cfg.data.apply_quality_filters,
        min_text_length=cfg.data.min_text_length,
        min_cypher_length=cfg.data.min_cypher_length,
        max_length_ratio=cfg.data.max_length_ratio,
        min_length_ratio=cfg.data.min_length_ratio,
    )
    datamodule.setup()
    dataloaders = {
        "train": datamodule.train_dataloader(),
        "val": datamodule.val_dataloader(),
        "test": datamodule.test_dataloader(),
    }

    # Instantiate model class
    model_class_path = MODEL_CLASSES[cfg.model.type]
    ModelClass = hydra.utils.get_class(model_class_path)
    model = ModelClass(
        model_name=cfg.model.name,
        model_type=cfg.model.type,
        learning_rate=cfg.training.learning_rate,
        warmup_steps=cfg.training.warmup_steps,
        weight_decay=cfg.training.weight_decay,
        use_quantization=cfg.model.quantization,
        peft_method=cfg.model.peft_method,
    )

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total model parameters: {total_params:,}")

    # Init wandb (disabled in dev/test environments)
    try:
        wandb_mode = os.getenv("WANDB_MODE", "disabled" if env_folder == "dev" else "online")
        run = wandb.init(project=f"{cfg.project_name}-training-{env_folder}", name=f"{cfg.model.name}-{cfg.model.peft_method}", tags=[f"pipeline:{pipeline_run_id}"], mode=wandb_mode)
        if run is not None:
            wandb.config.update(dict(cfg))
    except Exception as e:
        logger.warning(f"W&B init failed or disabled: {e}")

    # Train with pure PyTorch trainer (Accelerate under the hood)
    trainer = Text2CypherModelTrainer(model=model, dataloaders=dataloaders, config=cfg)
    trainer.train()
    logger.success("Training completed successfully")

    # Save model in HF format for inference
    logger.info("Saving model in hf format")
    hf_save_path = os.path.join(cfg.training.model_artifact_dir, f"{pipeline_run_id}/hf_model")
    try:
        if cfg.model.peft_method == "lora" and hasattr(model.model, "merge_and_unload"):
            model.model.merge_and_unload().save_pretrained(hf_save_path)
        else:
            model.model.save_pretrained(hf_save_path)
        model.tokenizer.save_pretrained(hf_save_path)
        logger.success("Model saved successfully")
    except Exception as e:
        logger.error(f"Failed to save HF model: {e}")

    wandb.finish()
