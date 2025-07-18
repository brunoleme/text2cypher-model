import os
from datetime import datetime

import hydra
from loguru import logger
from omegaconf import DictConfig

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import wandb

from text2cypher.finetuning.data.notechat_dataset import NoteChatDataModule
from text2cypher.finetuning.utils.logger import setup_logger

MODEL_CLASSES = {
    "t5": "text2cypher.finetuning.models.t5_model.T5NoteGenerationModel",
}

def train(cfg: DictConfig):
    setup_logger(cfg.logging.log_path)
    logger.info(f"Starting training pipeline")

    safe_model_name = cfg.model.name.replace("/", "_")

    datamodule = NoteChatDataModule(
        model_name=cfg.model.name,
        source_data_path=cfg.data.input_data_uri,
        batch_size=cfg.training.batch_size,
        max_length=cfg.model.max_length,
        num_workers=cfg.training.num_workers,
        train_samples=cfg.data.train_samples,
        val_samples=cfg.data.val_samples,
        test_samples=cfg.data.test_samples,
        shuffle=cfg.data.shuffle,
        shuffle_seed=cfg.data.shuffle_seed,
    )

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

    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.training.checkpoint_dir,
        filename=f"best-{safe_model_name}-peft{cfg.model.peft_method}-{now}-epoch{{epoch:02d}}-val_loss{{val_loss:.2f}}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_weights_only=True,
        auto_insert_metric_name=False,
    )

    callbacks = [
        checkpoint_callback,
        EarlyStopping(monitor="val_loss", patience=cfg.training.patience, mode="min"),
    ]

    wandb.init(project=cfg.project_name, name=f"{cfg.model.name}-training")
    wandb_logger = WandbLogger(project=cfg.project_name, log_model=True, save_dir=cfg.training.checkpoint_dir)
    wandb_logger.experiment.config.update(dict(cfg))

    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator="auto",
        devices=cfg.training.devices,
        callbacks=callbacks,
        logger=wandb_logger,
        gradient_clip_val=cfg.training.gradient_clip_val,
    )

    logger.info("Starting model training")
    trainer.fit(model, datamodule=datamodule)
    logger.success("Training completed successfully")
    logger.info(f"Best model path: {checkpoint_callback.best_model_path}")
    wandb_logger.experiment.config.update({"best_model_path": checkpoint_callback.best_model_path})
    wandb.finish()


if __name__ == "__main__":
    train()
