# scripts/run_pipeline.py

import hydra
from omegaconf import DictConfig
from loguru import logger

from text2cypher.finetuning.train import train
from text2cypher.finetuning.evaluate_model import evaluate_model


@hydra.main(config_path="../src/text2cypher/finetuning/config", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    logger.info(f"🚀 Starting pipeline with config: {cfg}")

    # Step 1: Train
    logger.info("📦 Training model...")
    train(cfg)

    # Step 2: Evaluate
    logger.info("📊 Running evaluation...")
    evaluate_model(cfg)

    logger.success("✅ Pipeline completed!")


if __name__ == "__main__":
    main()
