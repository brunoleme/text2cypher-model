# post_training.py
import os
import tarfile
from loguru import logger
from omegaconf import DictConfig

def package_model(cfg: DictConfig):
    pipeline_run_id = os.getenv("PIPELINE_RUN_ID", "no-pipeline-id")
    model_dir = os.path.join(cfg.training.model_artifact_dir, f"{pipeline_run_id}/hf_model")
    tar_path = os.path.join(cfg.training.model_artifact_dir, f"{pipeline_run_id}/model.tar.gz")

    logger.info(f"Packing model directory: {model_dir}")
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(model_dir, arcname=".")

    logger.success(f"Model packaged at: {tar_path}")
