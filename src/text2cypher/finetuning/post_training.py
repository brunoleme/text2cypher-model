# post_training.py
import os
import tarfile
import shutil
from loguru import logger
from omegaconf import DictConfig
from text2cypher.api import inference  # Keeps import for packaging logic validation

def package_model(cfg: DictConfig):
    pipeline_run_id = os.getenv("PIPELINE_RUN_ID", "no-pipeline-id")
    model_dir = os.path.join(cfg.training.model_artifact_dir, f"{pipeline_run_id}/hf_model")
    tar_path = os.path.join(cfg.training.model_artifact_dir, f"{pipeline_run_id}/model.tar.gz")

    # Path to your inference.py source
    inference_src_path = os.path.abspath("src/text2cypher/api/inference.py")
    inference_dst_path = os.path.join(model_dir, "inference.py")

    # Copy inference.py into the hf_model folder so it gets tarred
    logger.info(f"Copying inference.py to model directory: {inference_dst_path}")
    shutil.copy(inference_src_path, inference_dst_path)

    # Now package everything inside model_dir
    logger.info(f"Packing model directory: {model_dir} into {tar_path}")
    with tarfile.open(tar_path, "w:gz") as tar:
        for filename in os.listdir(model_dir):
            file_path = os.path.join(model_dir, filename)
            tar.add(file_path, arcname=filename)

    logger.success(f"Model packaged at: {tar_path}")
