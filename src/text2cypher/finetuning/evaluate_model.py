import os
import gc
import json
import pandas as pd
import torch
from omegaconf import DictConfig

from loguru import logger
from typing import Dict
import wandb

from text2cypher.finetuning.data.text2cypher_dataset import Text2CypherDataModule
from text2cypher.finetuning.utils.load_models import load_model
from text2cypher.finetuning.eval.metrics import (
    calculate_rouge, calculate_bleu, calculate_bertscore,
    calculate_exact_match, calculate_cypher_lint_rate,
    compute_group_metrics_from_rows, calculate_average_latency, calculate_model_size_in_params
)

def load_rows(cfg, samples, env_folder):
    data_module = Text2CypherDataModule(
        model_name=cfg.model.name,
        preprocessed_input_data_folder=cfg.data.preprocessed_input_data_folder,
        source_data_path=cfg.data.source_data_path,
        env_folder=env_folder,
        batch_size=cfg.training.batch_size,
        max_length=cfg.model.max_length,
        num_workers=cfg.training.num_workers,
        train_samples=1,
        val_samples=1,
        test_samples=samples,
        shuffle=cfg.data.shuffle,
        shuffle_seed=cfg.data.shuffle_seed,
    )
    data_module.setup()
    # Reconstruct rows from dataset columns
    ds = data_module.test_dataset
    rows = [{
        "question": ds[i]["question"],
        "schema": ds[i]["schema"],
        "cypher": ds[i]["cypher"],
    } for i in range(len(ds))]
    return rows

def run_metric_evaluation(title, cfg, sample_count, metrics_dict, model, env_folder):
    logger.info(f"Starting {title} evaluation")
    rows = load_rows(cfg, sample_count, env_folder)
    results_df = compute_group_metrics_from_rows(model, rows, cfg.model.max_length, metrics_dict)
    gc.collect()
    return results_df

def evaluate_model(cfg: DictConfig):
    env_folder = os.getenv("ENV", "no-env")
    pipeline_run_id = os.getenv("PIPELINE_RUN_ID", "no-pipeline-id")

    with wandb.init(project=f"{cfg.project_name}-evaluation-{env_folder}", name=f"{cfg.model.name}-{cfg.model.peft_method}", tags=[f"pipeline:{pipeline_run_id}"]) as run:
        logger.info("Starting computing evaluation metrics")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info("Defining evaluation metrics to be computed")
        lexical_metrics_dict = {
            "exact_match": calculate_exact_match,
            "bleu_score": calculate_bleu,
        }
        semantical_metrics_dict = {
            "bert_score": calculate_bertscore,
            "cypher_lint_rate": calculate_cypher_lint_rate,
        }

        logger.info("Loading model from checkpoint")

        model_ckpt = f"{cfg.evaluation.model_artifact_dir}/{pipeline_run_id}/checkpoints/best_model.ckpt"

        model_name = cfg.model.name
        model_type = cfg.model.type
        peft_method = cfg.model.peft_method

        model = load_model(model_ckpt, model_name, model_type, device, peft_method)

        results = []

        model_name_df = pd.DataFrame({"pipeline_run_id": [pipeline_run_id]})
        results.append(model_name_df)

        # Lexical metrics #def run_metric_evaluation(title, cfg, dataloader_samples, metrics_dict, model, device):
        logger.info("Computing lexical metrics")
        lexical_metrics_results_df = run_metric_evaluation("lexical metrics", cfg, cfg.evaluation.test_samples_lexical_metrics, lexical_metrics_dict, model, env_folder)
        results.append(lexical_metrics_results_df)

        logger.info("Computing semantic metrics")
        semantical_metrics_results_df = run_metric_evaluation("semantical metrics", cfg, cfg.evaluation.test_samples_semantic_metrics, semantical_metrics_dict, model, env_folder)
        results.append(semantical_metrics_results_df)

        # Skipping AI-as-a-judge for now in text2cypher baseline (no LLM dependency)

        logger.info("Computing system metrics")
        rows = load_rows(cfg, cfg.evaluation.test_samples_semantic_metrics, env_folder)
        latency = calculate_average_latency(model, rows, cfg.model.max_length)
        size_params = calculate_model_size_in_params(model)
        system_metrics_df = pd.DataFrame({
            "model_size_params": [size_params],
            "avg_latency_sec": [latency],
        })
        results.append(system_metrics_df)

        del model
        gc.collect()

        model_metrics_df = pd.concat(results, axis=1)
        wandb.log({"evaluation_results": wandb.Table(dataframe=model_metrics_df)})
        reports_folder = f'{cfg.training.model_artifact_dir}/{pipeline_run_id}/reports'
        os.makedirs(reports_folder, exist_ok=True)
        logger.info(f"Contents of {f'{cfg.training.model_artifact_dir}/{pipeline_run_id}'}: {os.listdir(f'{cfg.training.model_artifact_dir}/{pipeline_run_id}')}")  
        model_metrics_df.to_json(f'{reports_folder}/eval_metrics.json', lines=True, orient='records')
        run.finish()
        logger.info("Finished computing evaluation metrics")

if __name__ == "__main__":
    evaluate_model()
