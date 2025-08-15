import importlib
import os
from urllib.parse import urlparse
import boto3
from loguru import logger

MODEL_CLASSES = {
    "t5": "text2cypher.finetuning.models.t5_model.T5NoteGenerationModel",
    # Add more model types here as needed
}


def load_class_from_path(class_path):
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def load_model(model_ckpt, model_name, model_type, device, peft_method=None):
    if model_type not in MODEL_CLASSES:
        raise ValueError(f"Unsupported model type: {model_type}")

    logger.info(f"Loading model `{model_name}` from `{model_ckpt}` using type `{model_type}`")

    model_class = load_class_from_path(MODEL_CLASSES[model_type])

    model = model_class.load_model_from_checkpoint(
        checkpoint_path=model_ckpt,
        model_name=model_name,
        model_type=model_type,
        peft_method=peft_method
    )

    model.to(device)
    return model


def resolve_checkpoint_path(uri: str, local_dir: str = "/tmp/model") -> str:
    """If `uri` is s3://, download to local_dir and return local path; otherwise return as-is."""
    if not uri.startswith("s3://"):
        return uri

    os.makedirs(local_dir, exist_ok=True)

    p = urlparse(uri)
    bucket = p.netloc
    key = p.path.lstrip("/")
    filename = os.path.basename(key)
    local_path = os.path.join(local_dir, filename)

    s3 = boto3.client("s3")  # region will come from ECS metadata/env
    s3.download_file(bucket, key, local_path)
    return local_path

