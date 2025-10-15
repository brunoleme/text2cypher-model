import datetime
import os
import platform
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from loguru import logger
from peft import LoraConfig, PromptTuningConfig
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import BitsAndBytesConfig, get_linear_schedule_with_warmup

class BaseText2CypherModel(nn.Module, ABC):
    def __init__(
        self,
        model_name: str,
        model_type: str,
        learning_rate: float = 2e-5,
        warmup_steps: int = 500,
        weight_decay: float = 0.01,
        use_quantization: bool = True,
        quantization_type: str = "8bit",
        peft_method: Optional[str] = None,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        prompt_tuning_n_tokens: int = 20,
        **kwargs,
    ):
        super().__init__()
        self.model_name = model_name
        self.model_type = model_type
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.use_quantization = use_quantization
        self.quantization_type = quantization_type
        self.peft_method = peft_method
        
        
        # Memory management
        self._memory_optimization_enabled = True
        
        # Device management
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Initializing model: {model_name} ({model_type}), PEFT Method: {peft_method}, Quantization: {use_quantization}")

        # Store experiment metadata
        self.experiment_metadata = {
            "model_type": model_type,
            "timestamp": datetime.datetime.now().isoformat(),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "hyperparameters": {
                "model_name": model_name,
                "model_type": model_type,
                "learning_rate": learning_rate,
                "warmup_steps": warmup_steps,
                "weight_decay": weight_decay,
                "use_quantization": use_quantization,
                "peft_method": peft_method,
            },
        }

        # Configure quantization based on type
        if use_quantization:
            if quantization_type == "8bit":
                self.quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                    bnb_8bit_compute_dtype=torch.float16,
                    bnb_8bit_use_double_quant=True,
                )
            elif quantization_type == "4bit":
                self.quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
            else:
                logger.warning(
                    f"Unknown quantization type: {quantization_type}, disabling quantization"
                )
                self.quantization_config = None
                self.use_quantization = False
        else:
            self.quantization_config = None

        # PEFT Config
        self.peft_config = None
        if self.peft_method == "lora":
            # Get model-specific target modules
            target_modules = self._get_lora_target_modules()
            
            self.peft_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=target_modules,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM" if self._is_decoder_only() else "SEQ_2_SEQ_LM",
            )
        
        elif self.peft_method == "prompt_tuning":
            from peft import TaskType
            self.peft_config = PromptTuningConfig(
                task_type=TaskType.CAUSAL_LM if self._is_decoder_only() else TaskType.SEQ_2_SEQ_LM,
                prompt_tuning_init="TEXT",
                num_virtual_tokens=prompt_tuning_n_tokens,
                prompt_tuning_init_text="Convert the following text to a Cypher query:",
                tokenizer_name_or_path=model_name,
            )

        # Initialize the model
        self.model, self.tokenizer = self._initialize_model(
            model_name, model_type, use_quantization, peft_method, **kwargs
        )

    def _is_decoder_only(self) -> bool:
        """Check if this is a decoder-only model (GPT-style)."""
        return (
            "gpt" in self.model_type.lower()
            or "llama" in self.model_type.lower()
            or "mistral" in self.model_type.lower()
            or "phi" in self.model_type.lower()
            or "decoder" in self.model_type.lower()
        )

    def _get_lora_target_modules(self):
        """Get the appropriate target modules for LoRA based on model type."""
        if "llama" in self.model_type.lower():
            return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        elif "mistral" in self.model_type.lower():
            return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        elif "phi" in self.model_type.lower():
            return ["q_proj", "k_proj", "v_proj", "dense"]
        elif "t5" in self.model_type.lower():
            return ["q", "v", "k", "o", "wi", "wo"]
        else:
            logger.warning(f"Unknown model type {self.model_type}, using default LoRA targets")
            return ["q_proj", "v_proj"]

    @abstractmethod
    def _initialize_model(self, model_name: str, model_type: str, use_quantization: bool, peft_method: Optional[str] = None, **kwargs):
        """Initialize the model and tokenizer. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def forward(self, **inputs) -> Any:
        """Forward pass through the model. Must be implemented by subclasses."""
        pass

    def setup_training(self):
        """Setup model for training mode."""
        self.model.train()
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()

    def setup_inference(self):
        """Setup model for inference mode."""
        self.model.eval()
        if hasattr(self.model, 'gradient_checkpointing_disable'):
            self.model.gradient_checkpointing_disable()

    def get_optimizer_and_scheduler(self, num_training_steps: int, lr_scheduler_config: Dict = None):
        """Get optimizer and learning rate scheduler."""
        # Get all parameters that require gradients
        params = [p for p in self.model.parameters() if p.requires_grad]
        
        optimizer = AdamW(
            params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            eps=1e-8,
        )

        # Setup scheduler
        if lr_scheduler_config is None:
            lr_scheduler_config = {"name": "linear_warmup"}

        scheduler_name = lr_scheduler_config.get("name", "linear_warmup")
        
        if scheduler_name == "linear_warmup":
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.warmup_steps,
                num_training_steps=num_training_steps,
            )
        else:
            logger.warning(f"Unknown scheduler: {scheduler_name}, using linear warmup")
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.warmup_steps,
                num_training_steps=num_training_steps,
            )

        return optimizer, scheduler

    def compute_loss(self, batch):
        """Compute training loss for a batch."""
        # Move batch to device
        batch = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in batch.items()}
        
        # Forward pass
        outputs = self.forward(**batch)
        
        # Extract loss
        if hasattr(outputs, 'loss'):
            return outputs.loss
        else:
            # Fallback for models that don't return loss directly
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            labels = batch.get('labels')
            if labels is not None:
                loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
                return loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            else:
                logger.warning("No labels provided and model doesn't return loss")
                return torch.tensor(0.0, requires_grad=True, device=self.device)

    def compute_validation_metrics(self, batch, batch_idx):
        """Compute validation metrics for a batch."""
        # Basic implementation - subclasses can override for specific metrics
        loss = self.compute_loss(batch)
        return {"val_loss": loss.item()}

    def clear_memory_if_needed(self, batch_idx: int, is_validation: bool = False):
        """Clear GPU memory periodically to prevent OOM."""
        if self._memory_optimization_enabled and batch_idx % 10 == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def to(self, device):
        """Move model to device."""
        self.device = device
        if hasattr(self, 'model'):
            self.model.to(device)
        return super().to(device)

    def parameters(self):
        """Get model parameters."""
        if hasattr(self, 'model'):
            return self.model.parameters()
        return super().parameters()

    def named_parameters(self):
        """Get named model parameters."""
        if hasattr(self, 'model'):
            return self.model.named_parameters()
        return super().named_parameters()

    def state_dict(self):
        """Get model state dict."""
        if hasattr(self, 'model'):
            return self.model.state_dict()
        return super().state_dict()

    def load_state_dict(self, state_dict, strict=True):
        """Load model state dict."""
        if hasattr(self, 'model'):
            return self.model.load_state_dict(state_dict, strict=strict)
        return super().load_state_dict(state_dict, strict=strict)

    def train(self, mode=True):
        """Set training mode."""
        if hasattr(self, 'model'):
            self.model.train(mode)
        return super().train(mode)

    def eval(self):
        """Set evaluation mode."""
        if hasattr(self, 'model'):
            self.model.eval()
        return super().eval()