"""
MindFu Training Service - QLoRA Fine-tuning with HuggingFace
Uses transformers + peft + trl (no unsloth dependency)
"""
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import mlflow
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class TrainingConfig:
    """Training configuration."""

    def __init__(self):
        # Devstral-Small-2-24B uses 'ministral3' architecture - requires transformers from git
        self.base_model = os.getenv("BASE_MODEL", "mistralai/Devstral-Small-2-24B-Instruct-2512")
        self.output_dir = os.getenv("OUTPUT_DIR", "/models/fine-tuned")
        self.conversations_dir = os.getenv("CONVERSATIONS_DIR", "/conversations")

        # LoRA config
        self.lora_rank = int(os.getenv("LORA_RANK", "16"))
        self.lora_alpha = int(os.getenv("LORA_ALPHA", "32"))
        self.lora_dropout = float(os.getenv("LORA_DROPOUT", "0.05"))

        # Training config
        self.batch_size = int(os.getenv("BATCH_SIZE", "2"))
        self.gradient_accumulation_steps = int(os.getenv("GRADIENT_ACCUMULATION_STEPS", "8"))
        self.learning_rate = float(os.getenv("LEARNING_RATE", "2e-4"))
        self.num_epochs = int(os.getenv("NUM_EPOCHS", "3"))
        self.max_seq_length = int(os.getenv("MAX_SEQ_LENGTH", "2048"))
        self.warmup_ratio = float(os.getenv("WARMUP_RATIO", "0.03"))

        # 4-bit quantization
        self.load_in_4bit = os.getenv("LOAD_IN_4BIT", "true").lower() == "true"

        # MLflow
        self.mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        self.experiment_name = os.getenv("EXPERIMENT_NAME", "mindfu-training")


def load_conversations(conversations_dir: str) -> list:
    """Load conversation logs for training."""
    conversations = []
    conv_path = Path(conversations_dir)

    if not conv_path.exists():
        logger.warning(f"Conversations directory not found: {conversations_dir}")
        return conversations

    for file in conv_path.glob("*.json"):
        try:
            with open(file, "r") as f:
                data = json.load(f)
                if isinstance(data, list):
                    conversations.extend(data)
                else:
                    conversations.append(data)
        except Exception as e:
            logger.warning(f"Failed to load {file}: {e}")

    logger.info(f"Loaded {len(conversations)} conversations")
    return conversations


def format_for_instruct(item: dict) -> str:
    """Format item for instruction tuning (Alpaca-style or chat format)."""
    # Handle Alpaca-style format
    if "instruction" in item:
        instruction = item.get("instruction", "")
        input_text = item.get("input", "")
        output = item.get("output", "")

        if input_text:
            return f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
        else:
            return f"### Instruction:\n{instruction}\n\n### Response:\n{output}"

    # Handle chat format
    messages = item.get("messages", [])
    response = item.get("response", {})

    formatted = ""
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if role == "system":
            formatted += f"### System:\n{content}\n\n"
        elif role == "user":
            formatted += f"### Instruction:\n{content}\n\n"
        elif role == "assistant":
            formatted += f"### Response:\n{content}\n\n"

    if response:
        formatted += f"### Response:\n{response.get('content', '')}"

    return formatted


def prepare_dataset(conversations: list) -> Dataset:
    """Prepare dataset from conversations."""
    formatted = []

    for conv in conversations:
        try:
            text = format_for_instruct(conv)
            if text.strip():
                formatted.append({"text": text})
        except Exception as e:
            logger.warning(f"Failed to format conversation: {e}")

    if not formatted:
        # Add a placeholder if no conversations
        formatted.append({
            "text": "### Instruction:\nHello\n\n### Response:\nHello! How can I help you today?"
        })

    return Dataset.from_list(formatted)


def train(config: TrainingConfig, dataset: Optional[Dataset] = None):
    """Run QLoRA fine-tuning with HuggingFace stack."""
    logger.info("Starting training...")
    logger.info(f"Base model: {config.base_model}")
    logger.info(f"Output directory: {config.output_dir}")

    # Setup MLflow
    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    mlflow.set_experiment(config.experiment_name)

    with mlflow.start_run():
        # Log parameters
        mlflow.log_params({
            "base_model": config.base_model,
            "lora_rank": config.lora_rank,
            "lora_alpha": config.lora_alpha,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "num_epochs": config.num_epochs,
        })

        # Configure 4-bit quantization
        bnb_config = None
        if config.load_in_4bit:
            logger.info("Configuring 4-bit quantization...")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                bnb_4bit_use_double_quant=True,
            )

        # Load tokenizer
        logger.info("Loading tokenizer...")
        tokenizer = None

        # Try direct loading first
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                config.base_model,
                trust_remote_code=True,
            )
        except (ValueError, Exception) as e:
            logger.warning(f"Default tokenizer loading failed: {e}")

        # Fall back to compatible Mistral tokenizer
        if tokenizer is None:
            logger.info("Trying Mistral base tokenizer...")
            try:
                # Use Mistral-7B tokenizer which is compatible with Devstral
                tokenizer = AutoTokenizer.from_pretrained(
                    "mistralai/Mistral-7B-Instruct-v0.3",
                    trust_remote_code=True,
                )
            except Exception as e2:
                logger.warning(f"Mistral tokenizer failed: {e2}")

        # Last resort: load tokenizer.json directly
        if tokenizer is None:
            logger.info("Trying direct tokenizer.json loading...")
            from tokenizers import Tokenizer
            from transformers import PreTrainedTokenizerFast
            from huggingface_hub import hf_hub_download

            tokenizer_path = hf_hub_download(
                repo_id=config.base_model,
                filename="tokenizer.json",
            )
            base_tokenizer = Tokenizer.from_file(tokenizer_path)
            tokenizer = PreTrainedTokenizerFast(tokenizer_object=base_tokenizer)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        # Load model
        logger.info("Loading model...")

        # First, check model config to determine the right class
        from transformers import AutoConfig
        model_config = AutoConfig.from_pretrained(config.base_model, trust_remote_code=True)
        model_type = getattr(model_config, 'model_type', None)
        logger.info(f"Model type: {model_type}")

        # Try loading with specific model class for Mistral3/Ministral3/Devstral
        model = None
        if model_type in ['mistral3', 'ministral3', 'devstral']:
            # Try Ministral3ForCausalLM (note the 'i' - transformers naming)
            try:
                from transformers import Ministral3ForCausalLM
                logger.info("Loading with Ministral3ForCausalLM...")
                model = Ministral3ForCausalLM.from_pretrained(
                    config.base_model,
                    quantization_config=bnb_config,
                    device_map="auto",
                    trust_remote_code=True,
                    torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                )
            except ImportError:
                logger.warning("Ministral3ForCausalLM not available")
            except Exception as e:
                logger.warning(f"Ministral3ForCausalLM failed: {e}")

        # Fall back to AutoModelForCausalLM
        if model is None:
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    config.base_model,
                    quantization_config=bnb_config,
                    device_map="auto",
                    trust_remote_code=True,
                    torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                )
            except ValueError as e:
                logger.error(f"AutoModelForCausalLM failed: {e}")
                logger.info("Model may require a newer transformers version or different architecture")
                raise

        # Prepare model for k-bit training
        if config.load_in_4bit:
            model = prepare_model_for_kbit_training(model)

        # Configure LoRA
        logger.info("Configuring LoRA...")
        lora_config = LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            bias="none",
            task_type="CAUSAL_LM",
        )

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        # Prepare dataset
        if dataset is None:
            conversations = load_conversations(config.conversations_dir)
            dataset = prepare_dataset(conversations)

        logger.info(f"Dataset size: {len(dataset)}")
        mlflow.log_metric("dataset_size", len(dataset))

        # Training arguments
        training_args = TrainingArguments(
            output_dir=config.output_dir,
            per_device_train_batch_size=config.batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            learning_rate=config.learning_rate,
            num_train_epochs=config.num_epochs,
            warmup_ratio=config.warmup_ratio,
            logging_steps=10,
            save_strategy="epoch",
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            optim="paged_adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=42,
            report_to="mlflow",
            gradient_checkpointing=True,
            max_grad_norm=0.3,
        )

        # Create trainer
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            dataset_text_field="text",
            max_seq_length=config.max_seq_length,
            args=training_args,
            packing=False,
        )

        # Train
        logger.info("Starting training...")
        train_result = trainer.train()

        # Log metrics
        mlflow.log_metrics({
            "train_loss": train_result.training_loss,
            "train_runtime": train_result.metrics.get("train_runtime", 0),
            "train_samples_per_second": train_result.metrics.get("train_samples_per_second", 0),
        })

        # Save model
        logger.info("Saving model...")
        output_path = Path(config.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path.mkdir(parents=True, exist_ok=True)

        # Save LoRA adapter
        trainer.save_model(str(output_path / "lora"))
        tokenizer.save_pretrained(str(output_path / "lora"))

        # Merge and save full model
        logger.info("Merging LoRA weights...")
        merged_model = trainer.model.merge_and_unload()
        merged_model.save_pretrained(str(output_path / "merged"))
        tokenizer.save_pretrained(str(output_path / "merged"))

        mlflow.log_artifact(str(output_path))
        logger.info(f"Model saved to {output_path}")

        return str(output_path)


def main():
    """Main entry point."""
    config = TrainingConfig()

    # Check if running as API or direct training
    if os.getenv("RUN_API", "false").lower() == "true":
        run_api(config)
    else:
        train(config)


def run_api(config: TrainingConfig):
    """Run training as an API service."""
    from fastapi import BackgroundTasks, FastAPI, HTTPException
    from pydantic import BaseModel
    import uvicorn

    app = FastAPI(title="MindFu Training Service")

    class TrainRequest(BaseModel):
        experiment_name: str | None = None
        lora_rank: int | None = None
        num_epochs: int | None = None
        learning_rate: float | None = None

    training_status = {"running": False, "last_run": None, "error": None}

    @app.get("/health")
    def health():
        return {"status": "healthy"}

    @app.get("/status")
    def status():
        return training_status

    @app.post("/train/start")
    def start_training(request: TrainRequest, background_tasks: BackgroundTasks):
        if training_status["running"]:
            raise HTTPException(status_code=409, detail="Training already in progress")

        # Update config from request
        if request.experiment_name:
            config.experiment_name = request.experiment_name
        if request.lora_rank:
            config.lora_rank = request.lora_rank
        if request.num_epochs:
            config.num_epochs = request.num_epochs
        if request.learning_rate:
            config.learning_rate = request.learning_rate

        def run_training():
            training_status["running"] = True
            training_status["error"] = None
            try:
                result = train(config)
                training_status["last_run"] = {
                    "output_path": result,
                    "timestamp": datetime.now().isoformat(),
                }
            except Exception as e:
                training_status["error"] = str(e)
                logger.exception("Training failed")
            finally:
                training_status["running"] = False

        background_tasks.add_task(run_training)
        return {"message": "Training started"}

    uvicorn.run(app, host="0.0.0.0", port=5001)


if __name__ == "__main__":
    main()
