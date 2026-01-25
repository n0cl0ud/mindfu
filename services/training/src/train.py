"""
MindFu Training Service - Unsloth Fine-tuning with QLoRA
"""
# Import unsloth first to apply optimizations
from unsloth import FastLanguageModel

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import mlflow
import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import TrainingArguments
from trl import SFTTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class TrainingConfig:
    """Training configuration."""

    def __init__(self):
        self.base_model = os.getenv("BASE_MODEL", "mistralai/Devstral-Small-2-24B-Instruct-2512")
        self.output_dir = os.getenv("OUTPUT_DIR", "/models/fine-tuned")
        self.conversations_dir = os.getenv("CONVERSATIONS_DIR", "/conversations")

        # LoRA config
        self.lora_rank = int(os.getenv("LORA_RANK", "16"))
        self.lora_alpha = int(os.getenv("LORA_ALPHA", "32"))
        self.lora_dropout = float(os.getenv("LORA_DROPOUT", "0.05"))

        # Training config
        self.batch_size = int(os.getenv("BATCH_SIZE", "4"))
        self.gradient_accumulation_steps = int(os.getenv("GRADIENT_ACCUMULATION_STEPS", "4"))
        self.learning_rate = float(os.getenv("LEARNING_RATE", "2e-4"))
        self.num_epochs = int(os.getenv("NUM_EPOCHS", "3"))
        self.max_seq_length = int(os.getenv("MAX_SEQ_LENGTH", "4096"))
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


def format_conversation(conversation: dict) -> str:
    """Format a conversation for training in chat format."""
    messages = conversation.get("messages", [])
    response = conversation.get("response", {})

    # Build the conversation string
    formatted = ""

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if role == "system":
            formatted += f"<|system|>\n{content}</s>\n"
        elif role == "user":
            formatted += f"<|user|>\n{content}</s>\n"
        elif role == "assistant":
            formatted += f"<|assistant|>\n{content}</s>\n"

    # Add the response
    if response:
        formatted += f"<|assistant|>\n{response.get('content', '')}</s>"

    return formatted


def prepare_dataset(conversations: list) -> Dataset:
    """Prepare dataset from conversations."""
    formatted = []

    for conv in conversations:
        try:
            text = format_conversation(conv)
            if text.strip():
                formatted.append({"text": text})
        except Exception as e:
            logger.warning(f"Failed to format conversation: {e}")

    if not formatted:
        # Add a placeholder if no conversations
        formatted.append({
            "text": "<|user|>\nHello</s>\n<|assistant|>\nHello! How can I help you today?</s>"
        })

    return Dataset.from_list(formatted)


def train(config: TrainingConfig, dataset: Optional[Dataset] = None):
    """Run fine-tuning with Unsloth."""
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

        # Load model with Unsloth
        logger.info("Loading model with Unsloth...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=config.base_model,
            max_seq_length=config.max_seq_length,
            dtype=None,  # Auto-detect
            load_in_4bit=config.load_in_4bit,
        )

        # Configure LoRA
        logger.info("Configuring LoRA...")
        model = FastLanguageModel.get_peft_model(
            model,
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42,
        )

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
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=42,
            report_to="mlflow",
        )

        # Create trainer
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            dataset_text_field="text",
            max_seq_length=config.max_seq_length,
            args=training_args,
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

        # Save in multiple formats
        model.save_pretrained(str(output_path / "lora"))
        tokenizer.save_pretrained(str(output_path / "lora"))

        # Save merged model for vLLM
        logger.info("Merging and saving for vLLM...")
        model.save_pretrained_merged(
            str(output_path / "merged"),
            tokenizer,
            save_method="merged_16bit",
        )

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
