"""
MindFu Model Export - Convert fine-tuned models for vLLM inference
"""
import logging
import os
import shutil
from pathlib import Path
from typing import Optional

import torch
from huggingface_hub import HfApi
from safetensors.torch import save_file

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelExporter:
    """Export fine-tuned models for deployment."""

    def __init__(self, models_dir: str = "/models"):
        self.models_dir = Path(models_dir)

    def list_checkpoints(self) -> list:
        """List available fine-tuned checkpoints."""
        checkpoints = []

        for path in self.models_dir.glob("fine-tuned/*"):
            if path.is_dir():
                checkpoints.append({
                    "name": path.name,
                    "path": str(path),
                    "has_lora": (path / "lora").exists(),
                    "has_merged": (path / "merged").exists(),
                })

        return sorted(checkpoints, key=lambda x: x["name"], reverse=True)

    def merge_lora(
        self,
        checkpoint_path: str,
        output_path: Optional[str] = None,
        save_method: str = "merged_16bit",
    ) -> str:
        """Merge LoRA weights with base model."""
        from unsloth import FastLanguageModel

        checkpoint = Path(checkpoint_path)
        lora_path = checkpoint / "lora"

        if not lora_path.exists():
            raise ValueError(f"LoRA weights not found at {lora_path}")

        output_path = output_path or str(checkpoint / "merged")

        logger.info(f"Loading LoRA from {lora_path}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(lora_path),
            max_seq_length=4096,
            dtype=None,
            load_in_4bit=False,  # Load in full precision for merging
        )

        logger.info(f"Merging and saving to {output_path}")
        model.save_pretrained_merged(
            output_path,
            tokenizer,
            save_method=save_method,
        )

        return output_path

    def convert_to_safetensors(self, model_path: str) -> str:
        """Convert model to safetensors format for faster loading."""
        model_path = Path(model_path)
        output_path = model_path / "safetensors"
        output_path.mkdir(exist_ok=True)

        # Find all .bin files
        for bin_file in model_path.glob("*.bin"):
            logger.info(f"Converting {bin_file.name}")
            state_dict = torch.load(bin_file, map_location="cpu")
            safetensor_file = output_path / bin_file.name.replace(".bin", ".safetensors")
            save_file(state_dict, str(safetensor_file))

        # Copy config and tokenizer files
        for file in model_path.glob("*.json"):
            shutil.copy(file, output_path / file.name)

        for file in model_path.glob("*.txt"):
            shutil.copy(file, output_path / file.name)

        for file in model_path.glob("*.model"):
            shutil.copy(file, output_path / file.name)

        return str(output_path)

    def quantize_awq(
        self,
        model_path: str,
        output_path: Optional[str] = None,
        bits: int = 4,
    ) -> str:
        """Quantize model to AWQ format for efficient inference."""
        from awq import AutoAWQForCausalLM
        from transformers import AutoTokenizer

        model_path = Path(model_path)
        output_path = output_path or str(model_path.parent / f"{model_path.name}-awq")

        logger.info(f"Loading model from {model_path}")
        model = AutoAWQForCausalLM.from_pretrained(str(model_path))
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))

        logger.info("Quantizing to AWQ...")
        model.quantize(
            tokenizer,
            quant_config={
                "w_bit": bits,
                "q_group_size": 128,
                "zero_point": True,
            },
        )

        logger.info(f"Saving to {output_path}")
        model.save_quantized(output_path)
        tokenizer.save_pretrained(output_path)

        return output_path

    def push_to_hub(
        self,
        model_path: str,
        repo_id: str,
        token: Optional[str] = None,
        private: bool = True,
    ):
        """Push model to HuggingFace Hub."""
        token = token or os.getenv("HF_TOKEN")
        if not token:
            raise ValueError("HuggingFace token required")

        api = HfApi()

        logger.info(f"Pushing {model_path} to {repo_id}")
        api.upload_folder(
            folder_path=model_path,
            repo_id=repo_id,
            repo_type="model",
            token=token,
            private=private,
        )

        logger.info(f"Model pushed to https://huggingface.co/{repo_id}")

    def prepare_for_vllm(
        self,
        checkpoint_path: str,
        output_path: Optional[str] = None,
    ) -> str:
        """Prepare a checkpoint for vLLM deployment."""
        checkpoint = Path(checkpoint_path)

        # Check if merged model exists
        merged_path = checkpoint / "merged"
        if not merged_path.exists():
            logger.info("Merged model not found, merging LoRA...")
            self.merge_lora(checkpoint_path)

        output_path = output_path or str(checkpoint / "vllm-ready")
        output = Path(output_path)
        output.mkdir(parents=True, exist_ok=True)

        # Copy all necessary files
        for file in merged_path.iterdir():
            if file.suffix in [".json", ".bin", ".safetensors", ".model", ".txt"]:
                shutil.copy(file, output / file.name)

        # Create a deployment config
        deploy_config = {
            "model_name": checkpoint.name,
            "source_checkpoint": str(checkpoint),
            "recommended_settings": {
                "max_model_len": 4096,
                "gpu_memory_utilization": 0.9,
                "trust_remote_code": True,
            },
        }

        import json
        with open(output / "deploy_config.json", "w") as f:
            json.dump(deploy_config, f, indent=2)

        logger.info(f"Model prepared for vLLM at {output_path}")
        return str(output)


if __name__ == "__main__":
    exporter = ModelExporter()

    # List available checkpoints
    checkpoints = exporter.list_checkpoints()
    print("Available checkpoints:")
    for cp in checkpoints:
        print(f"  - {cp['name']}: lora={cp['has_lora']}, merged={cp['has_merged']}")

    # If there's a checkpoint, prepare it for vLLM
    if checkpoints:
        latest = checkpoints[0]
        print(f"\nPreparing latest checkpoint: {latest['name']}")
        exporter.prepare_for_vllm(latest["path"])
