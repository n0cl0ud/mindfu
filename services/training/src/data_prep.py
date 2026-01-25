"""
MindFu Data Preparation - Convert conversation logs to training format
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import psycopg2
from psycopg2.extras import RealDictCursor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreparator:
    """Prepare training data from conversation logs."""

    def __init__(
        self,
        postgres_host: str = "postgres",
        postgres_port: int = 5432,
        postgres_user: str = "mindfu",
        postgres_password: str = "mindfu_secret",
        postgres_db: str = "mindfu",
    ):
        self.connection_params = {
            "host": postgres_host,
            "port": postgres_port,
            "user": postgres_user,
            "password": postgres_password,
            "database": postgres_db,
        }

    def get_connection(self):
        """Get database connection."""
        return psycopg2.connect(**self.connection_params)

    def export_conversations(
        self,
        output_dir: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        min_quality_score: float = 0.0,
    ) -> int:
        """Export conversations from database to JSON files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        query = """
            SELECT id, messages, response, model, timestamp, metadata
            FROM conversations
            WHERE 1=1
        """
        params = []

        if start_date:
            query += " AND timestamp >= %s"
            params.append(start_date)

        if end_date:
            query += " AND timestamp <= %s"
            params.append(end_date)

        if min_quality_score > 0:
            query += " AND (metadata->>'quality_score')::float >= %s"
            params.append(min_quality_score)

        query += " ORDER BY timestamp"

        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(query, params)
                    rows = cur.fetchall()

            conversations = []
            for row in rows:
                conversations.append({
                    "id": str(row["id"]),
                    "messages": row["messages"],
                    "response": row["response"],
                    "model": row["model"],
                    "timestamp": row["timestamp"].isoformat() if row["timestamp"] else None,
                    "metadata": row["metadata"] or {},
                })

            # Write to JSON file
            output_file = output_path / f"conversations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(output_file, "w") as f:
                json.dump(conversations, f, indent=2)

            logger.info(f"Exported {len(conversations)} conversations to {output_file}")
            return len(conversations)

        except Exception as e:
            logger.error(f"Failed to export conversations: {e}")
            return 0

    def create_training_split(
        self,
        input_dir: str,
        output_dir: str,
        train_ratio: float = 0.9,
        seed: int = 42,
    ):
        """Split conversation data into train/validation sets."""
        import random

        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Load all conversations
        all_conversations = []
        for file in input_path.glob("*.json"):
            with open(file, "r") as f:
                data = json.load(f)
                if isinstance(data, list):
                    all_conversations.extend(data)
                else:
                    all_conversations.append(data)

        # Shuffle
        random.seed(seed)
        random.shuffle(all_conversations)

        # Split
        split_idx = int(len(all_conversations) * train_ratio)
        train_data = all_conversations[:split_idx]
        val_data = all_conversations[split_idx:]

        # Save splits
        with open(output_path / "train.json", "w") as f:
            json.dump(train_data, f, indent=2)

        with open(output_path / "validation.json", "w") as f:
            json.dump(val_data, f, indent=2)

        logger.info(f"Created splits: {len(train_data)} train, {len(val_data)} validation")

    def filter_quality(
        self,
        input_dir: str,
        output_dir: str,
        min_turns: int = 2,
        max_turns: int = 20,
        min_response_length: int = 50,
    ):
        """Filter conversations by quality criteria."""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        filtered = []

        for file in input_path.glob("*.json"):
            with open(file, "r") as f:
                data = json.load(f)
                conversations = data if isinstance(data, list) else [data]

            for conv in conversations:
                messages = conv.get("messages", [])
                response = conv.get("response", {})

                # Apply filters
                if len(messages) < min_turns or len(messages) > max_turns:
                    continue

                response_content = response.get("content", "")
                if len(response_content) < min_response_length:
                    continue

                # Check for empty messages
                if any(not msg.get("content", "").strip() for msg in messages):
                    continue

                filtered.append(conv)

        # Save filtered
        output_file = output_path / "filtered.json"
        with open(output_file, "w") as f:
            json.dump(filtered, f, indent=2)

        logger.info(f"Filtered to {len(filtered)} conversations")

    def augment_with_metadata(
        self,
        input_file: str,
        output_file: str,
    ):
        """Add training metadata to conversations."""
        with open(input_file, "r") as f:
            conversations = json.load(f)

        for conv in conversations:
            # Add training metadata
            messages = conv.get("messages", [])
            response = conv.get("response", {})

            conv["_training_metadata"] = {
                "num_turns": len(messages),
                "response_length": len(response.get("content", "")),
                "has_system_message": any(m.get("role") == "system" for m in messages),
                "total_tokens_estimate": sum(
                    len(m.get("content", "").split()) * 1.3
                    for m in messages + [response]
                ),
            }

        with open(output_file, "w") as f:
            json.dump(conversations, f, indent=2)


if __name__ == "__main__":
    import os

    preparator = DataPreparator(
        postgres_host=os.getenv("POSTGRES_HOST", "localhost"),
        postgres_port=int(os.getenv("POSTGRES_PORT", "5432")),
        postgres_user=os.getenv("POSTGRES_USER", "mindfu"),
        postgres_password=os.getenv("POSTGRES_PASSWORD", "mindfu_secret"),
        postgres_db=os.getenv("POSTGRES_DB", "mindfu"),
    )

    # Export conversations
    preparator.export_conversations("/conversations/raw")

    # Filter by quality
    preparator.filter_quality("/conversations/raw", "/conversations/filtered")

    # Create train/val split
    preparator.create_training_split("/conversations/filtered", "/conversations/splits")
