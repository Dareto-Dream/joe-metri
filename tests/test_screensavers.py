from __future__ import annotations

import json
import tempfile
from pathlib import Path
import time
import unittest

from screensavers.main import build_payload


class ScreensaverPayloadTests(unittest.TestCase):
    def test_build_payload_reads_orchestrator_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            data_dir = root / "data"
            model_dir = root / "models"
            (data_dir / "logs").mkdir(parents=True)
            (data_dir / "tokenized").mkdir(parents=True)
            model_dir.mkdir(parents=True)

            now = int(time.time())
            metrics = {
                "timestamp": now,
                "mode": "assisted",
                "dashboard": {
                    "levels_scraped": 10,
                    "levels_tokenized": 7,
                    "tokens_generated": 1234,
                    "tokens_per_second": 12.5,
                    "dataset_size": 7,
                    "queue_sizes": {"raw_queue": 3, "token_queue": 2, "training_queue": 99},
                    "training_loss": 1.25,
                },
                "tokenizer": {"token_entropy": 4.2, "grammar_validity": 1.0},
                "trainer": {"steps": 99, "training_loss": 1.25, "validation_loss": 1.35},
                "allocation": {"training_examples_per_cycle": 100},
            }
            event = {"timestamp": now, "event": "TRAINING_STEP"}
            (data_dir / "logs" / "orchestrator_metrics.jsonl").write_text(json.dumps(metrics) + "\n", encoding="utf-8")
            (data_dir / "logs" / "orchestrator_events.jsonl").write_text(json.dumps(event) + "\n", encoding="utf-8")
            (data_dir / "tokenized" / "vocab.json").write_text(json.dumps({"vocab_size": 42}), encoding="utf-8")

            payload = build_payload(data_dir=data_dir, model_dir=model_dir, stale_after_seconds=10**9)

            self.assertTrue(payload["orchestrator"]["active"])
            self.assertFalse(payload["shutdown_allowed"])
            self.assertEqual(payload["training"]["levels_tokenized"], 7)
            self.assertEqual(payload["training"]["raw_queue"], 3)
            self.assertEqual(payload["training"]["event"], "TRAINING_STEP")
            self.assertEqual(payload["orchestrator"]["vocab_size"], 42)


if __name__ == "__main__":
    unittest.main()
