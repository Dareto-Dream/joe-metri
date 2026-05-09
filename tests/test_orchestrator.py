from __future__ import annotations

import base64
from collections import Counter
import gzip
import tempfile
from pathlib import Path
import unittest

from gd_scraper.orchestrator import (
    ContinuousPipelineOrchestrator,
    OrchestratorConfig,
    merge_vocab_append_only,
)
from gd_scraper.storage import CheckpointStore, count_lines, dumps_jsonl, dumps_pretty, loads_json
from gd_scraper.tokenizer import build_vocab


def encoded_level(raw: str) -> str:
    return base64.urlsafe_b64encode(gzip.compress(raw.encode("utf-8"))).decode("ascii").rstrip("=")


def raw_level_record(level_id: int) -> dict[str, object]:
    raw_level = (
        "kS1,0;"
        "1,1,2,15,3,15;"
        "1,1,2,45,3,15;"
        "1,8,2,105,3,45;"
        "1,36,2,135,3,105"
    )
    return {
        "level_id": level_id,
        "difficulty": "HARD",
        "song_id": 999,
        "source": "featured",
        "level_data": encoded_level(raw_level),
    }


class OrchestratorTests(unittest.IsolatedAsyncioTestCase):
    async def test_tokenizer_cycle_tails_raw_queue_and_deduplicates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_dir = Path(tmp_dir) / "data"
            store = CheckpointStore(data_dir)
            store.ensure()
            store.levels_path.write_bytes(dumps_jsonl(raw_level_record(123)))

            orchestrator = ContinuousPipelineOrchestrator(
                OrchestratorConfig(
                    data_dir=data_dir,
                    model_dir=Path(tmp_dir) / "models",
                    scraper_enabled=False,
                    trainer_enabled=False,
                    evaluator_enabled=False,
                    min_gameplay_objects=1,
                    min_tokens=1,
                )
            )

            first_metrics = await orchestrator.run_once()
            second_metrics = await orchestrator.run_once()

            self.assertEqual(count_lines(store.mechanics_tokens_path), 1)
            self.assertEqual(first_metrics["dashboard"]["levels_tokenized"], 1)
            self.assertEqual(second_metrics["dashboard"]["levels_tokenized"], 1)
            self.assertIn("123", store.orchestrator_raw_processed_ids_path.read_text(encoding="utf-8"))

            vocab = loads_json(store.vocab_path.read_bytes())
            self.assertEqual(vocab["token_to_id"]["START"], 2)
            self.assertIn("BLOCK", vocab["token_to_id"])

    async def test_training_and_evaluation_cycle_write_live_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_dir = Path(tmp_dir) / "data"
            model_dir = Path(tmp_dir) / "models"
            store = CheckpointStore(data_dir)
            store.ensure()
            token_record = {
                "level_id": 1,
                "difficulty": "HARD",
                "source": "test",
                "tokens": ["START", "DIFF_HARD", "ALIGN_UNKNOWN", "BLOCK", "Y0", "WIDTH_1", "STEP", "END"],
            }
            store.mechanics_tokens_path.write_bytes(b"".join(dumps_jsonl({**token_record, "level_id": item}) for item in range(5)))
            store.vocab_path.write_bytes(dumps_pretty(build_vocab(token_counts=Counter(token_record["tokens"]))))

            orchestrator = ContinuousPipelineOrchestrator(
                OrchestratorConfig(
                    data_dir=data_dir,
                    model_dir=model_dir,
                    scraper_enabled=False,
                    tokenizer_enabled=False,
                    train_min_records=1,
                    train_window_records=5,
                    train_max_examples=100,
                    sample_tokens=24,
                    min_sample_tokens=10,
                )
            )

            await orchestrator.run_once()

            self.assertTrue((model_dir / "model_live.json").exists())
            self.assertTrue((model_dir / "training_stats_live.jsonl").exists())
            sample = loads_json((model_dir / "sample_generation_live.json").read_bytes())
            self.assertEqual(sample["tokens"][0], "START")
            self.assertEqual(sample["tokens"][-1], "END")


class VocabMergeTests(unittest.TestCase):
    def test_merge_vocab_append_only_preserves_existing_token_ids(self) -> None:
        existing = build_vocab(token_counts=Counter({"START": 1, "STEP": 2}))
        previous_ids = dict(existing["token_to_id"])

        merged = merge_vocab_append_only(
            existing,
            [{"tokens": ["START", "NEW_MECHANIC", "STEP"]}],
            [{"unknown_object_count": 2, "top_unknown_object_ids": [{"object_id": 9999, "count": 2}]}],
        )

        for token, token_id in previous_ids.items():
            self.assertEqual(merged["token_to_id"][token], token_id)
        self.assertEqual(merged["token_to_id"]["NEW_MECHANIC"], len(previous_ids))
        self.assertEqual(merged["token_counts"]["START"], 2)
        self.assertEqual(merged["token_counts"]["STEP"], 3)
        self.assertEqual(merged["unknown_object_count"], 2)
        self.assertEqual(merged["vocab_version"], existing["vocab_version"] + 1)


if __name__ == "__main__":
    unittest.main()
