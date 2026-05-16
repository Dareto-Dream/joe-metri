from __future__ import annotations

import base64
import gzip
import unittest

from gd_scraper.gd_objects import BASE_VOCAB_TOKENS, TOKENIZER_VERSION
from gd_scraper.tokenizer import (
    TokenizerConfig,
    build_vocab,
    parse_level_objects,
    tokenized_record_from_level,
)
from runtime.reconstructor import reconstruct_layout


def encoded_level(raw: str) -> str:
    return base64.urlsafe_b64encode(gzip.compress(raw.encode("utf-8"))).decode("ascii").rstrip("=")


class TokenizerTests(unittest.TestCase):
    def test_parse_level_objects(self) -> None:
        header, objects = parse_level_objects("kS1,0;1,8,2,105,3,45,6,90;1,36,2,105,3,105")
        self.assertEqual(header, "kS1,0")
        self.assertEqual(len(objects), 2)
        self.assertEqual(objects[0].object_id, 8)
        self.assertEqual(objects[0].x, 105)
        self.assertEqual(objects[0].y, 45)
        self.assertEqual(objects[0].rotation, 90)

    def test_tokenized_record_groups_width_and_preserves_same_step_order(self) -> None:
        raw_level = (
            "kS1,0;"
            "1,1,2,15,3,15;"
            "1,1,2,45,3,15;"
            "1,1,2,75,3,15;"
            "1,8,2,105,3,45;"
            "1,36,2,105,3,105;"
            "1,13,2,135,3,75"
        )
        level = {
            "level_id": 123,
            "difficulty": "HARD",
            "song_id": 999,
            "song_type": "custom",
            "audio": {"song_id": 999, "song_type": "custom", "conditioning_ready": True},
            "source": "featured",
            "level_data": encoded_level(raw_level),
        }
        artifacts = tokenized_record_from_level(
            level,
            TokenizerConfig(min_gameplay_objects=1, min_tokens=1, max_unknown_ratio=1.0),
        )

        self.assertIsNotNone(artifacts.token_record)
        assert artifacts.token_record is not None
        tokens = artifacts.token_record["tokens"]
        self.assertEqual(tokens[:3], ["START", "DIFF_HARD", "ALIGN_UNKNOWN"])
        self.assertIn("END", tokens)
        self.assertEqual(tokens[3:6], ["BLOCK", "Y0", "WIDTH_3"])

        spike_index = tokens.index("SPIKE")
        orb_index = tokens.index("ORB_YELLOW")
        self.assertLess(spike_index, orb_index)
        self.assertEqual(tokens[spike_index : spike_index + 2], ["SPIKE", "Y1"])
        self.assertEqual(tokens[orb_index : orb_index + 2], ["ORB_YELLOW", "Y3"])
        self.assertEqual(artifacts.token_record["level_id"], 123)
        self.assertEqual(artifacts.token_record["song_type"], "custom")
        self.assertTrue(artifacts.token_record["audio"]["conditioning_ready"])

    def test_tokenized_floor_blocks_roundtrip_to_runtime_export_coordinates(self) -> None:
        raw_level = (
            "kS1,0;"
            "1,1,2,15,3,15;"
            "1,1,2,45,3,15;"
            "1,1,2,75,3,15;"
            "1,8,2,105,3,45;"
            "1,36,2,105,3,105;"
            "1,13,2,135,3,75"
        )
        level = {
            "level_id": 125,
            "difficulty": "HARD",
            "song_id": 999,
            "source": "featured",
            "level_data": encoded_level(raw_level),
        }
        artifacts = tokenized_record_from_level(
            level,
            TokenizerConfig(min_gameplay_objects=1, min_tokens=1, max_unknown_ratio=1.0),
        )

        self.assertIsNotNone(artifacts.token_record)
        assert artifacts.token_record is not None
        layout = reconstruct_layout([str(token) for token in artifacts.token_record["tokens"]])

        self.assertEqual(layout.errors, [])
        self.assertIn("1,1,2,15,3,15", layout.gd_object_strings)
        self.assertIn("1,1,2,45,3,15", layout.gd_object_strings)
        self.assertIn("1,1,2,75,3,15", layout.gd_object_strings)
        self.assertIn("1,8,2,105,3,45", layout.gd_object_strings)
        self.assertIn("1,36,2,105,3,105", layout.gd_object_strings)
        self.assertIn("1,13,2,135,3,75", layout.gd_object_strings)

    def test_rejects_all_objects_in_one_timestep(self) -> None:
        raw_level = "kS1,0;1,8,2,15,3,45;1,36,2,15,3,75;1,13,2,15,3,105"
        level = {
            "level_id": 124,
            "difficulty": "EXTREME_DEMON",
            "song_id": 999,
            "source": "featured",
            "level_data": encoded_level(raw_level),
        }
        artifacts = tokenized_record_from_level(
            level,
            TokenizerConfig(min_gameplay_objects=1, min_tokens=1, max_unknown_ratio=1.0),
        )
        self.assertIsNone(artifacts.token_record)
        self.assertEqual(artifacts.stats_record["reason"], "all_objects_one_timestep")

    def test_vocab_contains_tokenizer_scope(self) -> None:
        vocab = build_vocab(
            [
                {
                    "tokens": [
                        "START",
                        "DIFF_HARD",
                        "ALIGN_UNKNOWN",
                        "BLOCK",
                        "Y0",
                        "WIDTH_1",
                        "STEP",
                        "END",
                    ]
                }
            ]
        )
        self.assertEqual(vocab["tokenizer_version"], TOKENIZER_VERSION)
        self.assertEqual(vocab["token_counts"]["START"], 1)
        self.assertEqual(vocab["unknown_token_count"], 0)
        for token in BASE_VOCAB_TOKENS:
            self.assertIn(token, vocab["token_to_id"])


if __name__ == "__main__":
    unittest.main()
