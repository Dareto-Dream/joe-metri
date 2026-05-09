from __future__ import annotations

import base64
import gzip
import unittest

from gd_scraper.parser import (
    ValidationError,
    candidate_from_search_level,
    comment_page_record,
    difficulty_label,
    level_record,
    parse_download_response,
    parse_key_value_pairs,
    parse_comment_response,
    parse_search_response,
    validate_level_data,
)


def encoded_level(raw: str) -> str:
    return base64.urlsafe_b64encode(gzip.compress(raw.encode("utf-8"))).decode("ascii").rstrip("=")


class ParserTests(unittest.TestCase):
    def test_parse_key_value_pairs(self) -> None:
        self.assertEqual(
            parse_key_value_pairs("1:123:2:Example:10:50"),
            {"1": "123", "2": "Example", "10": "50"},
        )

    def test_parse_search_response_and_candidate(self) -> None:
        raw = (
            "1:123:2:Example:5:1:6:42:8:10:9:50:10:1000:12:0:14:55:15:3:17:0:18:6:25::35:999"
            "#42:Player:7"
            "#1~|~999~|~2~|~Song Name~|~4~|~Artist"
            "#1:0:10#hash"
        )
        parsed = parse_search_response(raw)
        self.assertEqual(len(parsed.levels), 1)
        self.assertIn("_raw", parsed.levels[0])
        self.assertEqual(parsed.creators[42].username, "Player")
        self.assertEqual(parsed.songs[0].song_id, 999)

        candidate = candidate_from_search_level(parsed.levels[0], parsed.creators, "test", 0, raw, sequence=7)
        self.assertIsNotNone(candidate)
        assert candidate is not None
        self.assertEqual(candidate.level_id, 123)
        self.assertEqual(candidate.sequence, 7)
        self.assertEqual(candidate.author, "Player")
        self.assertEqual(candidate.metadata["difficulty"], "INSANE")

    def test_download_validation(self) -> None:
        level_data = encoded_level("kS1,0;k1,1,2,15;k1,2,2,30")
        raw = f"1:123:2:Example:4:{level_data}:6:42:8:10:9:30:10:10:12:0:14:5#hash1#hash2##"
        download = parse_download_response(raw)
        result = validate_level_data(download.level["4"])
        self.assertEqual(result.object_count, 2)

        candidate = candidate_from_search_level(
            {"1": "123", "2": "Example", "6": "42"},
            {},
            "featured",
            12,
            "1:123:2:Example",
            sequence=3,
        )
        assert candidate is not None
        record = level_record(download, candidate, result)
        self.assertEqual(record["dataset_version"], 1)
        self.assertEqual(record["scraper_version"], "1.0.0")
        self.assertEqual(record["level_id"], 123)
        self.assertEqual(record["song_type"], "official")
        self.assertEqual(record["source_page"], 12)
        self.assertEqual(record["level_hash"], "hash1")
        self.assertIsInstance(record["fetched_at"], int)
        self.assertEqual(record["metadata"]["candidate_sequence"], 3)

    def test_invalid_level_data_rejected(self) -> None:
        with self.assertRaises(ValidationError) as caught:
            validate_level_data("not-valid")
        self.assertEqual(caught.exception.reason, "base64_decode_failed")

    def test_demon_difficulty_mapping(self) -> None:
        self.assertEqual(difficulty_label({"17": "1", "43": "6"}), "EXTREME_DEMON")

    def test_comment_response_record_preserves_raw(self) -> None:
        raw = "2~Comment~3~42#1:0:10"
        comments, page_info = parse_comment_response(raw)
        self.assertEqual(comments, ["2~Comment~3~42"])
        self.assertEqual(page_info, {"total": 1, "offset": 0, "amount": 10})

        record = comment_page_record(
            level_id=123,
            source="featured",
            source_page=0,
            page=0,
            mode=0,
            count=10,
            raw=raw,
        )
        self.assertEqual(record["level_id"], 123)
        self.assertEqual(record["comment_count"], 1)
        self.assertEqual(record["raw"], raw)


if __name__ == "__main__":
    unittest.main()
