from __future__ import annotations

import unittest

from gd_scraper.reconstruction import reconstruct_tokens, validate_reconstruction_record, validate_token_grammar


class ReconstructionTests(unittest.TestCase):
    def test_reconstructs_legal_token_sequence(self) -> None:
        tokens = [
            "START",
            "DIFF_HARD",
            "ALIGN_UNKNOWN",
            "BLOCK",
            "Y1",
            "WIDTH_4",
            "SPIKE",
            "Y2",
            "STEP",
            "ORB_YELLOW",
            "Y5",
            "STEP",
            "END",
        ]
        objects, errors = reconstruct_tokens(tokens)
        self.assertEqual(errors, [])
        self.assertEqual(len(objects), 3)
        self.assertEqual(objects[0].token, "BLOCK")
        self.assertEqual(objects[0].x_step, 0)
        self.assertEqual(objects[0].width, 4)
        self.assertEqual(objects[1].x_step, 0)
        self.assertEqual(objects[2].x_step, 1)

    def test_rejects_orphan_attribute(self) -> None:
        errors = validate_token_grammar(["START", "DIFF_HARD", "Y1", "STEP", "END"])
        self.assertIn("orphan_y_token:Y1", errors)

    def test_validates_against_gameplay_objects(self) -> None:
        token_record = {
            "level_id": 123,
            "tokens": [
                "START",
                "DIFF_HARD",
                "ALIGN_UNKNOWN",
                "BLOCK",
                "Y1",
                "WIDTH_2",
                "STEP",
                "ORB_BLUE",
                "Y4",
                "STEP",
                "END",
            ],
        }
        gameplay_record = {
            "level_id": 123,
            "objects": [
                {"token": "BLOCK", "x_step": 0, "y_lane": 1, "width": 2},
                {"token": "ORB_BLUE", "x_step": 1, "y_lane": 4},
            ],
        }
        result = validate_reconstruction_record(token_record, gameplay_record)
        self.assertTrue(result["valid"])
        self.assertEqual(result["reason"], "")


if __name__ == "__main__":
    unittest.main()
