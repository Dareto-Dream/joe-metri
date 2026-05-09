from __future__ import annotations

import unittest

from gd_scraper.reconstruction import validate_token_grammar
from gd_scraper.train_mechanics import close_sample


class TrainMechanicsTests(unittest.TestCase):
    def test_close_sample_finishes_open_object(self) -> None:
        tokens = ["START", "DIFF_HARD", "ALIGN_UNKNOWN", "SPIKE"]
        close_sample(tokens)
        self.assertEqual(tokens[-1], "END")
        self.assertEqual(validate_token_grammar(tokens), [])

    def test_close_sample_finishes_open_solid_width(self) -> None:
        tokens = ["START", "DIFF_HARD", "ALIGN_UNKNOWN", "BLOCK", "Y3"]
        close_sample(tokens)
        self.assertEqual(tokens[-3:], ["WIDTH_1", "STEP", "END"])
        self.assertEqual(validate_token_grammar(tokens), [])


if __name__ == "__main__":
    unittest.main()
