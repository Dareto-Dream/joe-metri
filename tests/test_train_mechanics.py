from __future__ import annotations

import unittest

from gd_scraper.reconstruction import validate_token_grammar
from gd_scraper.train_mechanics import TinyNgramModel, close_sample, model_from_json


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

    def test_model_expands_vocab_without_rebuilding_existing_weights(self) -> None:
        model = TinyNgramModel(vocab_size=3, context_size=2, embedding_dim=4, seed=1)
        original_embedding = model.embeddings[0][:]
        model.expand_vocab(5, seed=99)

        self.assertEqual(model.vocab_size, 5)
        self.assertEqual(len(model.embeddings), 5)
        self.assertEqual(len(model.output_bias), 5)
        self.assertEqual(len(model.output_weights[0]), 5)
        self.assertEqual(model.embeddings[0], original_embedding)

        restored = model_from_json(model.to_json())
        self.assertEqual(restored.vocab_size, 5)
        self.assertEqual(restored.embeddings[0], original_embedding)


if __name__ == "__main__":
    unittest.main()
