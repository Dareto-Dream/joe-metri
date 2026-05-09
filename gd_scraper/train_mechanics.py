from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Any

from .gd_objects import (
    DIFFICULTY_TOKENS,
    PRIMARY_MECHANIC_TOKENS,
    SOLID_TOKENS,
    TOKENIZER_VERSION,
    WIDTH_TOKENS,
    Y_TOKENS,
)
from .models import DATASET_VERSION, epoch_seconds
from .storage import CheckpointStore, dumps_jsonl, dumps_pretty, loads_json


def load_token_records(path: Path, limit: int = 0) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("rb") as handle:
        for raw_line in handle:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            record = loads_json(raw_line)
            if isinstance(record.get("tokens"), list):
                records.append(record)
            if limit > 0 and len(records) >= limit:
                break
    return records


def build_examples(
    records: list[dict[str, Any]],
    token_to_id: dict[str, int],
    context_size: int,
    max_examples: int,
) -> list[tuple[list[int], int]]:
    pad_id = token_to_id["<PAD>"]
    unk_id = token_to_id["<UNK>"]
    examples: list[tuple[list[int], int]] = []
    for record in records:
        token_ids = [token_to_id.get(str(token), unk_id) for token in record["tokens"]]
        prefix = [pad_id] * context_size
        history = [*prefix, *token_ids]
        for index, target_id in enumerate(token_ids):
            context = history[index : index + context_size]
            examples.append((context, target_id))
            if max_examples > 0 and len(examples) >= max_examples:
                return examples
    return examples


class TinyNgramModel:
    def __init__(
        self,
        *,
        vocab_size: int,
        context_size: int,
        embedding_dim: int,
        seed: int,
    ) -> None:
        rng = random.Random(seed)
        self.vocab_size = vocab_size
        self.context_size = context_size
        self.embedding_dim = embedding_dim
        self.embeddings = [
            [rng.uniform(-0.05, 0.05) for _ in range(embedding_dim)]
            for _ in range(vocab_size)
        ]
        self.output_weights = [
            [rng.uniform(-0.05, 0.05) for _ in range(vocab_size)]
            for _ in range(embedding_dim)
        ]
        self.output_bias = [0.0 for _ in range(vocab_size)]

    def expand_vocab(self, new_vocab_size: int, *, seed: int) -> None:
        if new_vocab_size <= self.vocab_size:
            return

        rng = random.Random(seed + self.vocab_size)
        added = new_vocab_size - self.vocab_size
        for _ in range(added):
            self.embeddings.append([rng.uniform(-0.05, 0.05) for _ in range(self.embedding_dim)])
            self.output_bias.append(0.0)
        for weights in self.output_weights:
            weights.extend(rng.uniform(-0.05, 0.05) for _ in range(added))
        self.vocab_size = new_vocab_size

    def context_vector(self, context: list[int]) -> list[float]:
        vector = [0.0 for _ in range(self.embedding_dim)]
        scale = 1.0 / max(len(context), 1)
        for token_id in context:
            embedding = self.embeddings[token_id]
            for index in range(self.embedding_dim):
                vector[index] += embedding[index] * scale
        return vector

    def logits(self, context: list[int]) -> tuple[list[float], list[float]]:
        hidden = self.context_vector(context)
        logits = self.output_bias[:]
        for dim_index, hidden_value in enumerate(hidden):
            weights = self.output_weights[dim_index]
            for token_index in range(self.vocab_size):
                logits[token_index] += hidden_value * weights[token_index]
        return hidden, logits

    def probabilities(self, context: list[int]) -> list[float]:
        _hidden, logits = self.logits(context)
        return softmax(logits)

    def train_one(self, context: list[int], target_id: int, learning_rate: float) -> float:
        hidden, logits = self.logits(context)
        probabilities = softmax(logits)
        loss = -math.log(max(probabilities[target_id], 1e-12))
        probabilities[target_id] -= 1.0

        hidden_gradient = [0.0 for _ in range(self.embedding_dim)]
        for dim_index in range(self.embedding_dim):
            weights = self.output_weights[dim_index]
            for token_index, grad in enumerate(probabilities):
                hidden_gradient[dim_index] += weights[token_index] * grad

        for dim_index, hidden_value in enumerate(hidden):
            weights = self.output_weights[dim_index]
            for token_index, grad in enumerate(probabilities):
                weights[token_index] -= learning_rate * hidden_value * grad

        for token_index, grad in enumerate(probabilities):
            self.output_bias[token_index] -= learning_rate * grad

        context_scale = 1.0 / max(len(context), 1)
        for token_id in context:
            embedding = self.embeddings[token_id]
            for dim_index, grad in enumerate(hidden_gradient):
                embedding[dim_index] -= learning_rate * grad * context_scale

        return loss

    def to_json(self) -> dict[str, Any]:
        return {
            "model_type": "tiny_embedding_ngram",
            "dataset_version": DATASET_VERSION,
            "tokenizer_version": TOKENIZER_VERSION,
            "context_size": self.context_size,
            "embedding_dim": self.embedding_dim,
            "vocab_size": self.vocab_size,
            "embeddings": self.embeddings,
            "output_weights": self.output_weights,
            "output_bias": self.output_bias,
        }


def model_from_json(value: dict[str, Any]) -> TinyNgramModel:
    model = TinyNgramModel(
        vocab_size=int(value["vocab_size"]),
        context_size=int(value["context_size"]),
        embedding_dim=int(value["embedding_dim"]),
        seed=0,
    )
    model.embeddings = [[float(item) for item in row] for row in value["embeddings"]]
    model.output_weights = [[float(item) for item in row] for row in value["output_weights"]]
    model.output_bias = [float(item) for item in value["output_bias"]]
    model.vocab_size = len(model.output_bias)
    return model


def softmax(logits: list[float]) -> list[float]:
    maximum = max(logits)
    exps = [math.exp(max(-50.0, min(50.0, value - maximum))) for value in logits]
    total = sum(exps)
    return [value / total for value in exps]


def evaluate_loss(model: TinyNgramModel, examples: list[tuple[list[int], int]], max_eval_examples: int = 5000) -> float:
    if not examples:
        return 0.0
    total = 0.0
    count = 0
    for context, target_id in examples[:max_eval_examples]:
        probabilities = model.probabilities(context)
        total += -math.log(max(probabilities[target_id], 1e-12))
        count += 1
    return total / max(count, 1)


def allowed_next_tokens(tokens: list[str], *, min_generated: int, max_consecutive_steps: int) -> list[str]:
    if not tokens:
        return ["START"]
    last = tokens[-1]
    if last == "START":
        return DIFFICULTY_TOKENS
    if last.startswith("DIFF_"):
        return ["ALIGN_UNKNOWN"]
    if last == "ALIGN_UNKNOWN":
        return PRIMARY_MECHANIC_TOKENS + ["STEP"]
    if last in PRIMARY_MECHANIC_TOKENS:
        return Y_TOKENS
    if last in Y_TOKENS:
        object_token = previous_object_token(tokens)
        if object_token in SOLID_TOKENS:
            return WIDTH_TOKENS
        return end_capable_tokens(tokens, min_generated, max_consecutive_steps)
    if last in WIDTH_TOKENS:
        return end_capable_tokens(tokens, min_generated, max_consecutive_steps)
    if last == "STEP":
        return end_capable_tokens(tokens, min_generated, max_consecutive_steps)
    if last == "END":
        return []
    return ["END"]


def end_capable_tokens(tokens: list[str], min_generated: int, max_consecutive_steps: int) -> list[str]:
    allowed = [*PRIMARY_MECHANIC_TOKENS, "STEP"]
    if len(tokens) >= min_generated:
        allowed.append("END")
    if consecutive_steps(tokens) >= max_consecutive_steps and "STEP" in allowed:
        allowed.remove("STEP")
    return allowed


def previous_object_token(tokens: list[str]) -> str:
    for token in reversed(tokens[:-1]):
        if token in PRIMARY_MECHANIC_TOKENS:
            return token
        if token in {"STEP", "START", "ALIGN_UNKNOWN"} or token.startswith("DIFF_"):
            break
    return ""


def consecutive_steps(tokens: list[str]) -> int:
    count = 0
    for token in reversed(tokens):
        if token != "STEP":
            break
        count += 1
    return count


def sample_from_allowed(
    probabilities: list[float],
    allowed_tokens: list[str],
    token_to_id: dict[str, int],
    id_to_token: list[str],
    rng: random.Random,
    temperature: float,
) -> str:
    allowed_ids = [token_to_id[token] for token in allowed_tokens if token in token_to_id]
    if not allowed_ids:
        return "END"
    weights = []
    for token_id in allowed_ids:
        probability = max(probabilities[token_id], 1e-12)
        weights.append(probability ** (1.0 / max(temperature, 1e-6)))
    total = sum(weights)
    threshold = rng.random() * total
    running = 0.0
    for token_id, weight in zip(allowed_ids, weights):
        running += weight
        if running >= threshold:
            return id_to_token[token_id]
    return id_to_token[allowed_ids[-1]]


def generate_sample(
    model: TinyNgramModel,
    *,
    token_to_id: dict[str, int],
    id_to_token: list[str],
    prefix: list[str],
    max_new_tokens: int,
    min_generated: int,
    seed: int,
    temperature: float,
) -> list[str]:
    rng = random.Random(seed)
    pad_id = token_to_id["<PAD>"]
    unk_id = token_to_id["<UNK>"]
    tokens = prefix[:]
    while len(tokens) < max_new_tokens and (not tokens or tokens[-1] != "END"):
        context_tokens = tokens[-model.context_size :]
        context_ids = [token_to_id.get(token, unk_id) for token in context_tokens]
        context_ids = [pad_id] * max(0, model.context_size - len(context_ids)) + context_ids
        probabilities = model.probabilities(context_ids)
        allowed = allowed_next_tokens(tokens, min_generated=min_generated, max_consecutive_steps=6)
        next_token = sample_from_allowed(probabilities, allowed, token_to_id, id_to_token, rng, temperature)
        tokens.append(next_token)
    if tokens[-1] != "END":
        close_sample(tokens)
    return tokens


def close_sample(tokens: list[str]) -> None:
    if not tokens:
        tokens.extend(["START", "DIFF_NA", "ALIGN_UNKNOWN", "STEP", "END"])
        return
    last = tokens[-1]
    if last in PRIMARY_MECHANIC_TOKENS:
        tokens.append("Y0")
        if last in SOLID_TOKENS:
            tokens.append("WIDTH_1")
    elif last in Y_TOKENS:
        object_token = previous_object_token(tokens)
        if object_token in SOLID_TOKENS:
            tokens.append("WIDTH_1")
    elif last.startswith("DIFF_"):
        tokens.append("ALIGN_UNKNOWN")
    if tokens[-1] != "STEP":
        tokens.append("STEP")
    tokens.append("END")


def train(args: argparse.Namespace) -> int:
    store = CheckpointStore(args.data_dir)
    vocab_path = store.tokenized_dir / "vocab.json"
    vocab = loads_json(vocab_path.read_bytes())
    token_to_id = {str(key): int(value) for key, value in vocab["token_to_id"].items()}
    id_to_token = [str(token) for token in vocab["id_to_token"]]
    records = load_token_records(store.mechanics_tokens_path, args.max_records)
    examples = build_examples(records, token_to_id, args.context_size, args.max_examples)
    if not examples:
        raise SystemExit("No token examples found. Run gd_scraper.tokenizer first.")

    rng = random.Random(args.seed)
    model = TinyNgramModel(
        vocab_size=len(id_to_token),
        context_size=args.context_size,
        embedding_dim=args.embedding_dim,
        seed=args.seed,
    )

    args.model_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.model_dir / "training_stats.jsonl"
    with metrics_path.open("wb") as metrics_handle:
        for epoch in range(1, args.epochs + 1):
            rng.shuffle(examples)
            total_loss = 0.0
            for context, target_id in examples:
                total_loss += model.train_one(context, target_id, args.learning_rate)
            train_loss = total_loss / max(len(examples), 1)
            eval_loss = evaluate_loss(model, examples)
            metric = {
                "dataset_version": DATASET_VERSION,
                "tokenizer_version": TOKENIZER_VERSION,
                "model_type": "tiny_embedding_ngram",
                "timestamp": epoch_seconds(),
                "epoch": epoch,
                "examples": len(examples),
                "train_loss": round(train_loss, 6),
                "eval_loss": round(eval_loss, 6),
            }
            metrics_handle.write(dumps_jsonl(metric))
            print(f"epoch={epoch} train_loss={train_loss:.4f} eval_loss={eval_loss:.4f}")

    model_json = {
        **model.to_json(),
        "vocab": vocab,
        "trained_at": epoch_seconds(),
        "training": {
            "records": len(records),
            "examples": len(examples),
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
        },
    }
    model_path = args.model_dir / "model.json"
    model_path.write_bytes(dumps_pretty(model_json))

    prefix = args.prefix.split()
    sample = generate_sample(
        model,
        token_to_id=token_to_id,
        id_to_token=id_to_token,
        prefix=prefix,
        max_new_tokens=args.sample_tokens,
        min_generated=args.min_sample_tokens,
        seed=args.seed + 1,
        temperature=args.temperature,
    )
    sample_path = args.model_dir / "sample_generation.json"
    sample_path.write_bytes(
        dumps_pretty(
            {
                "dataset_version": DATASET_VERSION,
                "tokenizer_version": TOKENIZER_VERSION,
                "model_type": "tiny_embedding_ngram",
                "prefix": prefix,
                "tokens": sample,
            }
        )
    )
    print("sample:")
    print(" ".join(sample))
    print(f"wrote {model_path}, {metrics_path}, and {sample_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gd-train-mechanics",
        description="Train a tiny mechanics-token proof-of-life model.",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--model-dir", type=Path, default=Path("models") / "mechanics_v1")
    parser.add_argument("--max-records", type=int, default=100)
    parser.add_argument("--max-examples", type=int, default=30_000)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--context-size", type=int, default=4)
    parser.add_argument("--embedding-dim", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--prefix", default="START DIFF_HARD ALIGN_UNKNOWN")
    parser.add_argument("--sample-tokens", type=int, default=120)
    parser.add_argument("--min-sample-tokens", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=0.9)
    return parser


def main() -> int:
    return train(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
