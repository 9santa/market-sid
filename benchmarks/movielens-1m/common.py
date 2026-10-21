"""Shared paths and model loading for the MovieLens benchmark."""

import hashlib
from pathlib import Path
import sys

import numpy as np
import polars as pl
import torch

BENCHMARK_DIR = Path(__file__).resolve().parent
ROOT = BENCHMARK_DIR.parents[1]
sys.path.insert(0, str(ROOT))

from src.models.generative_recommender import GenerativeRecommender

DATA_DIR = BENCHMARK_DIR / "data" / "processed"
CHECKPOINT_DIR = BENCHMARK_DIR / "checkpoints"
RESULT_DIR = ROOT / "artifacts" / "benchmarks" / BENCHMARK_DIR.name
ARTIFACT_DIR = RESULT_DIR / "evaluation"
HISTORY_DIR = RESULT_DIR / "training"


def load_data(data_dir):
    catalog = pl.read_parquet(data_dir / "catalog.parquet").sort("item_idx")
    users = pl.read_parquet(data_dir / "sequences.parquet").sort("user_idx")
    codes = np.load(data_dir / "semantic_ids.npy")
    if catalog["item_idx"].to_list() != list(range(1, len(catalog) + 1)):
        raise ValueError("Catalog IDs must be contiguous, starting at 1")
    if len(codes) != len(catalog) or len(np.unique(codes, axis=0)) != len(catalog):
        raise ValueError("Semantic IDs must map one-to-one to the catalog")
    sequences = users["item_sequence"].to_list()
    counts = np.bincount(
        np.concatenate([sequence[:-2] for sequence in sequences]),
        minlength=len(catalog) + 1,
    )
    item_sids = torch.from_numpy(
        np.vstack([np.zeros((1, codes.shape[1]), dtype=np.int64), codes])
    )
    # Refuse to evaluate a checkpoint against changed splits, mappings or SIDs.
    signature = {}
    for name in ("catalog.parquet", "sequences.parquet", "semantic_ids.npy"):
        with (data_dir / name).open("rb") as stream:
            signature[name] = hashlib.file_digest(stream, "sha256").hexdigest()
    return item_sids, sequences, counts, signature


def make_model(item_sids, config, vocab_sizes, variant_config, codebooks=None):
    return GenerativeRecommender(
        item_sids=item_sids,
        vocab_sizes=vocab_sizes,
        **config,
        embedding_mode=variant_config["embedding_mode"],
        tie_semantic_output=variant_config["tie_semantic_output"],
        rq_codebooks=codebooks,
    )


def load_model(checkpoint, item_sids, signature, device):
    if checkpoint["data_signature"] != signature:
        raise ValueError("Checkpoint data/SIDs differ; use its original data directory")
    state = checkpoint["model_state_dict"]
    variant = checkpoint["variant_config"]
    codebooks = None
    if variant["embedding_mode"] != "scratch":
        codebooks = [
            state[f"rq_base_{level}"]
            if variant["embedding_mode"] == "rq_anchor"
            else state[f"sid_embeddings.{level}.weight"]
            for level in range(len(checkpoint["vocab_sizes"]) - 1)
        ]
    model = make_model(
        item_sids, checkpoint["model_config"], checkpoint["vocab_sizes"],
        variant, codebooks,
    )
    model.load_state_dict(state)
    return model.to(device).eval()
