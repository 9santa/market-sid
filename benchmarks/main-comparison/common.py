"""Use the main dataset's existing IDs, splits, and popularity groups."""

import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import polars as pl
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset

BENCHMARK_DIR = Path(__file__).resolve().parent
ROOT = BENCHMARK_DIR.parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_genrec import make_eval_records

DATA_DIR = ROOT / "data/processed"
CHECKPOINT_DIR = BENCHMARK_DIR / "checkpoints"
RESULT_DIR = ROOT / "artifacts" / "benchmarks" / BENCHMARK_DIR.name
ARTIFACT_DIR = RESULT_DIR / "evaluation"
HISTORY_DIR = RESULT_DIR / "training"
REPORT_DIR = RESULT_DIR / "reports"
GROUPS = ("all", "warm", "head", "mid", "tail", "unseen")
KS = (10, 20, 50)


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def users_hash(user_ids):
    return hashlib.sha256(np.asarray(user_ids, dtype="<i8").tobytes()).hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


class MainData:
    def __init__(self, directory):
        directory = Path(directory)
        catalog = pl.read_parquet(directory / "catalog_model.parquet", columns=["item_idx"])
        users = pl.read_parquet(directory / "user_sequences.parquet").sort("user_idx")
        self.num_items = len(catalog)
        self.num_users = len(users)
        if sorted(catalog["item_idx"].to_list()) != list(range(1, self.num_items + 1)):
            raise ValueError("Expected item IDs 1..num_items, with 0 reserved for PAD")
        if users["user_idx"].to_list() != list(range(self.num_users)):
            raise ValueError("Expected user IDs 0..num_users-1")
        self.sequences = []
        for row in users.iter_rows(named=True):
            sequence = np.asarray(row["item_sequence"], dtype=np.int64)
            if len(sequence) < 4 or row["split_sequence"] != ["train"] * (len(sequence) - 2) + ["valid", "test"]:
                raise ValueError(f"Invalid leave-two-out sequence for user {row['user_idx']}")
            if np.any(sequence < 1) or np.any(sequence > self.num_items):
                raise ValueError("Sequence contains an unknown item ID")
            self.sequences.append(sequence)
        self.train = [sequence[:-2] for sequence in self.sequences]
        self.train_items = np.concatenate(self.train)
        self.train_users = np.repeat(np.arange(self.num_users), [len(x) for x in self.train])
        self.counts = np.bincount(self.train_items, minlength=self.num_items + 1)
        self.warm_ids = np.flatnonzero(self.counts)
        self.signature = {
            name: file_hash(directory / name)
            for name in ("catalog_model.parquet", "user_sequences.parquet", "interactions_model.parquet")
        }
        # Match GenRec's grouping source, and catch stale sequence exports.
        interactions = pl.read_parquet(directory / "interactions_model.parquet", columns=["item_idx", "split"])
        train_ids = interactions.filter(pl.col("split") == "train")["item_idx"].to_numpy()
        counts = np.bincount(train_ids, minlength=self.num_items + 1)
        if not np.array_equal(self.counts, counts):
            raise ValueError("Sequence and interaction training counts disagree")

    def num(self, field):
        # Minimal dataset interface required by the RecBole model classes.
        return self.num_users if field == "user_id" else self.num_items + 1

    def inter_matrix(self, form="csr"):
        matrix = sp.coo_matrix(
            (np.ones(len(self.train_items), dtype=np.float32), (self.train_users, self.train_items)),
            shape=(self.num_users, self.num_items + 1),
        ).tocsr()
        matrix.data[:] = 1  # Binary train-only graph/co-occurrence; collapse repeats.
        return matrix.asformat(form)

    def records(self, split, limit=None):
        records = make_eval_records(self.sequences, self.counts, split)
        for user, record in enumerate(records):
            record["user_idx"] = user
        return records if limit is None else records[:limit]


class TrainExamples(Dataset):
    def __init__(self, data, model, max_len, seed):
        self.data, self.model, self.max_len, self.seed = data, model, max_len, seed
        self.epoch = 0
        self.offsets = np.cumsum([0] + [len(x) - 1 for x in data.train])

    def __len__(self):
        return len(self.data.train_items) if self.model == "lightgcn" else int(self.offsets[-1])

    def __getitem__(self, index):
        if self.model == "lightgcn":
            user = int(self.data.train_users[index])
            target = int(self.data.train_items[index])
            rng = np.random.default_rng(self.seed + self.epoch * 1_000_003 + index)
            seen = self.data.train[user]
            if len(np.unique(seen)) == len(self.data.warm_ids):
                raise ValueError("User has no eligible warm negative")
            negative = int(rng.choice(self.data.warm_ids))
            while negative in seen:
                negative = int(rng.choice(self.data.warm_ids))
            return {"user_id": user, "item_id": target, "neg_item_id": negative}
        user = int(np.searchsorted(self.offsets, index, side="right") - 1)
        position = int(index - self.offsets[user] + 1)
        history = self.data.train[user][max(0, position - self.max_len):position]
        padded = np.zeros(self.max_len, dtype=np.int64)
        padded[:len(history)] = history  # RecBole GRU expects right padding.
        return {"user_id": user, "item_id": int(self.data.train[user][position]),
                "item_id_list": padded, "item_length": len(history)}


def history_batch(records, max_len, device):
    padded = np.zeros((len(records), max_len), dtype=np.int64)
    lengths = []
    for i, record in enumerate(records):
        history = record["history"][-max_len:]
        padded[i, :len(history)] = history
        lengths.append(len(history))
    return {
        "user_id": torch.tensor([r["user_idx"] for r in records], device=device),
        "item_id_list": torch.from_numpy(padded).to(device),
        "item_length": torch.tensor(lengths, device=device),
    }


def add_metrics(totals, record, recommendations):
    groups = ["all", record["group"]]
    if record["group"] != "unseen":
        groups.append("warm")
    rank = recommendations.index(record["target"]) + 1 if record["target"] in recommendations else None
    for group in groups:
        totals[group]["n"] += 1
        for k in KS:
            if rank is not None and rank <= k:
                totals[group][f"recall@{k}"] += 1
                totals[group][f"ndcg@{k}"] += 1 / math.log2(rank + 1)


@torch.inference_mode()
def evaluate(model, name, data, records, config, device, batch_size=64, beam_size=100):
    from src.models.beam_search import batched_constrained_beam_search, filter_seen_items
    from src.models.sid_trie import SIDTrie

    if name != "itemknn":
        model.eval()
    trie = SIDTrie(model.item_sids.cpu().numpy()) if name == "tiger" else None
    totals = {g: {"n": 0, **{f"{metric}@{k}": 0.0 for k in KS for metric in ("recall", "ndcg")}} for g in GROUPS}
    short, seen_targets = 0, 0
    for start in range(0, len(records), batch_size):
        rows = records[start:start + batch_size]
        if name == "tiger":
            batch = history_batch(rows, config["max_len"], device)
            inputs = torch.cat([batch["user_id"][:, None], batch["item_id_list"]], dim=1)
            beams = batched_constrained_beam_search(model, trie, inputs, beam_size=beam_size,
                                                  decode_batch_size=config["decode_batch_size"])
            ranked = [[i for i, _ in filter_seen_items(beam, row["history"], 50)] for beam, row in zip(beams, rows)]
        else:
            if name == "itemknn":
                scores = torch.from_numpy(model.score(rows, data.num_items))
            else:
                scores = model.full_sort_predict(history_batch(rows, config["max_len"], device))
                scores = scores.reshape(len(rows), data.num_items + 1)
            if not torch.isfinite(scores).all():
                raise ValueError("Non-finite recommendation scores")
            scores[:, 0] = -torch.inf
            for i, row in enumerate(rows):
                scores[i, torch.as_tensor(row["history"], device=scores.device)] = -torch.inf
            # Stable sort breaks exact ties by ascending item ID for reproducibility.
            ids = torch.argsort(scores, dim=1, descending=True, stable=True)[:, :min(50, data.num_items)]
            finite = torch.gather(scores, 1, ids).isfinite()
            ranked = [indices[mask].tolist() for indices, mask in zip(ids.cpu(), finite.cpu())]
        for row, recommendations in zip(rows, ranked):
            short += len(recommendations) < 50
            seen_targets += row["target"] in row["history"]
            add_metrics(totals, row, recommendations)
    metrics = {}
    for group, values in totals.items():
        if values["n"]:
            metrics[group] = {key: value if key == "n" else value / values["n"] for key, value in values.items()}
    return metrics, {"short_recommendations": short, "seen_targets": seen_targets}
