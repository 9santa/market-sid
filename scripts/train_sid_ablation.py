"""Train and export four-RQ tokenizers for the main dataset (no collision suffix)."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import polars as pl
import torch
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader, TensorDataset

from scripts.train_genrec import file_hash, save_history, set_seed
from src.data.embedding_cache import embedding_metadata, load_embedding_cache
from src.models.distilled_rqvae import DistilledRQVAE
from src.models.rqvae import RQVAE


DATA_DIR = Path("data/processed")
CHECKPOINT_DIR = Path("checkpoints")
HISTORY_DIR = Path("artifacts/main/training")


def make_model(checkpoint):
    if checkpoint["variant"] == "A":
        return RQVAE(**checkpoint["config"])
    return DistilledRQVAE(teacher_dim=checkpoint["teacher_dim"], **checkpoint["config"])


def assign_unique_fourth_codes(codes, residuals, codewords):
    """Minimize fourth-level squared error with distinct codes per prefix bucket."""
    prefixes, buckets, sizes = np.unique(
        codes[:, :3], axis=0, return_inverse=True, return_counts=True,
    )
    largest = int(sizes.argmax())
    if sizes[largest] > len(codewords):
        raise ValueError(
            f"Prefix {prefixes[largest].tolist()} contains {sizes[largest]} items, "
            f"but only {len(codewords)} fourth-level codewords are available. "
            "Unique assignment is infeasible with these fixed first three codes."
        )
    assigned = codes.copy()
    # Stable ordering preserves catalog order when the solver encounters ties.
    groups = np.split(np.argsort(buckets, kind="stable"), np.cumsum(sizes)[:-1])
    weights = codewords.astype(np.float64)
    extra_error = 0.0
    matched_buckets = 0
    for indices in groups:
        original = codes[indices, 3]
        if len(np.unique(original)) == len(indices):
            # Independent nearest-code choices are already a feasible optimum.
            continue
        residual = residuals[indices].astype(np.float64)
        costs = (
            (residual * residual).sum(axis=1, keepdims=True)
            + (weights * weights).sum(axis=1)
            - 2 * residual @ weights.T
        )
        costs = np.maximum(costs, 0.0)
        # Match the entire bucket, including items whose original code was unique.
        rows, columns = linear_sum_assignment(costs)
        assigned[indices[rows], 3] = columns
        extra_error += float(costs[rows, columns].sum() - costs[np.arange(len(indices)), original].sum())
        matched_buckets += 1
    return assigned, {
        "assignment": "min_cost_unique_fourth_per_prefix",
        "max_prefix_bucket": int(sizes.max()),
        "matched_buckets": matched_buckets,
        "changed_fourth_codes": int(np.count_nonzero(assigned[:, 3] != codes[:, 3])),
        "assignment_extra_squared_error": extra_error,
        "assignment_extra_squared_error_per_item": extra_error / len(codes),
    }


def export_codes(checkpoint_path, source, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    for filename, expected in checkpoint["input_hashes"].items():
        if file_hash(DATA_DIR / filename) != expected:
            raise ValueError(f"{filename} changed since tokenizer training")
    model = make_model(checkpoint).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    embeddings = np.load(DATA_DIR / checkpoint["input_file"], mmap_mode="r")
    batches = []
    residual_batches = []
    if len(model.quantizer.codebooks) != 4:
        raise ValueError("Unique fourth-level assignment requires exactly four RQ codebooks")
    with torch.inference_mode():
        for start in range(0, len(embeddings), 4096):
            x = torch.tensor(embeddings[start:start + 4096], dtype=torch.float32, device=device)
            z = model.encoder(x)
            _, batch_codes, _ = model.quantizer(z)
            residual = z
            for level in range(3):
                residual = residual - model.quantizer.codebooks[level](batch_codes[:, level])
            batches.append(batch_codes.cpu().numpy())
            residual_batches.append(residual.cpu().numpy())
    codes = np.concatenate(batches).astype(np.int32)
    unique, counts = np.unique(codes, axis=0, return_counts=True)
    raw_stats = {
        "raw_collision_rate": 1 - len(unique) / len(codes),
        "raw_max_collision_group": int(counts.max()),
    }
    report_path = Path("artifacts/main/tokenizers") / f"rqvae_{source}_export.json"
    try:
        codes, assignment_stats = assign_unique_fourth_codes(
            codes, np.concatenate(residual_batches),
            model.quantizer.codebooks[3].weight.detach().cpu().numpy(),
        )
    except ValueError as error:
        save_history({**raw_stats, "error": str(error)}, report_path)
        raise
    unique, counts = np.unique(codes, axis=0, return_counts=True)
    if len(unique) != len(codes):
        raise ValueError("Fourth-level matching failed to produce unique full SIDs")
    stats = {
        **raw_stats,
        **assignment_stats,
        "items": len(codes),
        "unique_prefixes": [len(np.unique(codes[:, :depth], axis=0)) for depth in range(1, 5)],
        "used_codes": [len(np.unique(codes[:, level])) for level in range(4)],
        "collision_rate": 1 - len(unique) / len(codes),
        "max_collision_group": int(counts.max()),
    }
    print(json.dumps(stats, indent=2), flush=True)
    save_history(stats, report_path)
    output = DATA_DIR / f"item_semantic_ids_{source}.npy"
    np.save(output, codes)
    save_history({
        **stats,
        "variant": checkpoint["variant"],
        "sid_sha256": file_hash(output),
        "rq_checkpoint_sha256": file_hash(checkpoint_path),
        "catalog_sha256": checkpoint["input_hashes"]["catalog_model.parquet"],
    }, output.with_suffix(".json"))
    print("Saved:", output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("A", "B", "C"), required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--behavior-weight", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--export-only", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0 or args.behavior_weight < 0:
        parser.error("Require positive epochs, batch-size, lr and nonnegative behavior-weight")
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    source = f"rq4_{args.variant.lower()}"
    checkpoint_path = CHECKPOINT_DIR / f"rqvae_{source}_best.pt"
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    if args.export_only:
        export_codes(checkpoint_path, source, device)
        return

    catalog = pl.read_parquet(DATA_DIR / "catalog_model.parquet").sort("item_idx")
    if catalog["item_idx"].to_list() != list(range(1, len(catalog) + 1)):
        raise ValueError("Catalog item IDs must be contiguous, starting at 1")
    # Both exports use this exact catalog order; the teacher stays frozen.
    teacher_path = DATA_DIR / "item_two_tower_embeddings.npy"
    teacher = np.load(teacher_path).astype(np.float32)
    teacher_metadata = json.loads(teacher_path.with_suffix(".json").read_text())
    mapping_hash = hashlib.sha256(
        json.dumps(catalog.select("item_idx", "parent_asin").rows()).encode("utf-8")
    ).hexdigest()
    if teacher_metadata["mapping_hash"] != mapping_hash:
        raise ValueError("Two-Tower embeddings use a different catalog mapping")
    input_file = "item_two_tower_embeddings.npy" if args.variant == "A" else "item_text_embeddings.npy"
    embeddings = teacher if args.variant == "A" else load_embedding_cache(
        DATA_DIR / input_file, embedding_metadata(catalog),
    )
    if teacher.ndim != 2 or embeddings.ndim != 2 or len(teacher) != len(catalog) or len(embeddings) != len(catalog):
        raise ValueError("Embedding rows must match the catalog")
    if not np.isfinite(teacher).all() or not np.isfinite(embeddings).all():
        raise ValueError("Nonfinite embeddings")
    if not np.allclose(np.linalg.norm(teacher, axis=1), 1, atol=1e-5):
        raise ValueError("Expected normalized Two-Tower vectors")

    config = {
        "input_dim": embeddings.shape[1], "hidden_dim": 256, "latent_dim": 128,
        "num_codebooks": 4, "codebook_size": 256, "beta": 0.25,
    }
    metadata = {
        "variant": args.variant, "config": config, "teacher_dim": teacher.shape[1],
        "input_file": input_file, "input_hashes": {
            name: file_hash(DATA_DIR / name) for name in {
                input_file, teacher_path.name, teacher_path.with_suffix(".json").name,
                "catalog_model.parquet",
            }
        },
        "settings": vars(args), "selection_metric": "reconstruction_loss",
    }
    model = make_model(metadata).to(device)
    # Dedicated generator keeps the data order identical for B and C.
    loader = DataLoader(
        TensorDataset(torch.from_numpy(embeddings), torch.from_numpy(teacher)),
        batch_size=args.batch_size, shuffle=True, num_workers=0,
        generator=torch.Generator().manual_seed(args.seed), pin_memory=device.type == "cuda",
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    history = {**metadata, "epochs": []}
    history_path = HISTORY_DIR / f"rqvae_{source}_history.json"
    best_recon = float("inf")
    print(f"SID variant: {args.variant} | device: {device} | input: {embeddings.shape}", flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        totals = dict.fromkeys(("loss", "reconstruction_loss", "vq_loss", "behavior_loss"), 0.0)
        for x, target in loader:
            x, target = x.to(device, non_blocking=True), target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            output = model(x) if args.variant == "A" else model(
                x, target, all_depths=args.variant == "C", behavior_weight=args.behavior_weight,
            )
            if not torch.isfinite(output["loss"]):
                raise ValueError("Nonfinite tokenizer loss")
            output["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            for key in totals:
                if key in output:
                    totals[key] += output[key].item() * len(x)
        averages = {key: value / len(embeddings) for key, value in totals.items()}
        print(f"Epoch {epoch:02d} | " + " | ".join(f"{key}={value:.6f}" for key, value in averages.items()), flush=True)
        history["epochs"].append({"epoch": epoch, **averages})
        if averages["reconstruction_loss"] < best_recon:
            best_recon = averages["reconstruction_loss"]
            torch.save({**metadata, "model_state_dict": model.state_dict(), "epoch": epoch, **averages}, checkpoint_path)
            history["best_epoch"] = epoch
        save_history(history, history_path)
    export_codes(checkpoint_path, source, device)


if __name__ == "__main__":
    main()
