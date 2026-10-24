"""Build frozen content/Two-Tower tokenizers for the new benchmark catalog."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

from common import ROOT, MainData, file_hash, load_features, load_protocol, save_json, checkpoint_matches_protocol
from models import make_model
import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader, TensorDataset

from scripts.train_genrec import set_seed
from src.data.embedding_cache import (
    ENCODER_CONFIG,
    embedding_metadata,
    load_embedding_cache,
    save_embedding_cache,
)
from src.models.rqvae import RQVAE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/final-v1")
    parser.add_argument(
        "--protocol", type=Path, default=Path(__file__).with_name("final_protocol.json")
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path(__file__).parent / "checkpoints/final-v1/tokenizers",
    )
    parser.add_argument(
        "--history-dir",
        type=Path,
        default=ROOT / "artifacts/benchmarks/final-main/training/tokenizers",
    )
    parser.add_argument(
        "--embeddings-only",
        action="store_true",
        help="Prepare text features before training the teacher",
    )
    parser.add_argument(
        "--teacher-checkpoint",
        type=Path,
        help="Train Two-Tower SID tokenizer using a newly fitted teacher",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--encode-batch-size", type=int, default=256)
    args = parser.parse_args()
    if args.encode_batch_size < 1 or (args.embeddings_only and args.teacher_checkpoint):
        parser.error(
            "Require positive batch size; embeddings-only does not use a teacher"
        )
    data = MainData(args.data_dir)
    protocol = load_protocol(args.protocol, data)
    catalog = pl.read_parquet(args.data_dir / "catalog_model.parquet").sort("item_idx")
    path = args.data_dir / "item_text_embeddings.npy"
    metadata = embedding_metadata(catalog)
    if path.exists():
        content = load_embedding_cache(path, metadata)
    else:
        from sentence_transformers import SentenceTransformer

        encoder = SentenceTransformer(
            ENCODER_CONFIG["model_name"],
            revision=ENCODER_CONFIG["revision"],
            device=args.device,
        )
        encoder.max_seq_length = ENCODER_CONFIG["max_seq_length"]
        content = encoder.encode(
            catalog["item_text"].to_list(),
            batch_size=args.encode_batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
        ).astype(np.float32)
        save_embedding_cache(path, content, metadata)
        del encoder
        if args.device == "cuda":
            torch.cuda.empty_cache()
    if args.embeddings_only:
        print("Saved/validated:", path)
        return
    source = "twotower" if args.teacher_checkpoint else "content"
    sid_path = args.data_dir / f"item_semantic_ids_{source}.npy"
    checkpoint_path = args.checkpoint_dir / f"rqvae_{source}_best.pt"
    history_path = args.history_dir / f"rqvae_{source}_history.json"
    if sid_path.exists() or checkpoint_path.exists() or history_path.exists():
        parser.error(
            "Tokenizer outputs exist; use a new benchmark directory rather than replacing IDs"
        )
    embeddings, teacher_hash = content, None
    if args.teacher_checkpoint:
        teacher = torch.load(
            args.teacher_checkpoint, map_location="cpu", weights_only=True
        )
        if (
            teacher["model"] != "two_tower"
            or teacher["data_signature"] != data.signature
            or not checkpoint_matches_protocol(teacher, args.protocol, protocol)
            or teacher["config"]["seed"] != protocol["teacher_seed"]
            or teacher["selection_metric"]
            != "valid.all." + protocol["selection_metric"]
        ):
            raise ValueError(
                "Teacher must be the registered seed's validation-selected Two-Tower from this benchmark"
            )
        teacher_history = json.loads(Path(teacher["history_file"]).read_text())
        if (
            not teacher_history.get("finished_at")
            or teacher_history["run_id"] != teacher["run_id"]
            or teacher_history["best_epoch"] != teacher["epoch"]
        ):
            raise ValueError("Finish the teacher run before exporting its tokenizer")
        inputs, features = load_features(data, "two_tower")
        if teacher["embeddings_sha256"] != features["embeddings_sha256"]:
            raise ValueError("Teacher text embeddings changed")
        tower = make_model("two_tower", teacher["config"], data, args.device, **inputs)
        tower.load_state_dict(teacher["model_state_dict"])
        tower.eval()
        with torch.inference_mode():
            embeddings = np.concatenate(
                [
                    tower.item_vectors(ids).cpu().numpy()
                    for ids in torch.arange(
                        1, data.num_items + 1, device=args.device
                    ).split(8192)
                ]
            )
        teacher_hash = file_hash(args.teacher_checkpoint)
        del tower
    set_seed(protocol["tokenizer_seed"])
    config = {
        "input_dim": embeddings.shape[1],
        "hidden_dim": 256,
        "latent_dim": 128,
        "num_codebooks": 3,
        "codebook_size": 256,
        "beta": 0.25,
    }
    model = RQVAE(**config).to(args.device)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(embeddings)), batch_size=1024, shuffle=True
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    history = {
        "config": config,
        "protocol_sha256": file_hash(args.protocol),
        "data_signature": data.signature,
        "embeddings_sha256": file_hash(path),
        "teacher_sha256": teacher_hash,
        "epochs": [],
        "selection_metric": "catalog reconstruction loss (tokenizer only, no interaction labels)",
    }
    best = float("inf")
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, protocol["tokenizer_epochs"] + 1):
        model.train()
        loss_sum = 0.0
        for (x,) in loader:
            x = x.to(args.device)
            optimizer.zero_grad(set_to_none=True)
            output = model(x)
            if not torch.isfinite(output["loss"]):
                raise ValueError("Non-finite tokenizer loss")
            output["loss"].backward()
            optimizer.step()
            loss_sum += output["reconstruction_loss"].item() * len(x)
        reconstruction = loss_sum / len(embeddings)
        history["epochs"].append(
            {"epoch": epoch, "reconstruction_loss": reconstruction}
        )
        if reconstruction < best:
            best = reconstruction
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "catalog_hash": data.signature["catalog_model.parquet"],
                },
                checkpoint_path,
            )
        save_json(history_path, history)
        print(f"RQ {source} epoch {epoch:02d} | recon={reconstruction:.6f}", flush=True)
    checkpoint = torch.load(
        checkpoint_path, map_location=args.device, weights_only=True
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    with torch.inference_mode():
        codes = np.concatenate(
            [
                model.encode_codes(x.to(args.device)).cpu().numpy()
                for x in torch.from_numpy(embeddings).split(4096)
            ]
        )
    counts, suffixes = defaultdict(int), []
    for row in codes:
        prefix = tuple(row)
        suffixes.append(counts[prefix])
        counts[prefix] += 1
    codes = np.column_stack([codes, suffixes]).astype(np.int64)
    np.save(sid_path, codes)
    checkpoint["semantic_ids_hash"] = file_hash(sid_path)
    torch.save(checkpoint, checkpoint_path)
    save_json(
        sid_path.with_suffix(".json"),
        {
            "sid_sha256": file_hash(sid_path),
            "data_signature": data.signature,
            "rq_checkpoint_sha256": file_hash(checkpoint_path),
            "protocol_sha256": file_hash(args.protocol),
            "teacher_sha256": teacher_hash,
            "embeddings_sha256": file_hash(path),
        },
    )
    print("Saved:", sid_path, checkpoint_path)


if __name__ == "__main__":
    main()
