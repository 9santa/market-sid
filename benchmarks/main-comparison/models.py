"""Small adapters for RecBole models and a TIGER-style T5 baseline."""

import numpy as np
import scipy.sparse as sp
import torch
from torch import nn


class ItemKNN:
    def __init__(self, similarities):
        self.similarities = similarities.tocsr()

    @classmethod
    def fit(cls, data, k=100, block_size=128, shrink=0.0):
        interactions = data.inter_matrix("csc")
        norms = np.sqrt(np.asarray(interactions.power(2).sum(axis=0)).ravel())
        rows, cols, values = [], [], []
        # Top-k per candidate column, as in RecBole ItemKNN; never materialize N×N.
        for start in range(1, data.num_items + 1, block_size):
            stop = min(start + block_size, data.num_items + 1)
            similarities = (interactions[:, start:stop].T @ interactions).toarray()
            denominator = norms[start:stop, None] * norms[None, :] + shrink
            np.divide(similarities, denominator, out=similarities, where=denominator > 0)
            similarities[np.arange(stop - start), np.arange(start, stop)] = 0
            for offset, weights in enumerate(similarities):
                positive = np.flatnonzero(weights > 0)
                # Only sort nonzero neighbors; item IDs break similarity ties.
                order = np.lexsort((positive, -weights[positive]))[:k]
                neighbors = positive[order]
                rows.extend(neighbors)
                cols.extend([start + offset] * len(neighbors))
                values.extend(weights[neighbors])
            if start == 1 or (start - 1) // block_size % 100 == 0:
                print(f"ItemKNN: {stop - 1:,}/{data.num_items:,} items", flush=True)
        return cls(sp.csr_matrix((np.asarray(values, dtype=np.float32), (rows, cols)),
                                 shape=(data.num_items + 1, data.num_items + 1)))

    def score(self, records, num_items):
        rows, cols = [], []
        for i, record in enumerate(records):
            items = np.unique(record["history"])
            rows.extend([i] * len(items))
            cols.extend(items)
        history = sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)),
                                shape=(len(records), num_items + 1))
        return (history @ self.similarities).toarray()


def recbole_config(config, device):
    return {
        "USER_ID_FIELD": "user_id", "ITEM_ID_FIELD": "item_id", "NEG_PREFIX": "neg_",
        "LIST_SUFFIX": "_list", "ITEM_LIST_LENGTH_FIELD": "item_length",
        "MAX_ITEM_LIST_LENGTH": config["max_len"], "device": device,
        "embedding_size": config["d_model"], "hidden_size": config["d_model"],
        "num_layers": config["n_layers"], "n_layers": config["n_layers"],
        "dropout_prob": config["dropout"], "loss_type": "CE",
        "reg_weight": config["reg_weight"], "require_pow": False,
    }


def make_lightgcn(config, data, device):
    from recbole.model.general_recommender.lightgcn import LightGCN

    class CompatibleLightGCN(LightGCN):
        def get_norm_adj_mat(self):
            # Same binary symmetric normalization, without SciPy's removed _update.
            interactions = self.interaction_matrix.tocsr()
            adjacency = sp.bmat([[None, interactions], [interactions.T, None]], format="csr")
            degree = np.asarray(adjacency.sum(axis=1)).ravel()
            inverse = np.power(degree + 1e-7, -0.5)
            normalized = (sp.diags(inverse) @ adjacency @ sp.diags(inverse)).tocoo()
            indices = torch.from_numpy(np.stack([normalized.row, normalized.col]).astype(np.int64))
            return torch.sparse_coo_tensor(indices, torch.from_numpy(normalized.data),
                                          normalized.shape, check_invariants=True).coalesce()

    return CompatibleLightGCN(recbole_config(config, device), data).to(device)


class Tiger(nn.Module):
    """Flatten SID histories into T5 tokens; keep full-vocabulary probabilities."""

    def __init__(self, item_sids, config):
        super().__init__()
        from transformers import T5Config, T5ForConditionalGeneration

        self.num_levels = item_sids.shape[1]
        self.vocab_sizes = config["vocab_sizes"]
        self.user_buckets = config["user_buckets"]
        offsets = np.cumsum([2] + self.vocab_sizes[:-1])
        self.register_buffer("offsets", torch.tensor(offsets), persistent=False)
        self.register_buffer("item_sids", torch.as_tensor(item_sids, dtype=torch.long), persistent=False)
        self.user_offset = 2 + sum(self.vocab_sizes)
        self.t5 = T5ForConditionalGeneration(T5Config(
            vocab_size=self.user_offset + self.user_buckets,
            d_model=config["d_model"], d_kv=config["d_kv"], d_ff=config["d_ff"],
            num_layers=config["n_layers"], num_decoder_layers=config["decoder_layers"],
            num_heads=config["n_heads"], dropout_rate=config["dropout"],
            feed_forward_proj="relu", pad_token_id=0, decoder_start_token_id=0,
            eos_token_id=None, use_cache=False, tie_word_embeddings=True,
        ))

    def encoder_inputs(self, users, histories):
        mask = histories.ne(0).unsqueeze(-1).expand(-1, -1, self.num_levels).flatten(1)
        tokens = (self.item_sids[histories] + self.offsets).flatten(1).masked_fill(~mask, 0)
        if self.user_buckets:
            # Stable hash of the existing indexed user; never use Python's salted hash.
            user_tokens = ((users * 2654435761) % 2147483647) % self.user_buckets + self.user_offset
            tokens = torch.cat([user_tokens[:, None], tokens], dim=1)
        return tokens, tokens.ne(0)

    def calculate_loss(self, batch):
        tokens, mask = self.encoder_inputs(batch["user_id"], batch["item_id_list"])
        labels = self.item_sids[batch["item_id"]] + self.offsets
        # Fixed-length four-code generation: no learned EOS decision.
        return self.t5(input_ids=tokens, attention_mask=mask, labels=labels).loss

    def encode_history(self, inputs):
        tokens, mask = self.encoder_inputs(inputs[:, 0], inputs[:, 1:])
        memory = self.t5.encoder(input_ids=tokens, attention_mask=mask).last_hidden_state
        return memory, ~mask

    def next_token_log_probs(self, memory, padding_mask, prefix):
        from transformers.modeling_outputs import BaseModelOutput

        level = prefix.shape[1]
        previous = prefix + self.offsets[:level]
        start = torch.zeros((len(prefix), 1), dtype=torch.long, device=prefix.device)
        logits = self.t5(
            encoder_outputs=BaseModelOutput(last_hidden_state=memory),
            attention_mask=~padding_mask,
            decoder_input_ids=torch.cat([start, previous], dim=1),
        ).logits[:, -1].float()
        # Normalize before slicing: invalid levels/user/PAD tokens still carry mass.
        log_probs = logits.log_softmax(-1)
        offset = int(self.offsets[level])
        return log_probs[:, offset:offset + self.vocab_sizes[level]]


def make_model(name, config, data, device, item_sids=None):
    if name == "gru4rec":
        from recbole.model.sequential_recommender.gru4rec import GRU4Rec
        return GRU4Rec(recbole_config(config, device), data).to(device)
    if name == "lightgcn":
        return make_lightgcn(config, data, device)
    if name == "tiger":
        return Tiger(item_sids, config).to(device)
    raise ValueError(name)
