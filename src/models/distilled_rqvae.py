"""Match a frozen item-tower representation at the final or every RQ depth."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.rqvae import RQVAE


class DistilledRQVAE(RQVAE):
    def __init__(self, teacher_dim, **config):
        super().__init__(**config)
        # Share the projection across depths, so B and C have identical parameters.
        self.teacher_projection = nn.Linear(config["latent_dim"], teacher_dim)

    def forward(self, x, teacher, *, all_depths=False, behavior_weight=0.05):
        z = self.encoder(x)
        z_q, codes, vq_loss, prefixes = self.quantizer(z, return_prefixes=True)
        reconstruction = self.decoder(z_q)
        reconstruction_loss = F.mse_loss(reconstruction, x)
        predictions = prefixes if all_depths else prefixes[-1:]
        depth_losses = torch.stack([
            (1 - F.cosine_similarity(self.teacher_projection(prefix), teacher, dim=-1)).mean()
            for prefix in predictions
        ])
        # Average, rather than sum, to keep the total supervision weight fixed.
        behavior_loss = depth_losses.mean()
        return {
            "loss": reconstruction_loss + vq_loss + behavior_weight * behavior_loss,
            "reconstruction_loss": reconstruction_loss,
            "vq_loss": vq_loss,
            "behavior_loss": behavior_loss,
            "codes": codes,
        }
