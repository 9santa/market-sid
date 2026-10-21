import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.rqvae import RQVAE


class BehaviorAwareRQVAE(RQVAE):
    def __init__(
        self,
        input_dim=384,
        hidden_dim=256,
        latent_dim=128,
        num_codebooks=3,
        codebook_size=256,
        beta=0.25,
        behavior_dim=64,
    ):
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_codebooks=num_codebooks,
            codebook_size=codebook_size,
            beta=beta,
        )

        self.behavior_dim = behavior_dim

        self.behavior_head = nn.Sequential(
            nn.Linear(
                latent_dim,
                latent_dim,
            ),
            nn.GELU(),
            nn.Linear(
                latent_dim,
                behavior_dim,
            ),
        )

    def forward(
        self,
        x,
        behavior_target=None,
        behavior_mask=None,
        behavior_weight=0.0,
    ):
        z = self.encoder(x)

        z_q, codes, vq_loss = self.quantizer(z)

        reconstruction = self.decoder(z_q)

        reconstruction_loss = F.mse_loss(
            reconstruction,
            x,
        )

        behavior_loss = x.new_zeros(())

        if (
            behavior_target is not None
            and behavior_mask is not None
            and behavior_mask.any()
        ):
            behavior_loss = relational_behavior_loss(
                z_q,
                behavior_target,
                behavior_mask,
            )

        loss = reconstruction_loss + vq_loss + behavior_weight * behavior_loss

        return {
            "loss": loss,
            "reconstruction_loss": reconstruction_loss,
            "vq_loss": vq_loss,
            "behavior_loss": behavior_loss,
            "codes": codes,
            "reconstruction": reconstruction,
            "latent": z,
            "quantized": z_q,
        }

    def forward1(
        self,
        x,
        behavior_target=None,
        behavior_mask=None,
        behavior_weight=0.0,
    ):
        z = self.encoder(x)

        z_q, codes, vq_loss = self.quantizer(z)

        reconstruction = self.decoder(z_q)

        reconstruction_loss = F.mse_loss(reconstruction, x)

        behavior_loss = x.new_zeros(())

        if (
            behavior_target is not None
            and behavior_mask is not None
            and behavior_mask.any()
        ):
            prediction = self.behavior_head(z_q[behavior_mask])

            prediction = F.normalize(prediction, dim=-1)

            target = behavior_target[behavior_mask]

            target = F.normalize(target, dim=-1)

            behavior_loss = 1.0 - F.cosine_similarity(prediction, target, dim=-1).mean()

        loss = reconstruction_loss + vq_loss + behavior_loss * behavior_weight

        return {
            "loss": loss,
            "reconstruction_loss": reconstruction_loss,
            "vq_loss": vq_loss,
            "behavior_loss": behavior_loss,
            "codes": codes,
            "reconstruction": reconstruction,
            "latent": z,
            "quantized": z_q,
        }


def relational_behavior_loss(
    quantized,
    behavior_target,
    behavior_mask,
):
    q = quantized[behavior_mask]

    b = behavior_target[behavior_mask]

    if len(q) < 2:
        return quantized.new_zeros(())

    q = F.normalize(q, dim=-1)

    b = F.normalize(b, dim=-1)

    q_sim = q @ q.T
    b_sim = b @ b.T

    n = len(q)

    mask = ~torch.eye(
        n,
        dtype=torch.bool,
        device=q.device,
    )

    return F.mse_loss(
        q_sim[mask],
        b_sim[mask],
    )
