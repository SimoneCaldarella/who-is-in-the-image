import torch
from torch import nn

class NoiseInjection(nn.Module):

    def __init__(self, noise_factor: float  = 2, position: str = "pre_connector"):
        super().__init__()
        self.alpha = noise_factor
        self.position = position
        self.enabled = True

    def forward(self, feats):

        if self.enabled:
            # Init noise and normalize
            noise = torch.randn_like(feats).to(feats.dtype)
            feats_norm = feats.norm(dim=-1, keepdim=True)
            feats /= feats.norm(dim=-1, keepdim=True)
            noise /= noise.norm(dim=-1, keepdim=True)

            # Compose
            feats = feats + self.alpha * noise

            # Rescale
            feats = feats * feats_norm / feats.norm(dim=-1, keepdim=True)

        return feats