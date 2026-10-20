import torch


class TokenPruning(torch.nn.Module):

    def __init__(self, sparsity: float = 0.7, position: str = "pre_connector"):
        super().__init__()
        self.sparsity = sparsity
        self.enabled = True
        self.position = position

    def forward(self, feats):


        if self.enabled:
            # smol vlm example
            # indices = torch.Size([13, 819, 768])
            # zero_mask = torch.Size([13, 1024, 768])
            # Zeroed out feats 10647      

            B = feats.shape[0]  # Batch size, number of features, feature dim
            N = feats.shape[1]

            k = int(self.sparsity * N)
            random_indices = torch.stack([
                torch.randperm(N)[:k] for _ in range(B)
            ]).to(feats.device)
            if feats.dim() == 2:  # [B, N]
                zero_mask = torch.ones_like(feats)  # [B, N]
                indices = random_indices  # [B, k]
                zero_mask.scatter_(1, indices, 0)
            else:  # [B, N, D]
                zero_mask = torch.ones_like(feats)  # [B, N, D]
                indices = random_indices.unsqueeze(-1).expand(-1, -1, feats.size(-1))
                zero_mask.scatter_(-2, indices, 0)

            feats = feats * zero_mask  # [B, N, D])

            # print(f"Zeroed out feats {((feats == 0).sum()/feats.numel())*100:.2f}%")

            
        return feats