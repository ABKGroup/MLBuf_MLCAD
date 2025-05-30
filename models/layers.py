import torch.nn as nn
import torch


class SelfAttentionBlock(nn.Module):
    def __init__(self, input_dim, num_heads):
        super(SelfAttentionBlock, self).__init__()
        self.self_attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Linear(input_dim, input_dim)
        )
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, x):
        x = x.unsqueeze(0)
        attn_output, _ = self.self_attention(x, x, x)
        x = self.norm(x + attn_output)
        fc_output = self.fc(x)
        x = self.norm(x + fc_output)
        x = x.squeeze(0)
        return x


class MaskClusterAttention(nn.Module):
    """
    multi-head self-attention，
    mask features from different cluster
    """

    def __init__(self, input_dim, num_heads):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(embed_dim=input_dim,
                                                    num_heads=num_heads,
                                                    batch_first=True)

        self.norm = nn.LayerNorm(input_dim)

    def build_attn_mask(self, cluster_id):
        """
        cluster_id: shape = [N], ignore those cluster_id[i] == -1

        return attn_mask: [N, N], bool。
        """
        N = cluster_id.size(0)
        mask = torch.zeros(N, N, dtype=torch.bool, device=cluster_id.device)
        for i in range(N):
            ci = cluster_id[i]
            if ci < 0:
                mask[i, :] = True
            else:
                for j in range(N):
                    cj = cluster_id[j]
                    if cj < 0 or cj != ci:
                        mask[i, j] = True
        return mask

    def forward(self, x, cluster_id):
        """
        x: shape [N, D],  regard x as a sequence when batch_size=1
        cluster_id: shape [N], cluster_id[i] represent the clusterID for node i (0..K-1 or -1)

        return: shape [N, D], perform self-attention in each cluster
        """
        # 1)  [1, N, D]
        x_in = x.unsqueeze(0)

        # 2) attn_mask => [N, N]
        mask = self.build_attn_mask(cluster_id)

        # 3) MultiHeadAttention
        #    The value will be masked when attn_mask is True
        attn_out, _ = self.self_attention(x_in, x_in, x_in, attn_mask=mask)
        # attn_out: [1, N, D]
        x_in_normed = self.norm(x_in + attn_out)

        # 4) [N, D]
        out = x_in_normed.squeeze(0)
        return out


class ClusteringModule(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ClusteringModule, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.fc(x)


class DecoderMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DecoderMLP, self).__init__()
        self.out_features = output_dim
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.fc(x)
