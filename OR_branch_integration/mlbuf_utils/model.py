import torch
import torch.nn as nn
import torch.nn.functional as F

# layers
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

# models
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
    
class MLBuf(nn.Module):
    def __init__(self, input_dim_share, input_dim_loc, input_dim_elc, hidden_dim, num_heads, clustering_output_dim,
                 max_clusters, output_dim_bloc, output_dim_btype):
        """
        max_clusters: the number of nodes
        """
        super(MLBuf, self).__init__()

        # ---------- 1) Shared encoder, location encoder and electrical encoder--------
        self.shared_encoder = SelfAttentionBlock(input_dim_share, num_heads)
        self.loc_encoder = SelfAttentionBlock(input_dim_loc, num_heads)
        self.elc_encoder = SelfAttentionBlock(input_dim_elc, num_heads)

        # ---------- 2) Clustering module -----------------------------------
        self.clustering_module = ClusteringModule(input_dim_share + input_dim_loc, hidden_dim,
                                                  clustering_output_dim)
        # cluster_embed -> cluster assignment logits [n, K]
        self.cluster_assign_head = nn.Linear(clustering_output_dim, max_clusters)  # [n, K]

        # --------- 3) Predict buffer type & loc -----------
        # 3.1 Type: aggregate features within clusters
        type_att_in_dim = clustering_output_dim + (input_dim_elc + input_dim_share)
        # self.cluster_type_att = MaskClusterAttention(type_att_in_dim, num_heads
        #                                              )
        self.cluster_type_att = SelfAttentionBlock(type_att_in_dim, num_heads
                                                   )
        self.type_fc = DecoderMLP(
            type_att_in_dim, hidden_dim,
            output_dim_btype)

        # 3.2 Location: input = cluster_dim + input_dim_loc
        loc_att_in_dim = clustering_output_dim + (input_dim_loc + input_dim_share) + output_dim_btype
        # self.cluster_loc_att = MaskClusterAttention(loc_att_in_dim, num_heads
        #                                             )
        self.cluster_loc_att = SelfAttentionBlock(loc_att_in_dim, num_heads
                                                  )
        self.loc_fc = DecoderMLP(loc_att_in_dim, hidden_dim, output_dim_bloc)

    def forward(self, x, x_loc, x_elc, temperature=1.0):
        """
        Args:
            x: Input features (shared encoding).
            x_loc: Location-specific input features.
            x_elc: Electrical-specific input features.
        Returns:
            buffer_location: Predicted buffer locations.
            buffer_type: Predicted buffer types.
            cluster_id:          [n]  (hard assignment)
            cluster_probs:       [n, K]
            cluster_embed:       [n, cluster_dim]
        """
        n = x.size(0)
        device = x.device
        #  ---------------- Encoding ----------------------
        shared_output = self.shared_encoder(x)
        loc_out = self.loc_encoder(x_loc)
        elc_out = self.elc_encoder(x_elc)

        # Concatenated embedding
        loc_share = torch.cat([loc_out, shared_output], dim=-1)
        elc_share = torch.cat([elc_out, shared_output], dim=-1)

        # ------------- Clustering module -----------------
        cluster_embed = self.clustering_module(loc_share)  # [n, cluster_output_dim]
        # assignment
        cluster_logits = self.cluster_assign_head(cluster_embed)  # [n, K]
        cluster_probs = F.gumbel_softmax(cluster_logits, tau=temperature, hard=False)
        cluster_id = cluster_probs.argmax(dim=-1)  # [n]  choose the highest probability cluster for each sink
        # shape: [n, K], each row represents the probability of assignment
        cluster_embeds = torch.matmul(cluster_probs.transpose(0, 1), cluster_embed)  # [K, cluster_output_dim]
        cluster_elc_share = torch.matmul(cluster_probs.transpose(0, 1), elc_share)  # [K, elc_share_dim]
        cluster_loc_share = torch.matmul(cluster_probs.transpose(0, 1), loc_share)  # [K, loc_share_dim]

        # cluster-level type prediction
        c_elc_features = torch.cat([cluster_embeds, cluster_elc_share], dim=-1)
        cluster_fused_type_feat = self.cluster_type_att(c_elc_features)  # [K, ]
        cluster_type_logits = self.type_fc(cluster_fused_type_feat)  # [K, ]

        # cluster-level loc prediction
        c_loc_features = torch.cat([cluster_embeds, cluster_loc_share, cluster_type_logits], dim=-1)
        cluster_fused_loc_feat = self.cluster_loc_att(c_loc_features)  # [K, ]
        cluster_loc_pred = self.loc_fc(cluster_fused_loc_feat)  # [K, ]

        # map to each node
        buf_type_logits = torch.matmul(cluster_probs, cluster_type_logits)  # [N, num_types]
        buf_loc_pred = torch.matmul(cluster_probs, cluster_loc_pred)  # [N, loc_dim]

        buffer_type_argmax = torch.argmax(buf_type_logits, dim=-1)  # [N]
        no_buf_mask = (buffer_type_argmax == 0)
        cluster_id[no_buf_mask] = -1

        return buf_type_logits, buf_loc_pred, cluster_id, cluster_probs, cluster_embed, cluster_type_logits, cluster_loc_pred

       