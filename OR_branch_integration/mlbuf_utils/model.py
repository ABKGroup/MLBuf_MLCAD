import sys
sys.path.append('/home/fetzfs_projects/MLBuf/flows/OR_branch_integration/mlbuf_utils')
import torch
import torch.nn as nn
from layers import *
import torch.nn.functional as F


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

        # hard assignment (argmax)
        # # cluster_id = cluster_probs.argmax(dim=-1).detach()  # [n]  choose the highest probability cluster for each sink
        # cluster_id = cluster_probs.argmax(dim=-1)  # [n]  choose the highest probability cluster for each sink
        #
        # # -------------- aggregate features within each cluster -> predict type -> predict locations
        #
        # buffer_type_logits = torch.zeros(n, self.type_fc.out_features, device=device)
        # buffer_loc_pred = torch.zeros(n, self.loc_fc.out_features, device=device)
        # unique_ids = cluster_id.unique()  # some clusters may do not contain sinks
        # for cid in unique_ids:
        #     mask = (cluster_id == cid)
        #     idx = mask.nonzero(as_tuple=True)[0]  # index of the cluster [Nc]
        #     if idx.numel() == 0:
        #         continue
        #     # only do attention + mean if the cluster has at least 1 node
        #
        #     # feature aggregation within clusters
        #     c_cluster_embed = cluster_embed[idx]  # [Nc, cluster_dim]
        #     c_elc_out = elc_share[idx]  # [Nc, input_dim_elc+input_dim_share]
        #     c_loc_out = loc_share[idx]  # [Nc, input_dim_loc+input_dim_share]
        #
        #     c_elc_features = torch.cat([c_cluster_embed, c_elc_out], dim=-1)
        #     # self-attention
        #     c_type_features_att = self.cluster_type_att(c_elc_features)
        #     # pooling
        #     cluster_fused_type_feat = c_type_features_att.mean(dim=0, keepdim=True)  # [1, D]
        #
        #     # Predict type
        #     cluster_type_logits = self.type_fc(cluster_fused_type_feat)  # [1, output_dim_btype]
        #     buffer_type_logits[idx] = cluster_type_logits  # map to each sink
        #
        #     # ---- identify if there is a buffer = 0 => no buffer ----
        #     # argmax
        #     type_pred = cluster_type_logits.argmax(dim=-1)  # [1]
        #     if type_pred.item() == 0:
        #         # no buffer => cluster_id = -1
        #         cluster_id[idx] = -1
        #         # set location to -1
        #         buffer_loc_pred[idx] = -1
        #     else:
        #
        #         c_loc_features = torch.cat([c_cluster_embed, c_loc_out, buffer_type_logits[idx]], dim=-1)
        #         c_loc_features_att = self.cluster_loc_att(c_loc_features)
        #         cluster_fused_loc_feat = c_loc_features_att.mean(dim=0, keepdim=True)  # [1, D]
        #         # # loc_in = torch.cat([cluster_fused_loc_feat, cluster_type_logits], dim=-1)  # [1, D+output_dim_btype]
        #         cluster_loc_pred = self.loc_fc(cluster_fused_loc_feat)  # [1, output_dim_bloc]
        #         buffer_loc_pred[idx] = cluster_loc_pred
        # return buffer_type_logits, buffer_loc_pred, cluster_id, cluster_probs, cluster_embed

    # def forward(self, x, x_loc, x_elc, temperature=1.0):
    #     """
    #     Args:
    #         x: Input features (shared encoding).
    #         x_loc: Location-specific input features.
    #         x_elc: Electrical-specific input features.
    #     Returns:
    #         buffer_location: Predicted buffer locations.
    #         buffer_type: Predicted buffer types.
    #         cluster_id:          [n]  (hard assignment)
    #         cluster_probs:       [n, K]
    #         cluster_embed:       [n, cluster_dim]
    #     """
    #     n = x.size(0)
    #     device = x.device
    #     #  ---------------- Encoding ----------------------
    #     shared_output = self.shared_encoder(x)
    #     loc_out = self.loc_encoder(x_loc)
    #     elc_out = self.elc_encoder(x_elc)
    #
    #     # Concatenated embedding
    #     loc_share = torch.cat([loc_out, shared_output], dim=-1)
    #     elc_share = torch.cat([elc_out, shared_output], dim=-1)
    #
    #     # ------------- Clustering module -----------------
    #     cluster_embed = self.clustering_module(loc_share)  # [n, cluster_output_dim]
    #     # assignment
    #     cluster_logits = self.cluster_assign_head(cluster_embed)  # [n, K]
    #     cluster_probs = F.gumbel_softmax(cluster_logits, tau=temperature, hard=False)
    #     # shape: [n, K], each row represents the probability of assignment
    #     # hard assignment (argmax)
    #     # cluster_id = cluster_probs.argmax(dim=-1).detach()  # [n]  choose the highest probability cluster for each sink
    #     cluster_id = cluster_probs.argmax(dim=-1)  # [n]  choose the highest probability cluster for each sink
    #
    #     # -------------- aggregate features within each cluster -> predict type -> predict locations
    #     # # ======= predict type ===========
    #     # combined_feat_for_type = torch.cat([cluster_embed, elc_share], dim=-1)  # shape [N, ?]
    #     # type_att_out = self.cluster_type_att(combined_feat_for_type, cluster_id)  # attention mask
    #     # # scatter mean -> [K, dim_for_type_att]
    #     # cluster_type_feat = scatter_mean(type_att_out, cluster_id, dim=0)
    #     # # MLP => cluster_type_logits
    #     # cluster_type_logits = self.type_fc(cluster_type_feat)  # [K, output_dim_btype]
    #     # # map to each node
    #     # buffer_type_logits = cluster_type_logits[cluster_id]  # [N, output_dim_btype]
    #     #
    #     # type_pred = buffer_type_logits.argmax(dim=-1)  # [N]
    #     # no_buf_mask = (type_pred == 0)
    #     # # cluster_id[no_buf_mask] = -1  # no buffer
    #     # modified_cluster_id = torch.where(
    #     #     no_buf_mask,
    #     #     torch.full_like(cluster_id, -1),  # if True => -1
    #     #     cluster_id  # else => original
    #     # )
    #     #
    #     # # ======= predict loc ===========
    #     # combined_feat_for_loc = torch.cat([cluster_embed, loc_share, buffer_type_logits], dim=-1)
    #     # # mask attention
    #     # loc_att_out = self.cluster_loc_att(combined_feat_for_loc, cluster_id)
    #     # valid_mask = (modified_cluster_id >= 0)
    #     # valid_cluster_id = modified_cluster_id[valid_mask]  # shape [M]
    #     # valid_loc_att_out = loc_att_out[valid_mask]
    #     # cluster_loc_feat = scatter_mean(valid_loc_att_out, valid_cluster_id, dim=0)
    #     #
    #     # cluster_loc_logits = self.loc_fc(cluster_loc_feat)  # [K, bloc_dim]
    #     # buffer_loc_pred = cluster_loc_logits[modified_cluster_id]  # [N, bloc_dim]
    #     # buffer_loc_pred[no_buf_mask] = -1
    #     # return buffer_type_logits, buffer_loc_pred, cluster_id, cluster_probs, cluster_embed
    #
    #     buffer_type_logits = torch.zeros(n, self.type_fc.out_features, device=device)
    #     buffer_loc_pred = torch.zeros(n, self.loc_fc.out_features, device=device)
    #     unique_ids = cluster_id.unique()  # some clusters may do not contain sinks
    #     for cid in unique_ids:
    #         mask = (cluster_id == cid)
    #         idx = mask.nonzero(as_tuple=True)[0]  # index of the cluster [Nc]
    #         if idx.numel() == 0:
    #             continue
    #         # only do attention + mean if the cluster has at least 1 node
    #
    #         # feature aggregation within clusters
    #         c_cluster_embed = cluster_embed[idx]  # [Nc, cluster_dim]
    #         c_elc_out = elc_share[idx]  # [Nc, input_dim_elc+input_dim_share]
    #         c_loc_out = loc_share[idx]  # [Nc, input_dim_loc+input_dim_share]
    #
    #         c_elc_features = torch.cat([c_cluster_embed, c_elc_out], dim=-1)
    #         # self-attention
    #         c_type_features_att = self.cluster_type_att(c_elc_features)
    #         # pooling
    #         cluster_fused_type_feat = c_type_features_att.mean(dim=0, keepdim=True)  # [1, D]
    #
    #         # # combine global location features
    #         # Nc = c_elc_out.size(0)  # the number of sinks in the cluster
    #         # global_loc_context_expanded = global_loc_context.expand(Nc, -1)
    #         # c_elc_features_plus_global = torch.cat([c_cluster_embed, c_elc_out, global_loc_context_expanded], dim=-1)
    #         #
    #         # # self-attention
    #         # c_type_features_att = self.cluster_type_att(c_elc_features_plus_global)
    #         # # pooling
    #         # cluster_fused_type_feat = c_type_features_att.mean(dim=0, keepdim=True)
    #
    #         # Predict type
    #         cluster_type_logits = self.type_fc(cluster_fused_type_feat)  # [1, output_dim_btype]
    #         buffer_type_logits[idx] = cluster_type_logits  # map to each sink
    #
    #         # ---- identify if there is a buffer = 0 => no buffer ----
    #         # argmax
    #         type_pred = cluster_type_logits.argmax(dim=-1)  # [1]
    #         if type_pred.item() == 0:
    #             # no buffer => cluster_id = -1
    #             cluster_id[idx] = -1
    #             # set location to -1
    #             buffer_loc_pred[idx] = -1
    #         else:
    #             # Predict location
    #             # Nc = c_loc_out.size(0)
    #             # global_loc_context_expanded = global_loc_context.expand(Nc, -1)
    #             # c_loc_features = torch.cat([
    #             #     c_cluster_embed,
    #             #     c_loc_out,
    #             #     buffer_type_logits[idx],
    #             #     global_loc_context_expanded  # global location distribution
    #             # ], dim=-1)
    #
    #             # c_loc_features_att = self.cluster_loc_att(c_loc_features)
    #             # cluster_fused_loc_feat = c_loc_features_att.mean(dim=0, keepdim=True)
    #             # cluster_loc_pred = self.loc_fc(cluster_fused_loc_feat)
    #             # buffer_loc_pred[idx] = cluster_loc_pred
    #
    #             c_loc_features = torch.cat([c_cluster_embed, c_loc_out, buffer_type_logits[idx]], dim=-1)
    #             c_loc_features_att = self.cluster_loc_att(c_loc_features)
    #             cluster_fused_loc_feat = c_loc_features_att.mean(dim=0, keepdim=True)  # [1, D]
    #             # # loc_in = torch.cat([cluster_fused_loc_feat, cluster_type_logits], dim=-1)  # [1, D+output_dim_btype]
    #             cluster_loc_pred = self.loc_fc(cluster_fused_loc_feat)  # [1, output_dim_bloc]
    #             buffer_loc_pred[idx] = cluster_loc_pred
    #     return buffer_type_logits, buffer_loc_pred, cluster_id, cluster_probs, cluster_embed
