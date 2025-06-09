import torch
import torch.nn.functional as F
import torch.nn as nn
import utils.util as util


def cluster_loss(embeddings, ground_truth, weight_pos=1.0,
                 weight_neg=1.0):
    """
    Computes the cluster loss for the given embeddings and ground truth cluster labels.

    Args:
        embeddings (Tensor): Embeddings of the sinks, shape [num_sinks, embedding_dim].
        ground_truth (Tensor): Ground truth cluster labels, shape [num_sinks].
                                Each sink is assigned a cluster ID. Sinks in the same cluster have the same ID.

    Returns:
        loss (Tensor): Computed cluster loss.
    """
    device = embeddings.device
    # remove driver
    sink_mask = (ground_truth != -1)

    embeddings = embeddings[sink_mask]
    ground_truth = ground_truth[sink_mask]

    num_sinks, embedding_dim = embeddings.shape
    if num_sinks <= 1:
        # No real pairwise clustering needed; just return 0
        return embeddings.new_tensor(0.0, requires_grad=True)

    # Normalize embeddings to unit length (for cosine similarity)
    embeddings_normalized = F.normalize(embeddings, p=2, dim=-1)  # Shape: [num_sinks, embedding_dim]

    # Compute pairwise cosine similarity (similarity matrix)
    similarity_matrix = torch.matmul(embeddings_normalized,
                                     embeddings_normalized.T)  # [num_sinks, num_sinks]

    # Convert similarity to range [0, 1] (cosine similarity is naturally in [-1, 1])
    D = 0.5 * (similarity_matrix + 1)  # Shape: [num_sinks, num_sinks]
    D = torch.clamp(D, min=1e-7, max=1 - 1e-7)

    # Create ground truth mask for pairs
    y = (ground_truth.unsqueeze(1) == ground_truth.unsqueeze(0)).float()  # Shape: [num_sinks, num_sinks]

    # Compute loss for pairs
    positive_loss = -torch.log(D + 1e-8) * y  # Loss for pairs in the same cluster (y = 1)
    negative_loss = -torch.log(1 - D + 1e-8) * (1 - y)  # Loss for pairs in different clusters (y = 0)

    # ---- refine
    mask_no_self = (1 - torch.eye(num_sinks, device=device))
    # masked pos/neg
    pos_vals = positive_loss * mask_no_self
    neg_vals = negative_loss * mask_no_self

    # average among actual pos/neg pairs
    # count how many pos pairs, neg pairs (excluding diagonal)
    num_pos = (y * mask_no_self).sum()
    num_neg = ((1 - y) * mask_no_self).sum()

    if num_pos > 0:
        loss_pos_mean = pos_vals.sum() / num_pos
    else:
        loss_pos_mean = torch.tensor(0.0, device=device, requires_grad=True)

    if num_neg > 0:
        loss_neg_mean = neg_vals.sum() / num_neg
    else:
        loss_neg_mean = torch.tensor(0.0, device=device, requires_grad=True)

    # 7) combine with weights
    total_loss = weight_pos * loss_pos_mean + weight_neg * loss_neg_mean


    return total_loss


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma  # reduce the weight of minority class
        self.alpha = alpha  # class weigth

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce_loss)  # Calculate the confidence of the prediction
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


def location_loss(type_gt, loc_pred, loc_gt):
    device = loc_pred.device
    # Only calculate loss for those type_gt != 0
    loc_mask = (type_gt != 0)
    if loc_mask.any():
        loc_pred_selected = loc_pred[loc_mask]  # shape [?, output_dim_bloc]
        loc_gt_selected = loc_gt[loc_mask]  # shape [?, output_dim_bloc]
        loss_loc = F.mse_loss(loc_pred_selected, loc_gt_selected)
    else:
        loss_loc = torch.tensor(0.0, device=device)
    return loss_loc


def compute_class_weights(labels, num_classes, alpha_min=0.1, alpha_max=20.0, eps=1e-8):
    """
    Compute balanced class weights for imbalanced classification.

    Args:
    - labels (Tensor): Ground-truth labels, shape (N,)
    - num_classes (int): Number of buffer types (e.g., 2)
    - alpha_min (float): Minimum weight per class (to avoid zero)
    - alpha_max (float): Maximum weight per class (to avoid explosion)
    - eps (float): Small value to avoid division by zero

    Returns:
    - class_weights (Tensor): Smoothed class weights, shape (num_classes,)
    """

    # Count occurrences of each class
    class_counts = torch.bincount(labels, minlength=num_classes).float()

    # Compute smoothed weight: total samples / (num_classes * count per class)
    total_samples = class_counts.sum()
    class_weights = total_samples / (num_classes * (class_counts + eps))

    # Apply smoothing: clip between alpha_min and alpha_max
    class_weights = torch.clamp(class_weights, min=alpha_min, max=alpha_max)

    # Normalize weights to sum to 1 (optional, depends on how you use them)
    # class_weights /= class_weights.sum() + eps

    return class_weights


def build_buf_cap_tensor(buf_info_df, num_types, device='cpu'):
    max_cap_per_type = torch.zeros(num_types, device=device)
    input_cap_per_type = torch.zeros(num_types, device=device)
    for idx, row in buf_info_df.iterrows():
        max_cap_per_type[idx] = torch.tensor(row["max_capacitance"] * 1e14, device=device)
        input_cap_per_type[idx] = torch.tensor(row["capacitance"] * 1e14, device=device)
    return max_cap_per_type, input_cap_per_type


def compute_cap_wirelength_penalty(features,
                                   cluster_buf_type, cluster_buf_loc, cluster_probs, buf_info_df, buf_type_map,
                                   max_wirelength):
    """
    soft assignment
    compute output cap of driver and buffers
    compute max wirelength between driver (buffers) and their children
    If cluster_type_pred[k] = 0 => no buffer => driver directly drive sinks in cluster k
    If cluster_type_pred[k] > 0 => has buffer =>  driver->buffer,  buffer->sinks

    Args:
      features:     [n, dim]  (the model's x input)
      cluster_buf_loc: [K, loc_out_dim],  # the predicted buf location for each cluster
      cluster_buf_type:     [K, btype_dim],    # predicted type for each cluster
      cluster_probs : [n, K],
      buf_info_df: buf info [buf type, area, input cap, etc.]
      max_wirelength = max_wirelength_constraints

    """
    device = features.device
    N, K = cluster_probs.shape
    wire_cap = 8.88758 * 1e3
    # original value 8.88758e-11 => scale
    dbu = 2000

    # 0) build a max_cap lookup
    #    suppose you have "num_types = cluster_type_logits.shape[1]"
    max_cap_per_type, in_cap_per_type = build_buf_cap_tensor(buf_info_df, 6, device=device)

    # ---- (A) penalty for "with-buffer" clusters in vector form ----
    cluster_type_pred = cluster_buf_type.argmax(dim=-1)  # [K]
    has_buf_mask = (cluster_type_pred > 0).float()  # [K], 1 or 0

    # compute wire_k => sum over i p_{i,k} * dist(i,k)
    dist_mat = manhattan_dist_matrix(features[:, :2], cluster_buf_loc)  # [N,K]
    wire_k = (cluster_probs * dist_mat).sum(dim=0)  # [K]
    wire_k = wire_k * has_buf_mask  # zero out "no buffer" clusters
    wire_k_meter = 0.9 * (wire_k / (dbu * 1e+6))  # meter
    wire_excess = F.relu(wire_k_meter - max_wirelength)  # [K]

    # compute cap_k => sum_i p_{i,k}*sink_elc[i] + wire_k*factor
    # print("test1: ", cluster_probs.shape, features.shape)
    child_cap_k = (cluster_probs * features[:, 2].unsqueeze(-1)).sum(dim=0)  # [K]
    cap_k = child_cap_k + wire_k_meter * wire_cap
    cap_k = cap_k * has_buf_mask

    # look up each cluster's buffer max_cap
    cluster_max_cap = max_cap_per_type[cluster_type_pred]  # [K]
    # if type=0 => cluster_max_cap=0 => or you can skip those with has_buf_mask
    cluster_max_cap = cluster_max_cap * has_buf_mask  # so no-buffer => 0 => no penalty
    cap_excess = F.relu(cap_k - cluster_max_cap)  # [K]
    # penalty = wire_excess.sum() + cap_excess.sum()

    # ---- （B）update driver wirelength and drvier output cap
    driver_xy = torch.tensor([0, 0]).to(device)
    cluster_frac_k = cluster_probs.sum(dim=0)  # [K], sum of p_{i,k} across i
    sink_driver_dist = torch.abs(features[:, :2] - driver_xy).sum(dim=-1)  # [N]
    dist_driver_sink_k = (sink_driver_dist.unsqueeze(-1) * cluster_probs).sum(dim=0)  # [K]
    dist_driver_buf = manhattan_dist_matrix(driver_xy.unsqueeze(0), cluster_buf_loc)  # => shape [1,K]
    dist_driver_buf = dist_driver_buf[0]  # [K]

    # now define final driver-wire
    # ctype=0 => driver->sink => dist_driver_sink_k
    # ctype>0 => driver->buffer => dist_driver_buf * cluster_frac_k
    ctype_mask0 = (cluster_type_pred == 0).float()
    ctype_mask1 = has_buf_mask
    driver_wire_k = ctype_mask0 * dist_driver_sink_k + ctype_mask1 * (dist_driver_buf * cluster_frac_k)
    driver_wire_meter = 0.9 * (driver_wire_k / (dbu * 1e+6))  # meter

    # driver cap:
    # type=0 => sum_i p_{i,k} sinkInCap[i] + driver_wire_meter[k]*wire_cap
    # type>0 => in_cap_per_type[type_k] + driver_wire_meter[k]* wire_cap
    # gather in_cap for buffer type>0
    buf_incap_per_cluster = in_cap_per_type[cluster_type_pred]  # [K], 0 if type=0 => can mask
    # final inCap_k
    inCap_driver_k = ctype_mask0 * ((cluster_probs * features[:, 2].unsqueeze(-1)).sum(dim=0)) \
                     + ctype_mask1 * (buf_incap_per_cluster)
    driver_output_cap = inCap_driver_k + driver_wire_meter * wire_cap

    return driver_wire_meter.sum(), driver_output_cap.sum(), wire_excess.sum(), cap_excess.sum()


def manhattan_dist_matrix(sink_coords, cluster_loc):
    """
    sink_coords: [N,2]
    cluster_loc: [K,2]
    returns dist_mat: [N,K], dist_mat[i,k] = manhattan distance
    """
    # sink_coords => shape [N,1,2]
    # cluster_loc => shape [1,K,2]
    # => broadcast => [N,K,2] => sum along last dim
    # to avoid loops
    sink_expand = sink_coords.unsqueeze(1)  # [N,1,2]
    clus_expand = cluster_loc.unsqueeze(0)  # [1,K,2]
    diff = torch.abs(sink_expand - clus_expand)  # [N,K,2]
    dist_mat = diff.sum(dim=-1)  # [N,K]
    return dist_mat
