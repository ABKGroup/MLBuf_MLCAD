import sys
sys.path.append('/home/fetzfs_projects/MLBuf/flows/OR_branch_integration/mlbuf_utils')
import torch
from collections import deque
import pandas as pd
import numpy as np


def adjust_model_output(features, loc_features, elc_features,
                        buffer_locations, buffer_types, cluster_id, buf_info_df, buf_type_map):
    """
    Generate new sequences for next-level prediction:
      - if sinks are driven by a buffer (cluster_id != -1)，replace these sinks to the buffer
      - else (cluster_id == -1), retain sinks

    Args:
      features:     [n, dim]  (the model's x input)
      loc_features: [n, 3]
      elc_features: [n, 5]
      buffer_locations: [n, loc_out_dim],  # the predicted location for each node
      buffer_types:     [n, btype_dim],    # predicted type for each node
      cluster_id : [n], each element is the new index of the buffer in the next-level array or -1
      buf_info_df: buf info [buf type, area, input cap, etc.]

    Returns:
        x_next, x_loc_next, x_elc_next for next-level prediction
        update driver features including["Output slew", "Output Cap"] --> estimate wirelength
        update sink features including ["Input Slew"]
        update buffer features including ["Input Slew", "Input Cap", "Output Cap=-1", "Max Output Cap", "Fanout"]
    """

    device = features.device
    # step 0: read buffer info
    # step 1: separate "no buffer" vs "with buffer"

    # no_buf_mask = (cluster_id == -1)
    # yes_buf_mask = (cluster_id != -1)
    buffer_type_argmax = torch.argmax(buffer_types, dim=-1)  # [N]
    buffer_type_argmax[0] = 0
    cluster_id[0] = -1  # driver node
    no_buf_mask = (buffer_type_argmax == 0)
    yes_buf_mask = (buffer_type_argmax > 0)

    # 2) keep no_buf_mask sinks as-is
    x_no = features[no_buf_mask]
    loc_no = loc_features[no_buf_mask]
    elc_no = elc_features[no_buf_mask]

    # 3) group the "with buffer" ones by cluster_id => each cluster id => 1 buffer node
    #    for simplicity, for each cluster_id in unique(), we average buffer_location & type
    new_buf_features_list = []
    new_buf_loc_list = []
    new_buf_elc_list = []
    new_buf_features_list_buf = []

    # selected_features = ["Node Type", "X", "Y", "Manhattan Distance", "Input Slew", "Output Slew",
    #                      "Input Cap", "Output Cap", "Max Output Cap","Resistance", "Fanout"]

    used_ids = cluster_id[yes_buf_mask].unique()

    for cid in used_ids:
        mask_c = (cluster_id == cid)
        #  cluster's buffer loc & type => already have buffer_locations[mask_c], buffer_types[mask_c]
        #  typically these are same for that cluster, so we can just pick the first
        buf_xy = buffer_locations[mask_c][0]  # shape [1, 2]
        # buf_type = buffer_types[mask_c][0]  # shape [1, btype_dim]
        buf_type_id = buffer_type_argmax[mask_c][0].item()  # shape [1, 1]

        # Build the new "buffer node" feature vector:
        # - Node Type = "Sink" => regard it as a sink for the next-level prediction.
        # - X, Y => predicted from buf_xy
        # - Manhattan Dist => dist from the driver (0,0)
        # - In/Out Slew, In/Out Cap
        # - MaxOutCap, Fanout =>

        # save the node type as Buffer
        node_type_val_buf = torch.tensor([1, 0, 0], device=device) 
        # node_type_val = x_no[1][0:2]  # same with "Sink"
        node_type_val = torch.tensor([0, 0, 1], device=device) 
        x_val, y_val = buf_xy.squeeze(0).cpu().numpy()
        driver_x, driver_y = 0, 0
        manh_val = abs(x_val - driver_x) + abs(y_val - driver_y)
        in_slew_val = 0.0  # need to be calculated
        out_slew_val = 0.0
        buf_type_name = buf_type_map[buf_type_id]
        row = buf_info_df[buf_info_df["buf_type"] == buf_type_name]
        in_cap_val = row["capacitance"].values[0] * 1e14  # (based on buf_type)
        out_cap_val = 0.0
        max_out_cap = row["max_capacitance"].values[0] * 1e14
        # fanout_val = len(buffer_locations[mask_c])
        buf_res = row["res"].values[0] * 0.01  # (based on buf_type)

        x_val_tensor = torch.tensor([x_val], device=device)
        y_val_tensor = torch.tensor([y_val], device=device)
        manh_val_tensor = torch.tensor([manh_val], device=device)
        in_slew_val_tensor = torch.tensor([in_slew_val], device=device)
        out_slew_val_tensor = torch.tensor([out_slew_val], device=device)
        in_cap_val_tensor = torch.tensor([in_cap_val], device=device)
        out_cap_val_tensor = torch.tensor([out_cap_val], device=device)
        max_out_cap_tensor = torch.tensor([max_out_cap], device=device)
        # fanout_val_tensor = torch.tensor([fanout_val], device=device)
        buf_res_tensor = torch.tensor([buf_res], device=device)

        # Construct the 13-d feature vector
        new_buf_feat = torch.cat([
            node_type_val, x_val_tensor, y_val_tensor, manh_val_tensor,
            in_slew_val_tensor, out_slew_val_tensor,
            in_cap_val_tensor, out_cap_val_tensor,
            max_out_cap_tensor,
            buf_res_tensor
        ], dim=0).unsqueeze(0)  # shape [1, 12]

        new_buf_feat_buf = torch.cat([
            node_type_val_buf, x_val_tensor, y_val_tensor, manh_val_tensor,
            in_slew_val_tensor, out_slew_val_tensor,
            in_cap_val_tensor, out_cap_val_tensor,
            max_out_cap_tensor,
            buf_res_tensor
        ], dim=0).unsqueeze(0)  # shape [1, 12]

        # loc = [x_val, y_val, manh_val]
        new_buf_loc = torch.tensor([x_val, y_val, manh_val], device=device).unsqueeze(0)
        # elc = [in_slew_val, out_slew_val, in_cap_val, out_cap_val, max_out_cap]
        new_buf_elc = torch.tensor([in_slew_val, out_slew_val, in_cap_val, out_cap_val, max_out_cap],
                                   device=device).unsqueeze(0)

        new_buf_features_list.append(new_buf_feat)
        new_buf_loc_list.append(new_buf_loc)
        new_buf_elc_list.append(new_buf_elc)
        new_buf_features_list_buf.append(new_buf_feat_buf)

    if len(new_buf_features_list) > 0:
        x_yes = torch.cat(new_buf_features_list, dim=0)  # shape [N_buf, 12]
        x_yes_buf = torch.cat(new_buf_features_list_buf, dim=0)
        loc_yes = torch.cat(new_buf_loc_list, dim=0)  # shape [N_buf, 3]
        elc_yes = torch.cat(new_buf_elc_list, dim=0)  # shape [N_buf, 5]
    else:
        # no new buffers
        x_yes = torch.zeros(0, features.size(1), device=device)
        x_yes_buf = torch.zeros(0, features.size(1), device=device)
        loc_yes = torch.zeros(0, loc_features.size(1), device=device)
        elc_yes = torch.zeros(0, elc_features.size(1), device=device)

        # Finally, combine the "no buffer" nodes with the "new buffer" nodes
    x_next = torch.cat([x_no, x_yes], dim=0)
    x_next_buf = torch.cat([x_no, x_yes_buf], dim=0)
    x_loc_next = torch.cat([loc_no, loc_yes], dim=0)
    x_elc_next = torch.cat([elc_no, elc_yes], dim=0)

    # -----------------update features------------------
    driver_i = 0
    driver_output_cap, driver_out_slew = calculate_updated_features(x_next, device)
    # print("test drvr cap&slew: ", driver_output_cap, driver_out_slew)
    x_next[driver_i, 9] = driver_output_cap
    x_next[driver_i, 7] = driver_out_slew
    x_next[1:, 6] = driver_out_slew

    x_next_buf[driver_i, 9] = driver_output_cap
    x_next_buf[driver_i, 7] = driver_out_slew
    x_next_buf[1:, 6] = driver_out_slew

    # elc = [in_slew_val, out_slew_val, in_cap_val, out_cap_val, max_out_cap]
    x_elc_next[driver_i, 3] = driver_output_cap
    x_elc_next[driver_i, 1] = driver_out_slew
    x_elc_next[1:, 0] = driver_out_slew

    return x_next, x_loc_next, x_elc_next, x_next_buf

def calculate_updated_features(x_next, device):
    """
    Compute driver's updated output_cap and output_slew.
    Handles the corner‑case where the driver has no children.
    """
    wire_res   = 3.5714e+04
    wire_cap   = 8.88758e+03
    elmore_k   = 1.39e+10
    dbu        = 2000
    driver_i   = 0

    # ----- 1. HPWL of children -----
    children_x = x_next[1:, 3]
    children_y = x_next[1:, 4]

    if children_x.numel() == 0:          # driver is alone – no wire, no load
        dx = dy = 0.0
    else:
        dx = (children_x.max() - children_x.min()).item()
        dy = (children_y.max() - children_y.min()).item()

    wire_length_m   = 1.2 * ((dx + dy) / (dbu * 1e6))      # metres
    total_wire_cap  = wire_length_m * wire_cap             # farads

    # ----- 2. Capacitive load -----
    children_in_cap = x_next[1:, 8].sum().item() if children_x.numel() > 1 else 0.0
    driver_output_cap = total_wire_cap + children_in_cap

    # ----- 3. Output slew (Elmore) -----
    r_drvr = x_next[driver_i, 11].item()                   # driver resistance
    driver_out_slew = (r_drvr + wire_length_m * wire_res) * driver_output_cap * elmore_k

    return driver_output_cap, driver_out_slew

def calculate_updated_features_needCheck(x_next, device):
    """
        calculate output slew, output cap,
    """
    wire_res = 3.5714e+04
    wire_cap = 8.88758e+03
    elmore_skew_factor_ = 1.39e+10
    dbu = 2000
    driver_i = 0
    # wirelength --> HPWL*1.2
    children_x = x_next[1:, 3]
    children_y = x_next[1:, 4]
    if children_x.numel() > 0:
        min_x, max_x = children_x.min(), children_x.max()
    else:
        min_x, max_x = 0, 0  
    # min_x, max_x = children_x.min(), children_x.max()
    # min_y, max_y = children_y.min(), children_y.max()
    if children_y.numel() > 0:
        min_y, max_y = children_y.min(), children_y.max()
    else:
        min_y, max_y = 0, 0  
    dx = (max_x - min_x).item()
    dy = (max_y - min_y).item()
    wire_length_m = 1.2 * ((dx + dy) / (dbu * 1e+6))  # meter
    total_wire_cap = wire_length_m * wire_cap  # (F)

    # output cap
    children_in_cap = x_next[1:, 8].sum()
    driver_output_cap = total_wire_cap + children_in_cap

    r_drvr = x_next[driver_i, 11].item()
    driver_out_slew = (r_drvr + wire_length_m * wire_res) * driver_output_cap * elmore_skew_factor_
    return driver_output_cap, driver_out_slew


def adjust_cluster_id(cluster_id):
    # -----------------------
    #    Build cluster_ID_list with shape = [n], for each node i in [0..n-1]
    #    If cluster_id[i] = -1 => unmerged => remain -1
    #    If cluster_id[i] != -1 => merged => new index of buffer in the next level (N_no + index_in 'used_ids')
    # -----------------------
    device = cluster_id.device
    n = cluster_id.size(0)
    cluster_ID_list = torch.full((n,), -1, dtype=torch.long, device=device)

    # separate "no buffer" vs "with buffer"
    no_buf_mask = (cluster_id == -1)
    yes_buf_mask = (cluster_id != -1)
    yes_buf_indices = torch.where(yes_buf_mask)[0]  # shape [N_yes]

    #    The merged sinks are grouped by used_ids => each used_id => one new buffer
    #    The new buffer nodes appear at indices [N_no..(N_no + len(yes_buf_indices)-1)]
    N_no = no_buf_mask.sum().item()
    cid_to_newindex = {}
    next_index = N_no  # Record index, strat with N_no

    for old_i in yes_buf_indices:
        c_val = int(cluster_id[old_i].item())
        if c_val not in cid_to_newindex:  # only assign cid when it first appears
            cid_to_newindex[c_val] = next_index
            next_index += 1
        cluster_ID_list[old_i] = cid_to_newindex[c_val]
    return cluster_ID_list


def compute_hpwl(coords):
    """
    coords: Tensor of shape [N, 2], each row is (x, y).
    Return HPWL as a float tensor.
    """
    x_min = coords[:, 0].min()
    x_max = coords[:, 0].max()
    y_min = coords[:, 1].min()
    y_max = coords[:, 1].max()
    hpwl = (x_max - x_min) + (y_max - y_min)
    return hpwl


# ----------------------------
# Build buffered tree
# ----------------------------

class TreeNode:
    def __init__(self,
                 node_type="Sink",
                 x=0.0,
                 y=0.0,
                 manh_dist=0.0,
                 in_slew=0.0,
                 out_slew=0.0,
                 in_cap=0.0,
                 out_cap=0.0,
                 max_out_cap=0.0,
                 fanout=0.0,
                 buf_id=0,
                 # res=0.0,
                 name=""):
        """
        node_type:  "Driver", "Buffer", or "Sink"
        x, y:       coordinates
        manh_dist, in_slew, out_slew, etc.: float fields for electrical metrics
        name:       optional string ID for debugging
        children:   list of child TreeNodes
        parent:     pointer to the parent TreeNode
        """
        self.node_type = node_type
        self.x = x
        self.y = y
        self.manh_dist = manh_dist
        self.in_slew = in_slew
        self.out_slew = out_slew
        self.in_cap = in_cap
        self.out_cap = out_cap
        self.max_out_cap = max_out_cap
        self.fanout = fanout
        # self.res = res
        self.buf_id = buf_id

        self.name = name
        self.children = []
        self.parent = None

    def add_child(self, child):
        self.children.append(child)
        child.parent = self

    def __repr__(self):
        return (f"TreeNode({self.node_type}, x={self.x:.1f}, y={self.y:.1f}, manh={self.manh_dist:.1f}, "
                f"name={self.name}, #children={len(self.children)})")


def print_tree_bfs(root):
    """
    Return a multiline string describing the subtree rooted at 'node'.
    """
    if not root:
        return "Tree is empty"

    queue = deque()
    id_counter = 0
    queue.append((root, 0, id_counter, None))
    id_counter += 1

    lines = []
    while queue:
        current, depth, node_id, parent_id = queue.popleft()
        indent = "  " * depth
        feature_info = (
            f"Type: {current.node_type}, X: {current.x:.3f}, Y: {current.y:.3f}, "
            f"Manh_dist: {current.manh_dist: .4f}, In_slew: {current.in_slew}, Out_slew: {current.out_slew}, "
            f"In_cap: {current.in_cap}, Out_cap: {current.out_cap}, Max_out_cap: {current.max_out_cap}, "
            f"Fanout: {current.fanout:.3f}", f"Buf_id: {current.buf_id}"
            # f"Resistance: {current.res}"
        )
        lines.append(f"{indent}Node id: {node_id}, Parent id: {parent_id}, {feature_info}")

        for child in current.children:
            queue.append((child, depth + 1, id_counter, node_id))
            id_counter += 1

    return "\n".join(lines)


def build_tree_bottom_up(
        pred_features,
        all_buffers,
        buf_info_df,
        buf_type_map,
        cluster_ids_history,
        net_name, file_key,
        out_file=None,
        print_tree=False
):
    """
    Build a single buffer tree, from the bottom level pred_features[0]
    up to pred_features[-1], where pred_features[-1][0] is the driver node.

    Args:
      pred_features: list of Tensors, each shape [N_i, feat_dim],
        where i=0 is the bottom, i=(num_levels-1) is the top
      all_buffers: list of [buffer_type_logits] for each step,
        all_buffers[i] is the buffers in level i-1 corresponding to sinks in level i

      cluster_ids_history[i] = cluster_id for each node at level i.
        cluster_id[j] = -1 => sink j not driven by a buffer in this level,
                   >=0 => driven by the buffer
        out_file: optional file path to save the textual tree.

    Returns:
      driver_node with children =buffers or sinks
    """

    num_levels = len(pred_features)
    # We store the TreeNode objects in "node_arrays[i]", one list per level
    level_nodes = []

    # 1) Create TreeNode objects for each level (bottom=0 ... top=num_levels-1)
    for i in range(num_levels):
        current_level = i
        # 'pred_features[i]' is the bottom-most level if i=0, or higher if i>0
        feats_i = pred_features[i]  # shape [N_i, feat_dim]
        feats_i_np = feats_i.cpu().numpy()
        bufs_id = all_buffers[i]
        if current_level > 0:
            # type_pred_i, loc_pred_i = all_buffers[current_level - 1]
            # buf_type_np = torch.argmax(type_pred_i, dim=-1).cpu().numpy()  # [N_i]
            # loc_pred_np = loc_pred_i.cpu().numpy()  # [N_i, 2]
            cid_i = cluster_ids_history[i - 1]
            # print("test cid: ", cid_i)

        node_list = []

        # For each node j in level i
        for j in range(feats_i_np.shape[0]):
            # if current_level == num_levels - 1:  # top level including driver and its direct children
            row = feats_i_np[j]
            # print("test build tree row: ", row)
            node_type = row[0:3]
            if j == 0:
                node_type_str = "Driver"
            elif current_level == 0:
                node_type_str = "Sink"
            else:
                # node_type_str = "Buffer" if j in cid_i else "Sink"
                node_type_str = "Buffer" if np.array_equal(node_type, np.array([1, 0, 0])) else "Sink"

            # columns: 0-2=node_type, 3=x, 4=y, 5=manh_dist, 6=in_slew,
            # 7=out_slew, 8=in_cap, 9=out_cap, 10=max_out_cap, 11=fanout
            x_val = float(row[3])
            y_val = float(row[4])
            manh_val = float(row[5])
            in_slew_val = float(row[6])
            out_slew_val = float(row[7])
            in_cap_val = float(row[8])
            out_cap_val = float(row[9])
            max_out_cap = float(row[10])
            fanout_val = float(row[11])
            # buf_type_name = buf_type_map[bufs_id[j]]
            # resistance = float(row[12])

            name_str = f"level{i}_node{j}"
            nd = TreeNode(node_type=node_type_str,
                          x=x_val, y=y_val,
                          manh_dist=manh_val,
                          in_slew=in_slew_val,
                          out_slew=out_slew_val,
                          in_cap=in_cap_val,
                          out_cap=out_cap_val,
                          max_out_cap=max_out_cap,
                          fanout=fanout_val,
                          buf_id=int(bufs_id[j]),
                          # res=resistance,
                          name=name_str)
            # print("[test nd!!!!!]: ", x_val, y_val, node_type_str, in_slew_val, out_slew_val, in_cap_val, out_cap_val,
            #       fanout_val, name_str)
            node_list.append(nd)
        level_nodes.append(node_list)  # bottom-up, level_nodes[0] is the most bottom nodes

    # 2) Link levels using cluster_ids_history
    #    cluster_ids_history[i] => links level i to i+1
    #    if cluster_ids_history[i][j] = c >= 0 => level_nodes[i][j] is child, level_nodes[i+1][c] is parent
    for i in range(num_levels - 1):
        cid_tensor = cluster_ids_history[i]  # shape [N_(i+1)]
        cids = cid_tensor.cpu().numpy()  # [N_(i+1)]

        if len(cids) > 0:
            cids[0] = -1  # driver doesn't not driven by any buffer

        children = level_nodes[i]  # level i
        parents = level_nodes[i + 1]  # level i+1

        for j, c in enumerate(cids):  # construct connection
            if c >= 0:
                parent_nd = parents[c]
                child_nd = children[j]
                parent_nd.add_child(child_nd)
            else:
                # c = -1 => no buffer merges => child_nd is "top" in that level
                pass

    # 3) The top level is level_nodes[-1].
    # Add nodes in level_nodes[-1] to the children of the driver
    if len(level_nodes[-1]) > 0:
        driver_node = level_nodes[-1][0]
        driver_node.node_type = "Driver"
        driver_node.name = "Driver_top"
    else:
        # If there's no node => create a dummy
        driver_node = TreeNode(node_type="Driver", name="dummy_driver")
    for node in level_nodes[-1][1:]:
        if node.parent is None:
            driver_node.add_child(node)
        else:
            print("[Warning] The top level node has already driven by some other node")

    # 4) Print the entire tree from driver down

    tree_str = print_tree_bfs(driver_node)
    if print_tree:
        print("=== Final Buffer Tree ===")
        print(tree_str)

    # 5) Save to a file
    if out_file is not None:
        with open(out_file, "a", encoding="utf-8") as f:
            f.write(f"===Predicted Buffered Tree for NET [{net_name}] in FILE [{file_key}]\n")
            f.write(tree_str)
            f.write("\n")

    return driver_node
