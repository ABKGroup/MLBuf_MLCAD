import os
import glob
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
import numpy as np


def process_feat_vec(feat_vec):
    """
    Do not use this func
    process features to make them distributed more evenly
    0..2 => one hot embedding (do not change)
    3..5 => x_real, y_real, dist (do not change)
    6 => in_slew (*10^10)
    7 => output_slew (*10^10)
    8 => in_cap (*10^14)
    9 => out_cap (*10^14)
    10 => max_cap (*10^14)
    11 => res (*10^-2)

    """
    new_feat = feat_vec.clone()
    new_feat[6] *= 1e10
    new_feat[7] *= 1e10
    new_feat[8:11] *= 1e14
    new_feat[11] *= 1e-2
    return new_feat


class NetTreeDataset(Dataset):
    def __init__(self, data_dirs, buf_info_df):
        """
        1) Collect all 'post' CSV files in `data_dir` matching pattern:
           buffered_trees_post_*.csv
        2) For each file, read the CSV and note all unique net names.
        3) Build a list of (file_key, net_name) so we can batch them individually.
        """
        self.buf_info_df = buf_info_df
        self.data_dirs = data_dirs
        self.post_files = []
        self.mid_files = []

        for data_dir in data_dirs:
            # Collect all post files
            post_files = sorted(glob.glob(os.path.join(data_dir, "buffered_trees_post_*.csv")))
            self.post_files.extend(post_files)

            # Collect all mid files
            mid_files = sorted(glob.glob(os.path.join(data_dir, "buffered_trees_mid_*.csv")))
            self.mid_files.extend(mid_files)

        # Map: file_key -> post_file
        # Example: data_dir/buffered_trees_post_XXXX.csv => file_key = "XXXX"
        self.file_key_to_file = {}
        self.post_dfs = {}

        for post_file in self.post_files:
            base = os.path.basename(post_file)
            # e.g. "buffered_trees_post_XXXX.csv"
            dir_name = os.path.basename(os.path.dirname(post_file))  # directory name
            key = f"{dir_name}_{base.split('buffered_trees_post_')[1].split('.csv')[0]}"
            # key = base.split("buffered_trees_post_")[1].split(".csv")[0]
            self.file_key_to_file[key] = post_file

            df_post = pd.read_csv(post_file)
            self.post_dfs[key] = df_post

        # Map: file_key -> mid_file
        # self.mid_files = sorted(glob.glob(os.path.join(data_dir, "buffered_trees_mid_*.csv")))
        self.mid_file_key_to_file = {}
        for mid_file in self.mid_files:
            base = os.path.basename(mid_file)
            dir_name = os.path.basename(os.path.dirname(mid_file))
            key = f"{dir_name}_{base.split('buffered_trees_mid_')[1].split('.csv')[0]}"
            # key = base.split("buffered_trees_mid_")[1].split(".csv")[0]
            self.mid_file_key_to_file[key] = mid_file

        # Build a list of (file_key, net_name)
        self.file_net_pairs = []
        for key, post_path in self.file_key_to_file.items():
            df_post = pd.read_csv(post_path)
            unique_nets = df_post["Net Name"].unique()
            for net_name in unique_nets:
                self.file_net_pairs.append((key, net_name))

        # 4) parse mid files：
        #    self.mid_data[file_key][net_name][tree_id][pin_name] -> row(Series)
        self.mid_data = {}
        for key, mid_path in self.mid_file_key_to_file.items():
            df_mid = pd.read_csv(mid_path)
            self.mid_data[key] = {}

            # For each row，(net_name, tree_id, pin_name)
            tree_id_map = {}  # (net_name -> { original_tree_id -> new_tree_id })
            counter_for_net = {}  # (net_name -> int )

            for i, row in df_mid.iterrows():
                net_name = row["Net Name"]
                original_tree_id = int(row["Tree ID"])
                pin_name = row["Pin Name"]

                if net_name not in tree_id_map:
                    tree_id_map[net_name] = {}
                    counter_for_net[net_name] = 0  # count from 0

                if original_tree_id not in tree_id_map[net_name]:
                    new_tree_id = counter_for_net[net_name]
                    tree_id_map[net_name][original_tree_id] = new_tree_id
                    counter_for_net[net_name] += 1
                else:
                    new_tree_id = tree_id_map[net_name][original_tree_id]

                if net_name not in self.mid_data[key]:
                    self.mid_data[key][net_name] = {}
                if new_tree_id not in self.mid_data[key][net_name]:
                    self.mid_data[key][net_name][new_tree_id] = {}

                self.mid_data[key][net_name][new_tree_id][pin_name] = row

        # Collect all distinct node types to build a map for one-hot
        # (Driver, Sink, Buffer, etc.)
        all_types = set()
        for post_path in self.post_files:
            df_tmp = pd.read_csv(post_path)
            all_types.update(df_tmp["Node Type"].unique())
        self.node_types = sorted(list(all_types))
        self.node_type_to_idx = {'Buffer': 0, 'Driver': 1, 'Sink': 2}
        # Buffer: 1,0,0
        # Driver: 0,1,0
        # Sink: 0,0,1
        # self.node_type_to_idx = {t: i for i, t in enumerate(self.node_types)}
        # Warning! buffer type
        self.buf_type_map = {
            '-1': 0,
            'BUF_X2': 1,
            'BUF_X4': 2,
            'BUF_X8': 3,
            'BUF_X16': 4,
            'BUF_X32': 5
        }
        # self.buf_type_map = {
        #     '-1': 0,
        #     'BUF_X2': 1,
        # }

    def __len__(self):
        return len(self.file_net_pairs)

    def __getitem__(self, idx):
        """
        Returns a tuple:
          (layers_data, net_name, file_key)

        Where layers_data is a list of dicts, one per depth:
            {
                "input_feats": tensor [N, feat_dim],
                "loc_feats":   tensor [N, 3],
                "elc_feats":   tensor [N, 5],
                "cluster_label":     tensor [N],  (sink-based cluster IDs)
                "buffer_type_label": tensor [N],  (0 or 1)
                "buffer_loc_label":  tensor [N,2],
                "depth": i,
                "node_ids": [...],
            }
        Depth = 1: driver + sinks & buffers(depth = 1)
        Depth = i: driver + buffer & sinks(depth=i) + sinks(depth<i)
        Depth = n: driver + sinks (no buffer)
        """
        file_key, net_name = self.file_net_pairs[idx]
        # post_file = self.file_key_to_file[file_key]

        # 1) Read CSV for this net
        # df = pd.read_csv(post_file)
        # df = df[df["Net Name"] == net_name].reset_index(drop=True)
        df = self.post_dfs[file_key]
        df = df[df["Net Name"] == net_name].reset_index(drop=True)

        if df.empty:
            return None

        # 2) Build node_dict with parent/children
        node_dict = {}
        max_depth = 0
        driver_id = None

        for i, row in df.iterrows():
            node_id = int(row["Node ID"])
            depth = int(row["Depth"])
            ntype = row["Node Type"]
            x = float(row["X"])
            y = float(row["Y"])
            parent = row["Parent Node ID"]
            if pd.isna(parent) or parent == "None":
                parent = None
            else:
                parent = int(parent)

            # Record Buffer Order
            buf_order = row["Buffer Order"] if not pd.isna(row["Buffer Order"]) else 0

            # Track driver
            if ntype == "Driver":
                driver_id = node_id

            if depth > max_depth:
                max_depth = depth

            # Build dictionary
            node_dict[node_id] = {
                "depth": depth,
                "type": ntype,
                "pin_name": row["Pin Name"],  # optional, if needed
                "x": x,
                "y": y,
                "in_slew": float(row["Input Slew"]) if row["Input Slew"] != -1 else 0.0,
                "out_slew": float(row["Output Slew"]) if row["Output Slew"] != -1 else 0.0,
                "in_cap": float(row["Input Cap"]) if row["Input Cap"] != -1 else 0.0,
                "out_cap": float(row["Output Cap"]) if row["Output Cap"] != -1 else 0.0,
                "max_out_cap": float(row["Max Output Cap"]) if row["Max Output Cap"] != -1 else 0.0,
                "fanout": int(row["Fanout"]) if row["Fanout"] != -1 else 0,
                "buf_order": int(buf_order),
                "buf_type": str(row["Buffer Type"]),
                "res": float(row["Resistance"]) if row["Resistance"] != -1 else 0.0, # new
                "parent": parent,
                "children": []

            }

        # Link children
        for nid, info in node_dict.items():
            p = info["parent"]
            if p is not None and p in node_dict:
                node_dict[p]["children"].append(nid)

        if driver_id is None:
            raise ValueError(f"No driver found for net='{net_name}' in {post_file}")

        has_sink = any(info["type"] == "Sink" for info in node_dict.values())
        if not has_sink:
            # if there is no sinks in the net, ignore the net
            return None

        # 3) Compute driver coords, manhattan distance, shift coords
        driver_x = node_dict[driver_id]["x"]
        driver_y = node_dict[driver_id]["y"]

        for nid, info in node_dict.items():
            dist = abs(info["x"] - driver_x) + abs(info["y"] - driver_y)
            info["manh_dist"] = dist
            info["x_rel"] = info["x"] - driver_x
            info["y_rel"] = info["y"] - driver_y
            # Adjust X, Y relative to driver

        # 4) Build hierarchical layers from depth=1..max_depth
        layers_data = []
        for depth_i in range(1, max_depth + 1):
            # => input nodes:
            #    driver (depth=0),
            #    all nodes at depth = i (**both** buffer & sink),
            #    all sink at depth < i.
            input_nids = set()
            # Always include driver:
            input_nids.add(driver_id)
            for nid, info in node_dict.items():
                d = info["depth"]
                t = info["type"]
                if d == depth_i:
                    # This layer: we treat Buffers & Sinks identically at depth i
                    if t in ["Sink", "Buffer"]:
                        input_nids.add(nid)
                elif d < depth_i:
                    # If it's a sink at a shallower depth, include it
                    if t == "Sink":
                        input_nids.add(nid)

            input_nids = sorted(list(input_nids))

            # =============================
            # IMO: import features from mid_data
            # =============================

            # 1) Find the largest value of Buffer Order in this depth
            depth_buf_orders = [
                node_dict[nid]["buf_order"]
                for nid in input_nids
                if node_dict[nid]["depth"] == depth_i and node_dict[nid]["type"] == "Buffer"
            ]
            # if len(depth_buf_orders) == 0:
            #     # only driver?
            #     max_bo = 0
            # else:
            if depth_i == max_depth:
                max_bo = 0
            else:
                max_bo = max(depth_buf_orders)
            # print("test0 Max buffer order: ", max_bo)

            # 2) Find features from self.mid_data[file_key][net_name][max_bo]
            mid_map = None
            if (file_key in self.mid_data and
                    net_name in self.mid_data[file_key] and
                    max_bo in self.mid_data[file_key][net_name]):
                mid_map = self.mid_data[file_key][net_name][max_bo]
            # mid_map : pin_name -> row(Series)

            # Prepare feature vectors
            input_feats_list = []
            loc_feats_list = []
            elc_feats_list = []

            # Prepare GT: cluster_label, buffer_type_label, buffer_loc_label
            cluster_labels_list = []
            buffer_type_list = []
            buffer_loc_list = []

            # We'll define a helper: if a node at depth i is "driven by" a node at depth i-1 that is a Buffer,
            # we unify them in the same cluster. We'll treat "buffer" as we did for sinks in the older example.
            def get_buffer_parent(nid):
                """
                Traverse up until we find a parent that is 'Buffer' at depth = (depth_i - 1).
                Return that node_id or None if none found.
                """
                p = node_dict[nid]["parent"]
                while p is not None:
                    # print("Visiting node:", p)  # Debugging output
                    if node_dict[p]["depth"] == (depth_i - 1) and node_dict[p]["type"] == "Buffer":
                        return p, node_dict[p]["buf_type"]
                    p = node_dict[p]["parent"]
                return None, None

            # We'll keep a map: buffer_parent -> cluster_id
            buffer_parent_to_cid = {}
            next_cluster_id = 0

            for nid in input_nids:  # process the node one by one based on NodeID in input_nids
                info = node_dict[nid]

                # -------------------------
                # (A) extract ground truth (label)
                # -------------------------
                d = info["depth"]
                t = info["type"]

                if d == depth_i and t in ["Sink", "Buffer"]:
                    # It's a current-level node (treated as "sink-like" )
                    bp, btype_str = get_buffer_parent(nid)
                    if bp is not None:
                        if bp not in buffer_parent_to_cid:
                            buffer_parent_to_cid[bp] = next_cluster_id
                            next_cluster_id += 1
                        cluster_id = buffer_parent_to_cid[bp]
                        buffer_type = self.buf_type_map[btype_str]
                        # buffer location => that buffer's x_rel, y_rel
                        bx = node_dict[bp]["x_rel"]
                        by = node_dict[bp]["y_rel"]
                    else:
                        # no buffer parent => new cluster
                        cluster_id = next_cluster_id
                        next_cluster_id += 1
                        buffer_type = 0
                        bx, by = -1, -1
                elif d < depth_i and t == "Sink":
                    # higher-level sink => alone in own cluster
                    cluster_id = next_cluster_id
                    next_cluster_id += 1
                    buffer_type = 0
                    bx, by = -1, -1
                else:
                    # driver, or buffer at shallower depth, or a node that doesn't fit the above
                    cluster_id = -1
                    buffer_type = 0
                    bx, by = -1, -1

                cluster_labels_list.append(cluster_id)
                buffer_type_list.append(buffer_type)
                buffer_loc_list.append([bx, by])
                
                # -------------------------
                # (B) extract features from mid files
                # -------------------------
                # extract features from mid files.
                # if can not find corresponding features, use features in post files
                pin_name = info["pin_name"]
                # print("pin name: ", pin_name)
                # print("mid map: ", mid_map)
                if (mid_map is None):
                    print("error mid map")
                    print("mid map: ", mid_map)
                    print("for debugging, file, net, maxbo: ", file_key, net_name, max_bo)
                if (pin_name not in mid_map):
                    print("Error pin name", pin_name)
                if (mid_map is not None) and (pin_name in mid_map):
                    mid_row = mid_map[pin_name]
                    # print("mid row: ", mid_row)

                    # node_type still use the value in post file
                    if info["type"] == "Buffer":
                        info["type"] = "Sink"  # treat buffer as sink in the input feature
                        ntype_idx = self.node_type_to_idx.get(info["type"], -1)
                        ntype_onehot = F.one_hot(torch.tensor(ntype_idx), num_classes=len(self.node_types)).float()
                        info["type"] = "Buffer"
                    else:
                        ntype_idx = self.node_type_to_idx.get(info["type"], -1)
                        ntype_onehot = F.one_hot(torch.tensor(ntype_idx), num_classes=len(self.node_types)).float()

                    x_m = float(mid_row["X"])
                    y_m = float(mid_row["Y"])
                    in_sl_m = float(mid_row["Input Slew"] * 1e10) if mid_row["Input Slew"] != -1 else 0.0
                    out_sl_m = float(mid_row["Output Slew"] * 1e10) if mid_row["Output Slew"] != -1 else 0.0
                    in_cp_m = float(mid_row["Input Cap"] * 1e14) if mid_row["Input Cap"] != -1 else 0.0
                    out_cp_m = float(mid_row["Output Cap"] * 1e14) if mid_row["Output Cap"] != -1 else 0.0
                    max_cp_m = float(mid_row["Max Output Cap"] * 1e14) if mid_row["Max Output Cap"] != -1 else 0.0
                    # fan_m = float(mid_row["Fanout"])
                    res_m = float(mid_row["Resistance"] * 1e-2) if mid_row["Resistance"] != -1 else 0.0
                    # calculate relative coor and man_dis between the driver
                    x_rel = x_m - driver_x
                    y_rel = y_m - driver_y
                    dist = abs(x_rel) + abs(y_rel)

                    # !!--- Set max output cap of buffer based on buf_info ----#
                    if info["type"] == "Buffer":
                        # max_cp_m = -1.0
                        row = self.buf_info_df[self.buf_info_df["buf_type"] == info["buf_type"]]
                        max_cp_m = row["max_capacitance"].values[0]
                        max_cp_m *= 1e14

                    numeric_feats = torch.tensor([
                        x_rel, y_rel, dist,
                        in_sl_m, out_sl_m,
                        in_cp_m, out_cp_m,
                        max_cp_m,
                        res_m
                    ], dtype=torch.float32)
                    feat_vec = torch.cat([ntype_onehot, numeric_feats], dim=0)

                else:
                    print("[Warning!!!!!] can not find corresponding mid file: ", file_key, net_name,
                          max_bo)

                input_feats_list.append(feat_vec)

                # loc_feats: e.g. [x_rel, y_rel, dist]
                loc_feats_list.append(torch.tensor([x_rel, y_rel, dist], dtype=torch.float32))

                # elc_feats: e.g. [in_slew, out_slew, in_cap, out_cap, max_out_cap]
                elc_feats_list.append(torch.tensor([
                    feat_vec[len(self.node_types) + 3].item(),  # in_slew
                    feat_vec[len(self.node_types) + 4].item(),  # out_slew
                    feat_vec[len(self.node_types) + 5].item(),  # in_cap
                    feat_vec[len(self.node_types) + 6].item(),  # out_cap
                    feat_vec[len(self.node_types) + 7].item()  # max_out_cap
                ], dtype=torch.float32))

            # Convert to tensors
            input_feats_tensor = torch.stack(input_feats_list, dim=0)
            loc_feats_tensor = torch.stack(loc_feats_list, dim=0)
            elc_feats_tensor = torch.stack(elc_feats_list, dim=0)

            cluster_label_tensor = torch.tensor(cluster_labels_list, dtype=torch.long)
            buffer_type_tensor = torch.tensor(buffer_type_list, dtype=torch.long)
            buffer_loc_tensor = torch.tensor(buffer_loc_list, dtype=torch.float32)

            layer_data = {
                "depth": depth_i,
                "input_feats": input_feats_tensor,
                "loc_feats": loc_feats_tensor,
                "elc_feats": elc_feats_tensor,
                "cluster_label": cluster_label_tensor,
                "buffer_type_label": buffer_type_tensor,
                "buffer_loc_label": buffer_loc_tensor,
                "node_ids": input_nids
            }
            layers_data.append(layer_data)

        return layers_data, net_name, file_key, node_dict


#
# Collate & DataLoader
#
def collate_fn(batch):
    """
    Each item in batch is (layers_data, net_name, file_key).
    If you use batch_size=1, you get exactly one net.
    If batch_size>1, you'd have multiple nets at once, which might or might not be desired.
    We'll just return them as-is in a list.
    """
    """
        Each item in batch is either None or a tuple:
            (layers_data, net_name, file_key, node_dict)
        If it's None, we skip it.
    """
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None
    return batch


def get_dataloaders(data_dirs, buf_info_df, batch_size=1, split_ratios=(0.7, 0.15, 0.15), seed=42):
    # dataset = NetTreeDataset(data_dir, buf_info_df)
    # dataloader = DataLoader(dataset, batch_size=batch_size,
    #                         shuffle=shuffle, collate_fn=collate_fn)
    full_dataset = NetTreeDataset(data_dirs, buf_info_df)
    total_size = len(full_dataset)
    train_size = int(split_ratios[0] * total_size)
    val_size = int(split_ratios[1] * total_size)
    test_size = total_size - train_size - val_size

    # randomly partition the dataset
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size], generator=generator
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader


def compute_feature_stats(layers_data):
    """
    input feature statistics：min, max, mean, std
    """
    all_feats = []

    for layer in layers_data:
        feats = layer["input_feats"]  # shape [N, feat_dim]
        all_feats.append(feats)

    all_feats = torch.cat(all_feats, dim=0)

    feat_dim = all_feats.shape[1]
    # min, max, mean, std
    mins = all_feats.min(dim=0)[0]
    maxs = all_feats.max(dim=0)[0]
    means = all_feats.mean(dim=0)
    stds = all_feats.std(dim=0)

    print("Feature dimension =", feat_dim)
    for d in range(feat_dim):
        print(f"Dim {d}: min={mins[d]}, max={maxs[d]}, mean={means[d]}, std={stds[d]}")

    return {
        "mins": mins,
        "maxs": maxs,
        "means": means,
        "stds": stds
    }


