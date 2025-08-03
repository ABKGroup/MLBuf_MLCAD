import sys
sys.path.append('/home/fetzfs_projects/MLBuf/flows/OR_branch_integration/mlbuf_utils')
import os
import glob
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
import numpy as np

from torch.utils.data import Dataset, DataLoader


class NetDataset(Dataset):
    """
    A Dataset that splits the entire CSV by Net Name,
    and extracts features for each net.
    """

    def __init__(self, csv_file):
        # Read all data at once
        df = pd.read_csv(csv_file)

        # Group by Net Name (each group is one net)
        self.net_groups = []
        for net_name, df_net in df.groupby("Net Name"):
            self.net_groups.append((net_name, df_net))

        # Alternatively: self.net_groups = list(df.groupby("Net Name"))

        # We'll process each group on-the-fly in __getitem__.
        # If you'd prefer pre-processing everything once, you can do so here.

    def __len__(self):
        return len(self.net_groups)

    def __getitem__(self, idx):
        net_name, df_net = self.net_groups[idx]

        # Extract features for this net
        result = feature_extraction_for_single_net(df_net.copy())
        # We use df_net.copy() to avoid altering the original data with extra columns.

        # result is either (layer_data, net_name) or None
        if result is None:
            # If the net has no Sink, we can either:
            # (A) skip it, or
            # (B) return an empty structure.
            # Here let's raise an exception so we notice it.
            raise ValueError(f"Net '{net_name}' has no Sinks in the data!")

        layer_data, net_name_final = result
        return layer_data, net_name_final


def feature_extraction_for_single_net(df_single_net):
    """
    Returns features:
      (input features, loc_feats, elc_feats)

        {
            "input_feats": tensor [N, feat_dim],
            "loc_feats":   tensor [N, 3],
            "elc_feats":   tensor [N, 5],
        }
    If the net has no 'Sink' node, returns None.
    """

    # Ensure there's at least one sink
    if not (df_single_net["Node Type"] == "Sink").any():
        return None

    # Get net name (assume entire group is the same net)
    net_name = df_single_net["Net Name"].iloc[0]

    driver_row = df_single_net[df_single_net["Node Type"] == "Driver"].iloc[0]
    driver_x = float(driver_row["X"])
    driver_y = float(driver_row["Y"])

    df_single_net["x_rel"] = df_single_net["X"] - driver_x
    df_single_net["y_rel"] = df_single_net["Y"] - driver_y
    df_single_net["manh_dist"] = df_single_net["x_rel"].abs() + df_single_net["y_rel"].abs()

    columns_to_fix = [
        "Input Slew", "Output Slew", "Input Cap",
        "Output Cap", "Max Output Cap", "Fanout"
    ]
    for col in columns_to_fix:
        df_single_net[col] = df_single_net[col].apply(lambda x: x if x != -1 else 0.0)

    # one-hot embedding
    node_type_to_idx = {'Buffer': 0, 'Driver': 1, 'Sink': 2}
    df_single_net["ntype_idx"] = df_single_net["Node Type"].map(lambda t: node_type_to_idx.get(t, -1))
    ntype_tensor = F.one_hot(torch.tensor(df_single_net["ntype_idx"].values, dtype=torch.long), num_classes=3).float()

    numeric_array = np.stack([
        df_single_net["x_rel"].values,
        df_single_net["y_rel"].values,
        df_single_net["manh_dist"].values,
        df_single_net["Input Slew"].values * 1e10,
        df_single_net["Output Slew"].values * 1e10,
        df_single_net["Input Cap"].values * 1e14,
        df_single_net["Output Cap"].values * 1e14,
        df_single_net["Max Output Cap"].values * 1e14,
        # df_single_net["Fanout"].values
        df_single_net["Resistance"].values * 1e-2,
    ], axis=1)
    numeric_tensor = torch.tensor(numeric_array, dtype=torch.float32)
    input_feats = torch.cat([ntype_tensor, numeric_tensor], dim=1)  # shape [N, 13]

    # loc_feats and elc_feats
    loc_feats = torch.tensor(df_single_net[["x_rel", "y_rel", "manh_dist"]].values, dtype=torch.float32)
    # elc_feats = torch.tensor(df_single_net[["Input Slew", "Output Slew", "Input Cap", "Output Cap", "Max Output Cap"]].values,dtype=torch.float32)
    elc_feats = torch.tensor(np.stack([
        df_single_net["Input Slew"].values * 1e10,
        df_single_net["Output Slew"].values * 1e10,
        df_single_net["Input Cap"].values * 1e14,
        df_single_net["Output Cap"].values * 1e14,
        df_single_net["Max Output Cap"].values * 1e14,
    ], axis=1), dtype=torch.float32)

    layer_data = {
        "input_feats": input_feats,
        "loc_feats": loc_feats,
        "elc_feats": elc_feats,
        "drvr_x": driver_x,
        "drvr_y": driver_y
    }

    return layer_data, net_name


# if __name__ == "__main__":
#     prob_net = '../../../mlbuf-dev/prob_nets.csv'
#     dataset = NetDataset(prob_net)

#     dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

#     print(f"Number of distinct nets = {len(dataset)}")

#     for i, (layer_data, net_name) in enumerate(dataloader):
#         # 'layer_data' is a dict of 3 tensors:
#         #   layer_data["input_feats"], layer_data["loc_feats"], layer_data["elc_feats"]
#         #
#         # 'net_name' is a list/string, but if batch_size=1 you can just net_name[0].
#         # Alternatively, you can keep it as is.

#         print(f"Net Name: {net_name}")

#         # layer_data is a dict, each entry has shape [N, ...]
#         input_feats = layer_data["input_feats"].squeeze(0)  # shape [N, 13]
#         loc_feats = layer_data["loc_feats"].squeeze(0)  # shape [N,  3]
#         elc_feats = layer_data["elc_feats"].squeeze(0)  # shape [N,  5]
#         dx = layer_data["drvr_x"]
#         dy = layer_data["drvr_y"]
#         print("drvr xy: ", dx, dy)

#         print("input_feats shape:", input_feats.shape)
#         print("loc_feats shape:  ", loc_feats.shape)
#         print("elc_feats shape:  ", elc_feats.shape)

