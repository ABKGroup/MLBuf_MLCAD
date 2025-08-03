import sys
sys.path.append('MLBuf/OR_branch_integration/mlbuf_utils')

import random
import time

from mlbuf_utils.util import adjust_cluster_id, build_tree_bottom_up, adjust_model_output
import torch
import pandas as pd
import argparse
from mlbuf_utils.data_process import NetDataset
import math
from mlbuf_utils.model import MLBuf
from torch.utils.data import Dataset, DataLoader

"""
This code is used for integrate_OR

OR will call mlbuf_infer.py when all problematic nets in this placement iteration
are saved in a pre-defined file and MLBuf is used

"""


def get_args():
    parser = argparse.ArgumentParser(description='Args for MLBuf with Scheduled Sampling')
    parser.add_argument('-num_epochs', type=int, default=600, help='Number of training epochs')
    parser.add_argument('-lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('-weight_decay', type=float, default=0.0001, help='Weight decay')

    # Model hyperparams
    parser.add_argument('-hidden_dim', type=int, default=256)
    parser.add_argument('-clustering_output_dim', type=int, default=128)
    parser.add_argument('-drop_n', type=float, default=0.3)
    parser.add_argument('-drop_c', type=float, default=0.2)
    parser.add_argument('-act_n', type=str, default='ELU')
    parser.add_argument('-act_c', type=str, default='ELU')
    parser.add_argument('-num_heads', type=int, default=1)
    parser.add_argument('-output_dim_bloc', type=int, default=2)
    parser.add_argument('-output_dim_btype', type=int, default=6)
    parser.add_argument('-input_dim_share', type=int, default=12)
    parser.add_argument('-input_dim_loc', type=int, default=3)
    parser.add_argument('-input_dim_els', type=int, default=5)
    parser.add_argument('-max_clusters', type=int, default=10)

    # Scheduled Sampling parameters
    parser.add_argument('-scheduled_sampling_max', type=float, default=1.0,
                        help='Initial teacher forcing probability (use ground truth).')
    parser.add_argument('-scheduled_sampling_min', type=float, default=0.0,
                        help='Minimum teacher forcing probability.')
    parser.add_argument('-scheduled_sampling_decay', type=float, default=0.02,
                        help='Decay rate for teacher forcing probability each epoch.')

    args, _ = parser.parse_known_args()
    return args


def generate_buffer_lines(all_buffers_coor, all_buffers_type, buf_info_df, buf_type_map, drvrX, drvrY):
    """
    Instead of writing directly, this function returns a list of string lines,
    each corresponding to one buffer's bounding box.
    """
    scale = math.sqrt(1.1)  # ~1.0488

    # If no buffers are predicted, return an empty list
    if not all_buffers_coor or not all_buffers_type:
        return []

    # Concatenate the list of tensors into single tensors
    all_buffers_coor = torch.cat(all_buffers_coor, dim=0)  # shape [M, 2]
    all_buffers_type = torch.cat(all_buffers_type, dim=0)  # shape [M]

    lines = []
    for buf_index in range(all_buffers_coor.shape[0]):
        buf_x = all_buffers_coor[buf_index, 0] + drvrX
        buf_y = all_buffers_coor[buf_index, 1] + drvrY

        buf_type = all_buffers_type[buf_index].item()
        buf_type_name = buf_type_map[buf_type]

        # Get buffer width and height from buf_info_df based on buf_type_name
        row = buf_info_df[buf_info_df["buf_type"] == buf_type_name]
        bufferWidth = row["width"].values[0]
        bufferHeight = row["height"].values[0]

        scaledW = bufferWidth * scale
        scaledH = bufferHeight * scale

        bbox_lx = buf_x - 0.5 * (scaledW - bufferWidth)
        bbox_ly = buf_y - 0.5 * (scaledH - bufferHeight)
        bbox_ux = bbox_lx + scaledW
        bbox_uy = bbox_ly + scaledH

        bbox_area = (bbox_ux - bbox_lx) * (bbox_uy - bbox_ly)

        line = f"{bbox_lx.item()},{bbox_ly.item()},{bbox_ux.item()},{bbox_uy.item()},{bbox_area.item()}"
        lines.append(line)
    return lines


def write_lines_to_file(lines, output_file, secondary_file):
    """
    Write accumulated lines to files in one go.
    """
    with open(output_file, 'w') as f:
        all_text = "\n".join(lines) + "\n"
        f.write(all_text)
        # f2.write(all_text)
    print(f"[INFO] Saved {len(lines)} buffers' bounding boxes to {output_file} ")


def inference(model, model_save_path, dataloader, device, buf_info_df, buf_type_map):
    """
    Inference (recursive) without teacher forcing:
    The model's own predictions feed into the next level.
    """
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    model.eval()
    all_pred_lines = []  # Accumulate prediction lines across nets
    for i, (layer_data, net_name) in enumerate(dataloader):
        record_list = []  # record runtime, net size
        record_list.append(net_name)
        
        only_infer_time = 0
        fea_update_time = 0
        total_infer_st = time.time()
        with torch.no_grad():

            input_features = layer_data['input_feats'].squeeze(0).to(device)
            loc_features = layer_data["loc_feats"].squeeze(0).to(device)
            elc_features = layer_data["elc_feats"].squeeze(0).to(device)
            record_list.append(len(input_features))

            drvrX = layer_data.get("drvr_x", torch.tensor(0.0)).squeeze()
            drvrY = layer_data.get("drvr_y", torch.tensor(0.0)).squeeze()

            all_buffers_coor = []
            all_buffers_type = []

            still_generating = True
            num_level = 0
            while still_generating:
                only_infer_start = time.time()
                buffer_type_logits, buffer_location_logits, cluster_id, cluster_probs, cluster_embed, _, _ = model(
                    input_features, loc_features, elc_features
                )
                only_infer_end = time.time()
                only_infer_time += only_infer_end - only_infer_start

                buffer_type_pred = torch.argmax(buffer_type_logits, dim=-1)

                if buffer_type_pred.eq(0).all():
                    still_generating = False
                else:
                    yes_buf_mask = (buffer_type_pred > 0)
                    all_buffers_coor.append(buffer_location_logits[yes_buf_mask])
                    all_buffers_type.append(buffer_type_pred[yes_buf_mask])

                    num_level += 1
                    fea_update_start = time.time()
                    input_features, loc_features, elc_features, input_features_buf = adjust_model_output(
                        input_features,
                        loc_features,
                        elc_features,
                        buffer_location_logits,
                        buffer_type_logits,
                        cluster_id,
                        buf_info_df,
                        buf_type_map
                    )
                    fea_update_end = time.time()
                    fea_update_time += fea_update_end - fea_update_start
                    input_features = input_features.to(device).float()
                    loc_features = loc_features.to(device).float()
                    elc_features = elc_features.to(device).float()

            if not all_buffers_type:
                all_buffers_type.append(torch.tensor([3]))
                bx = drvrX + random.uniform(0.1, 0.7)
                by = drvrY + random.uniform(0.1, 0.7)
                all_buffers_coor.append(torch.tensor([[bx, by]]))

            pred_lines = generate_buffer_lines(
                all_buffers_coor, all_buffers_type, buf_info_df, buf_type_map,
                drvrX, drvrY
            )
            all_pred_lines.extend(pred_lines)
    return all_pred_lines


def MLBuf_inference_result():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser(description="MLBuf inference.")
    parser.add_argument('--model', required=True, help='Path to the model file')
    parser.add_argument('--input', required=True, help='Path to input CSV file')
    parser.add_argument('--output', required=True, help='Path to output CSV file')
    parser.add_argument('--timeMLBuf', required=True, help='Path to the model file')
    parser.add_argument('--clusterNum', required=True, help='Path to the model file')
    args_input = parser.parse_args()
    args = get_args()
    clusterNum = args_input.clusterNum

    buf_info_file = '/home/fetzfs_projects/MLBuf/flows/OR_branch_integration/buf_data.csv'
    buf_info_df = pd.read_csv(buf_info_file)

    buf_type_map = {
        0: 'None',
        1: 'BUF_X2',
        2: 'BUF_X4',
        3: 'BUF_X8',
        4: 'BUF_X16',
        5: 'BUF_X32'
    }

    # Initialize model
    model = MLBuf(
        args.input_dim_share,
        args.input_dim_loc,
        args.input_dim_els,
        args.hidden_dim,
        args.num_heads,
        args.clustering_output_dim,
        int(clusterNum),
        args.output_dim_bloc,
        args.output_dim_btype,
    ).to(device)
    print(f"Cluster Num: {args_input.clusterNum}")

    # load data
    load_data_start = time.time()
    prob_net = args_input.input
    dataset = NetDataset(prob_net)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    print(f"Number of distinct nets = {len(dataset)}")
    load_data_end = time.time()

    print("Start inference ...")
    all_pred_lines = inference(model, args_input.model, dataloader, device, buf_info_df, buf_type_map)

    # Write all predictions in one file write operation:
    output_file = args_input.output
    secondary_file = 'mlbuf_buffer_save0414.csv'
    write_data_st = time.time()
    write_lines_to_file(all_pred_lines, output_file, secondary_file)
    write_data_end = time.time()
    return args_input


if __name__ == '__main__':
    time_record_list = []
    start_time = time.time()
    args_input = MLBuf_inference_result()
