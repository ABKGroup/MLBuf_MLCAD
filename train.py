import pandas as pd
import torch
from torch.optim import Adam
from models.model import MLBuf
from models.inference_model_0418 import inference_for_testing, inference
from models.losses import *
import utils.util as util
import argparse
from data.data_loader import get_dataloaders
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import random
import utils.plot_utils as plot_utils


def recursive_training(
        model,
        inputs,
        optimizer,
        teacher_forcing_prob,
        device,
        buf_info_df,
        buf_type_map,
        max_wirelength_constraints,
        penalty_flag_list,
        epoch,
        weight_list,
        init_tau=1,
        min_tau=0.1,
        anneal_rate=0.98
):
    """
    Perform recursive training with teacher forcing

    Args:
        model: The MLBuf model instance.
        inputs: A single training sample or batch of data (features, ground truths, etc.).
        optimizer: Optimizer instance.
        teacher_forcing_prob: Probability (0~1) of using ground-truth as input (teacher forcing)
                              instead of the model's own prediction.
        device: torch device (cpu/cuda).
        penalty_flag_list: [area_penalty, wire_penalty, cap_penalty]

    Returns:
        all_buffers: List of (buffer_type_pred, buffer_location_pred) for each level.
    """

    current_tau = max(min_tau, init_tau * (anneal_rate ** epoch))

    all_buffers = []
    cluster_ids_history = []
    pred_features = []  # store hierarchically predicted features

    local_loss_sum = torch.tensor(0.0, device=device, requires_grad=True)
    cluster_loss_sum = 0
    type_loss_sum = 0
    location_loss_sum = 0
    wire_penalty_sum = 0
    cap_penalty_sum = 0
    area_penalty_sum = 0
    total_area = torch.tensor(0.0, device=device, requires_grad=True)

    num_buffer_class = 6
    num_level = 0
    still_generating = True

    # --------------------------
    # Load data (bottom-up)
    # --------------------------

    max_level = len(inputs)
    current_level = max_level - num_level - 1
    input_features = inputs[current_level]["input_feats"].to(device)
    loc_features = inputs[current_level]["loc_feats"].to(device)
    elc_features = inputs[current_level]["elc_feats"].to(device)

    # Initialize "current" features that we feed to the model at each level
    current_input_feats = input_features
    current_loc_feats = loc_features
    current_elc_feats = elc_features
    pred_features.append(current_input_feats)

    while still_generating:
        # -------------------------------
        # 1) Forward pass at current level
        # -------------------------------
        buffer_type_pred, buffer_location_pred, cluster_id, cluster_probs, cluster_embed, cluster_buf_type, cluster_buf_loc = model(
            current_input_feats,
            current_loc_feats,
            current_elc_feats
        )

        # Load labels
        cluster_label = inputs[current_level]["cluster_label"].to(device)
        buffer_type_labels = inputs[current_level]["buffer_type_label"].to(device)
        buffer_loc_labels = inputs[current_level]["buffer_loc_label"].to(device)


        # -------------------------------
        # 2) Compute losses at this level
        # -------------------------------
        # -- Cluster loss
        cluster_loss_value = cluster_loss(cluster_embed, cluster_label)

        # -- Buffer type classification loss
        #    Use some class weighting / focal loss if desired
        class_weights = compute_class_weights(buffer_type_labels, num_buffer_class)
        type_loss_fn = FocalLoss(gamma=1.5, alpha=class_weights)
        type_loss_value = type_loss_fn(buffer_type_pred, buffer_type_labels)

        # -- Buffer location loss
        location_loss_value = location_loss(buffer_type_labels, buffer_location_pred, buffer_loc_labels)

        # Gumbel-Softmax => buffer_type_probs
        buffer_type_probs = F.gumbel_softmax(buffer_type_pred, tau=current_tau, hard=False)

        # --area penalty warning!!
        area_tensor = torch.tensor(
            [0, 1.064, 1.862, 3.458, 6.65, 13.03], device=device)
        epsilon = 1e-8
        # # Compute logs
        log_area = torch.log2(area_tensor + epsilon)
        log_area[0] = 0.0

        expected_area_per_node = (buffer_type_probs * log_area).sum(dim=-1)
        area_loss_value = expected_area_per_node.sum()
        total_area = total_area + area_loss_value

        # for each layer, compute the area penalty (less than sinks)
        area_penalty = compute_area_penalty(buffer_type_probs, log_area)
        area_penalty = area_penalty.sum()

        # ERC loss
        driver_wirelength, driver_output_cap, wire_penalty, cap_penalty = compute_cap_wirelength_penalty(
            current_input_feats,
            cluster_buf_type,
            cluster_buf_loc,
            cluster_probs,
            buf_info_df,
            buf_type_map,
            max_wirelength_constraints)

        # Combine them
        local_loss = (weight_list[0] * cluster_loss_value + weight_list[1] * type_loss_value + weight_list[
            2] * location_loss_value + penalty_flag_list[1] * wire_penalty * weight_list[4] + penalty_flag_list[
                          2] * cap_penalty * weight_list[5])

        # accumulate
        local_loss_sum = local_loss_sum + local_loss

        cluster_loss_sum += cluster_loss_value.item()
        type_loss_sum += type_loss_value.item()
        area_penalty_sum += area_penalty.item()
        location_loss_sum += location_loss_value.item()
        wire_penalty_sum += wire_penalty.item()
        cap_penalty_sum += cap_penalty.item()

        # -------------------------------
        # 4) Store predictions from this level
        # -------------------------------
        all_buffers.append((buffer_type_probs, buffer_location_pred))
        modified_cluster_id = util.adjust_cluster_id(cluster_id)
        cluster_ids_history.append(modified_cluster_id.clone().detach())

        # -------------------------------
        # 5) Decide whether to continue
        #    Check if "no new buffer" => all predicted type = 0, for example
        # -------------------------------
        buffer_type_pred = torch.argmax(buffer_type_pred, dim=-1)  # shape [batch_size]
        # A simple check: if all predicted types == 0
        if buffer_type_pred.eq(0).all() and not buffer_type_labels.eq(0).all():
            still_generating = True
            num_level += 1
            current_level = max_level - num_level - 1
            next_input_feats = inputs[current_level]["input_feats"]
            next_loc_feats = inputs[current_level]["loc_feats"]
            next_elc_feats = inputs[current_level]["elc_feats"]

            current_input_feats = next_input_feats.to(device)
            current_loc_feats = next_loc_feats.to(device)
            current_elc_feats = next_elc_feats.to(device)

            pred_features.append(next_input_feats)
        elif num_level + 1 >= max_level:
            still_generating = False
        else:
            num_level += 1

            # -------------------------------
            # 6) Teacher Forcing 
            # -------------------------------
            current_level = max_level - num_level - 1
            use_ground_truth = (random.random() < teacher_forcing_prob)
            use_ground_truth = True

            # Use ground truth to update input
            next_input_feats = inputs[current_level]["input_feats"]
            next_loc_feats = inputs[current_level]["loc_feats"]
            next_elc_feats = inputs[current_level]["elc_feats"]
            pred_features.append(next_input_feats)

            # Update for next iteration
            current_input_feats = next_input_feats.to(device)
            current_loc_feats = next_loc_feats.to(device)
            current_elc_feats = next_elc_feats.to(device)

    return (all_buffers, pred_features, local_loss_sum, cluster_loss_sum,
            type_loss_sum, location_loss_sum, area_penalty_sum, total_area,
            driver_wirelength, driver_output_cap, cap_penalty_sum, wire_penalty_sum)


def train_one_epoch(model, train_loader, optimizer, epoch, args, device, penalty_flag_list, weight_list):
    """
    Train the model for one epoch using teacher forcing + scheduled sampling.
    """
    model.train()

    # Decay teacher_forcing_prob from 1.0 to some lower value over training
    teacher_forcing_prob = max(args.scheduled_sampling_min,
                               args.scheduled_sampling_max *
                               np.exp(-args.scheduled_sampling_decay * epoch))


    total_penalty_loss = 0.0
    batch_num = 0
    total_loss_sum = 0
    cluster_loss_sum = 0
    location_loss_sum = 0
    type_loss_sum = 0
    area_penalty_sum = 0
    global_total_area_penalty = 0
    total_wire_penalty = 0
    total_cap_penalty = 0
    for batch_data in train_loader:
        if batch_data is None:
            continue
        # batch_data is a list of size `batch_size`; each element is (layers_data, net_name, file_key)
        layers_data, net_name, file_key, _ = batch_data[0]
        # print(f"File={file_key}, Net={net_name}, #Layers={len(layers_data)}")
        batch_num += 1

        # -------------------------------
        # 1) Recursive training per batch
        # -------------------------------
        (all_buffers, pred_feature_list, local_loss_sum, cluster_loss, type_loss,
         location_loss, area_penalty, total_area, driver_wirelength, driver_output_cap, buffer_cap_penalty,
         buffer_wire_penalty) = recursive_training(
            model,
            layers_data,
            optimizer,
            teacher_forcing_prob,
            device,
            buf_info_df,
            buf_type_map,
            max_wirelength_constraints,
            penalty_flag_list,
            epoch,
            weight_list
        )

        # -------------------------------
        # 2) Outer-loop penalties
        #    Impose global constraints (penalties)
        #    on the entire constructed tree
        # -------------------------------

        # ------ Area penalty
        global_area_penalty = F.relu(total_area-0.08)
  
        # ------ Wirelength penalty
        driver_wire_pen = F.relu(driver_wirelength - max_wirelength_constraints)
        # ------ Cap penalty
        driver_cap_pen = F.relu(driver_output_cap - pred_feature_list[0][0, 10])


        # total_loss = area_penalty + local_loss_sum
        total_loss = local_loss_sum + penalty_flag_list[
            1] * driver_wire_pen + penalty_flag_list[2] * driver_cap_pen + weight_list[3] * penalty_flag_list[
                         0] * global_area_penalty

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # total_loss_np = total_loss.detach.numpy()
        total_loss_sum += total_loss.item()
        cluster_loss_sum += cluster_loss
        type_loss_sum += type_loss
        location_loss_sum += location_loss
        area_penalty_sum += area_penalty  # layer
        global_total_area_penalty += global_area_penalty.item()  # global
        wire_penalty = driver_wire_pen.item() + buffer_wire_penalty
        cap_penalty = driver_cap_pen.item() + buffer_cap_penalty
        total_wire_penalty += wire_penalty
        total_cap_penalty += cap_penalty

    # step the LR scheduler here (per epoch)
    scheduler.step()

    # average
    total_loss_avg = total_loss_sum / batch_num
    cluster_loss_avg = cluster_loss_sum / batch_num
    type_loss_avg = type_loss_sum / batch_num
    area_penalty_avg = area_penalty_sum / batch_num
    global_area_penalty_avg = global_total_area_penalty / batch_num
    location_loss_avg = location_loss_sum / batch_num
    wire_penalty_avg = total_wire_penalty / batch_num
    cap_penalty_avg = total_cap_penalty / batch_num
    print(f"Epoch {epoch} - Total Loss = {total_loss_avg:.4f}")
    print(f"Epoch {epoch} - Cluster Loss = {cluster_loss_avg:.4f}")
    print(f"Epoch {epoch} - Type Loss = {type_loss_avg:.4f}")
    print(f"Epoch {epoch} - Loc Loss = {location_loss_avg:.4f}")
    print(f"Epoch {epoch} - Layer Area Penalty = {area_penalty_avg:.4f}")
    print(f"Epoch {epoch} - Global Area Penalty = {global_area_penalty_avg:.4f}")
    print(f"Epoch {epoch} - Wire Penalty = {wire_penalty_avg:.4f}")
    print(f"Epoch {epoch} - Cap Penalty = {cap_penalty_avg:.4f}")

    # record loss
    epoch_losses["total_loss"].append(total_loss_avg)
    epoch_losses["cluster_loss"].append(cluster_loss_avg)
    epoch_losses["type_loss"].append(type_loss_avg)
    epoch_losses["location_loss"].append(location_loss_avg)
    # epoch_losses["layer_area_penalty"].append(area_penalty_avg)
    epoch_losses["global_area_penalty"].append(global_area_penalty_avg)
    epoch_losses["wire_penalty"].append(wire_penalty_avg)
    epoch_losses["cap_penalty"].append(cap_penalty_avg)


def test_model(model, val_dataloader, device, max_wirelength_constraints, penalty_flag_list, epoch, weight_list):
    """
    Evaluate the model on a test dataset in inference mode (no teacher forcing).
    You can still compute average losses if you keep track of them, but
    typically at test time you run pure inference.
    """
    model.eval()

    avg_total_loss, avg_cluster_loss, avg_type_loss, avg_location_loss = inference_for_testing(model,
                                                                                               val_dataloader,
                                                                                               device, buf_info_df,
                                                                                               buf_type_map,
                                                                                               max_wirelength_constraints,
                                                                                               penalty_flag_list, epoch,
                                                                                               weight_list)

    return avg_total_loss


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
    parser.add_argument('-output_dim_btype', type=int, default=6)  # warning:buftype=6
    parser.add_argument('-input_dim_share', type=int, default=12)
    parser.add_argument('-input_dim_loc', type=int, default=3)
    parser.add_argument('-input_dim_els', type=int, default=5)
    parser.add_argument('-max_clusters', type=int, default=20)

    # Scheduled Sampling parameters
    parser.add_argument('-scheduled_sampling_max', type=float, default=1.0,
                        help='Initial teacher forcing probability (use ground truth).')
    parser.add_argument('-scheduled_sampling_min', type=float, default=0.0,
                        help='Minimum teacher forcing probability.')
    parser.add_argument('-scheduled_sampling_decay', type=float, default=0.02,
                        help='Decay rate for teacher forcing probability each epoch.')

    args, _ = parser.parse_known_args()
    return args


if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args = get_args()
    user_defined_name = 'test'
    data_dirs = ['data/training_data/jpeg_train/', 'data/training_data/ariane133_train/', 'data/training_data/ibex/']
    buf_info_file = 'data/buf_data.csv'
    model_save_path = 'results/model_dict/mlbuf_' + user_defined_name + '.pt'
    plot_save_path = 'results/plot/'
    btree_save_path = 'results/btree_pred/'
    figure_name = 'mlbuf_loss_' + user_defined_name + '.png'
    btree_file_name = 'mlbuf_btree_' + user_defined_name + '.csv'
    loss_save_path = plot_save_path + 'mlbuf_loss_' + user_defined_name + '.csv'

    area_penalty_flag = 1
    wire_penalty_flag = 1
    cap_penalty_flag = 1  # 1 represents true, otherwise 0
    penalty_flag_list = [area_penalty_flag, wire_penalty_flag, cap_penalty_flag]

    cluster_loss_weight = 1
    type_loss_weight = 100000
    location_loss_weight = 1
    area_loss_weight = 0.001
    wire_loss_weight = 1
    cap_loss_weight = 1
    weight_list = [cluster_loss_weight, type_loss_weight, location_loss_weight, area_loss_weight, wire_loss_weight,
                   cap_loss_weight]

    # Load data
    buf_info_df = pd.read_csv(buf_info_file)
    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(
        data_dirs=data_dirs,
        buf_info_df=buf_info_df,
        batch_size=1,
        split_ratios=(0.5, 0.2, 0.3),
        seed=42
    )
    print("Finish data loading ...")


    buf_type_map = {
        0: 'None',
        1: 'BUF_X2',
        2: 'BUF_X4',
        3: 'BUF_X8',
        4: 'BUF_X16',
        5: 'BUF_X32'
    }

    dbu = 2000
    max_wirelength_constraints = torch.tensor(1449128 / (dbu * 1e+6), dtype=torch.float32, device=device)

    # Initialize model
    model = MLBuf(
        args.input_dim_share,
        args.input_dim_loc,
        args.input_dim_els,
        args.hidden_dim,
        args.num_heads,
        args.clustering_output_dim,
        args.max_clusters,
        args.output_dim_bloc,
        args.output_dim_btype,
        # drop_n=args.drop_n,
        # drop_c=args.drop_c,
        # act_n=args.act_n,
        # act_c=args.act_c
    ).to(device)

    # Optimizer
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # Use LR scheduler
    # Step down by factor=0.1 every 100 epochs
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

    best_loss = float('inf')
    patience_counter = 0
    patience_limit = 60  # stop if we do not improve for many epochs

    # record loss
    epoch_losses = {
        "total_loss": [],
        "cluster_loss": [],
        "type_loss": [],
        "location_loss": [],
        # "layer_area_penalty": [],
        "global_area_penalty": [],
        "wire_penalty": [],
        "cap_penalty": []
    }

    # Main training loop
    for epoch in range(args.num_epochs):
        # -------------------------
        # 1) Train one epoch
        # -------------------------
        train_one_epoch(model, train_dataloader, optimizer, epoch, args, device, penalty_flag_list, weight_list)

        # -------------------------
        # 2) Evaluate on test set
        # -------------------------
        test_loss = test_model(model, val_dataloader, device, max_wirelength_constraints, penalty_flag_list, epoch,
                               weight_list)
        print("Test Result:", test_loss, "\tepoch=", epoch)
        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"New best loss {best_loss:.4f}, model saved.")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience_limit:
                print("Early stopping triggered.")
                break

    # Convert epoch_losses dictionary to DataFrame and save as CSV
    loss_df = pd.DataFrame(epoch_losses)
    loss_df.to_csv(loss_save_path, index=False)
    print(f"Losses saved to {loss_save_path}")
    plot_utils.plot_losses(epoch_losses, plot_save_path + figure_name)

    print("Start inference ...")
    inference(model, model_save_path, test_dataloader, device, buf_info_df, buf_type_map, btree_save_path,
              btree_file_name)
