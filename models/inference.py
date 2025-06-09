import pandas as pd
import torch
from torch.optim import Adam
from models.model import MLBuf
from models.losses import *
import utils.util as util


def inference(model, model_save_path, test_loader, device, buf_info_df, buf_type_map, btree_save_path, btree_file_name):
    """
    Inference (recursive) without teacher forcing:
    The model's own predictions feed into the next level.
    """
    model.load_state_dict(torch.load(model_save_path, ))
    model.eval()
    for batch_data in test_loader:
        if batch_data is None:
            continue
        layers_data, net_name, file_key, _ = batch_data[0]
        print(f"File={file_key}, Net={net_name}, #Layers={len(layers_data)}")

        with torch.no_grad():
            num_level = 0
            max_level = len(layers_data)
            current_level = max_level - num_level - 1
            input_features = layers_data[current_level]['input_feats'].to(device)
            loc_features = layers_data[current_level]["loc_feats"].to(device)
            elc_features = layers_data[current_level]["elc_feats"].to(device)

            all_buffers_all_type = []
            cluster_ids_history = []
            pred_features = []
            pred_features.append(input_features)

            still_generating = True
            while still_generating:
                buffer_type_logits, buffer_location_logits, cluster_id, cluster_probs, cluster_embed, _, _ = model(
                    input_features, loc_features, elc_features
                )

                buffer_type_pred = torch.argmax(buffer_type_logits, dim=-1)
                if current_level > 0:
                    buffer_type_labels = layers_data[current_level]["buffer_type_label"]

                # Store predictions
                all_buffers_all_type.append(buffer_type_pred)
                modified_cluster_id = util.adjust_cluster_id(cluster_id)
                if current_level > 0:
                    cluster_label = layers_data[current_level]["cluster_label"]
                    print("cluster label: ", cluster_label)
                cluster_ids_history.append(modified_cluster_id.clone().detach())

                if buffer_type_pred.eq(0).all():
                    still_generating = False
                # if num_level > max_level:
                #     still_generating = False
                else:
                    num_level += 1
                    current_level = max_level - num_level - 1
                    print("[Inference] Still generating... [Level]: ", num_level)
                    # Adjust input using predictions (no ground truth at inference)
                    input_features, loc_features, elc_features, input_features_buf = util.adjust_model_output(
                        input_features,
                        loc_features,
                        elc_features,
                        buffer_location_logits,
                        buffer_type_logits,
                        cluster_id,
                        buf_info_df,
                        buf_type_map
                    )
                    pred_features.append(input_features_buf)
                    input_features = input_features.to(device).float()
                    loc_features = loc_features.to(device).float()
                    elc_features = elc_features.to(device).float()
            # Ensure the length of all_buffer and pred_features is the same
            if len(all_buffers_all_type) == 0:
                dummy_ids = torch.zeros(pred_features[0].size(0), dtype=torch.long)
                all_buffers_all_type = [dummy_ids]
            if len(all_buffers_all_type) < len(pred_features):
                all_buffers_all_type.append(torch.zeros(pred_features[-1].size(0), dtype=torch.long))
            buffer_tree = util.build_tree_bottom_up(pred_features, all_buffers_all_type, buf_info_df, buf_type_map,
                                                    cluster_ids_history, net_name, file_key,
                                                    out_file=btree_save_path + btree_file_name, print_tree=False, autoBuf=True)


def inference_for_testing(model, test_loader, device, buf_info_df, buf_type_map, max_wirelength_constraints,
                          penalty_flag_list, epoch, weight_list):
    from train import recursive_training
    """
    Inference (recursive) model for testing:
    The model's own predictions feed into the next level.
    """
    model.eval()
    total_loss_sum = 0.0
    cluster_loss_sum = 0.0
    type_loss_sum = 0.0
    location_loss_sum = 0.0
    num_batches = 0
    for batch_data in test_loader:
        if batch_data is None:
            continue
        layers_data, net_name, file_key, _ = batch_data[0]
        # print(f"File={file_key}, Net={net_name}, #Layers={len(layers_data)}")

        with torch.no_grad():
            for lvl in range(len(layers_data)):
                for key, val in layers_data[lvl].items():
                    if isinstance(val, torch.Tensor):
                        layers_data[lvl][key] = val.to(device)

            (all_buffers, buffer_tree, pred_features, local_loss_sum, cluster_loss,
             type_loss, location_loss, area_penalty_sum, total_area,
             driver_wirelength, driver_output_cap, cap_penalty_sum, wire_penalty_sum) = recursive_training(
                model,
                layers_data,
                optimizer=None,
                teacher_forcing_prob=0.0,
                device=device,
                buf_info_df=buf_info_df,
                buf_type_map=buf_type_map,
                max_wirelength_constraints=max_wirelength_constraints,
                penalty_flag_list=penalty_flag_list,
                epoch=epoch,
                weight_list=weight_list
            )
            all_loss = local_loss_sum.item()
            total_loss_sum += all_loss
            cluster_loss_sum += cluster_loss
            type_loss_sum += type_loss
            location_loss_sum += location_loss

            num_batches += 1

    if num_batches > 0:
        avg_total_loss = total_loss_sum / num_batches
        avg_cluster_loss = cluster_loss_sum / num_batches
        avg_type_loss = type_loss_sum / num_batches
        avg_location_loss = location_loss_sum / num_batches
    else:
        avg_total_loss = 0.0
        avg_cluster_loss = 0.0
        avg_type_loss = 0.0
        avg_location_loss = 0.0

    print(f"[Test] Avg Total Loss: {avg_total_loss:.4f}")
    print(f"[Test] Avg Cluster Loss: {avg_cluster_loss:.4f}")
    print(f"[Test] Avg Type Loss: {avg_type_loss:.4f}")
    print(f"[Test] Avg Loc Loss: {avg_location_loss:.4f}")

    return avg_total_loss, avg_cluster_loss, avg_type_loss, avg_location_loss
