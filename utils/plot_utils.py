import matplotlib.pyplot as plt
import matplotlib.pyplot as plt


def plot_losses(epoch_losses, save_path):
    """
    Plots the training losses over epochs in four subplots.

    Args:
        epoch_losses (dict): A dictionary containing lists of losses per epoch.
                             Keys should be 'total_loss', 'cluster_loss',
                             'type_loss', and 'location_loss'.
    """
    fig, axes = plt.subplots(3, 3, figsize=(18, 16))  # 2 rows, 2 cols

    loss_names = ["Total Loss", "Cluster Loss", "Type Loss", "Location Loss", 
                  "Global Area Penalty", "Wire Penalty",
                  "Cap Penalty"]
    loss_keys = ["total_loss", "cluster_loss", "type_loss", "location_loss", 
                 "global_area_penalty", "wire_penalty",
                 "cap_penalty"]

    for i, ax in enumerate(axes.flat[:len(loss_keys)]):  # Iterate over subplots
        ax.plot(epoch_losses[loss_keys[i]], marker="o", linestyle="-", label=loss_names[i])
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(loss_names[i])
        ax.legend()
        ax.grid(True)

    plt.tight_layout()  # Adjust spacing
    # Save the figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Loss plot saved as {save_path}")

    plt.show()
