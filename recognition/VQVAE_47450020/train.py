"""
Used for training, validating, testing and saving the model. Uses the model described
in modules.py and the data in dataset.py.

"""
import os
from modules import VQVAE
from dataset import HipMRIProstateDataset
import torch.nn as nn
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import tempfile
from torchmetrics.functional import structural_similarity_index_measure as ssim

# path = "./../../../keras_slices_data/"
path = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data"

def save_plot(fig, save_dir, filename="plot.png"):
    """
    Save a matplotlib figure to a specified directory.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure object to save.
    save_dir : str
        Directory path where the image will be saved.
    filename : str
        Name of the saved file (default "plot.png").
    """
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)
    fig.savefig(filepath, bbox_inches='tight', dpi=300)
    plt.close(fig)  # Close the figure to free memory

def validate_vqvae(model, dataloader, device):
    model.eval()
    total_t_loss, total_recon, total_vq, total_commit = 0.0, 0.0, 0.0, 0.0
    ssim_scores = []
    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            batch = batch.float().to(device)
            if batch.dim() == 3:
                batch = batch.unsqueeze(0)

            x_recon, total_loss, logs, _ = model(batch)

            recon_loss = logs['recon_loss']
            vq_loss = logs['vq_loss']
            commit_loss = logs['commitment_loss']

            total_recon += recon_loss.item()
            total_vq += vq_loss.item()
            total_commit += commit_loss.item()
            total_t_loss += recon_loss.item()

            # --- SSIM Calculation ---
            # Ensure images are normalized to [0, 1]
            x_recon_clamped = torch.clamp(x_recon, -1, 1)
            batch_clamped = torch.clamp(batch, -1, 1)
            ssim_val = ssim(x_recon_clamped, batch_clamped, data_range=1.0)
            ssim_scores.append(ssim_val.item())

    n = len(dataloader)
    return total_t_loss / n, total_recon / n, total_vq / n, total_commit / n, sum(ssim_scores) / len(ssim_scores)


def train_for_one_epoch(epoch_idx, model, dataloader, optimizer, device, visualize=False,
                        num_visuals=3):
    """
    Train VQVAE for one epoch.

    Parameters
    ----------
    epoch_idx : int
        Current epoch index
    model : nn.Module
        VQVAE model
    dataloader : DataLoader
        PyTorch dataloader yielding (image, label) tuples (label ignored)
    optimizer : torch.optim.Optimizer
        Optimizer for model
    criterion : loss function
        Typically nn.MSELoss()
    config : dict
        Dictionary with weights: 'RECON_LOSS_WEIGHT', 'CODEBOOK_LOSS_WEIGHT', 'COMMITMENT_LOSS_WEIGHT'
    device : torch.device
        CUDA or CPU
    visualize : bool
        If True, plot input vs reconstructed images
    num_visuals : int
        Number of images to visualize per epoch
    """

    model.train()
    recon_losses = []
    codebook_losses = []
    commitment_losses = []
    total_losses = []
    embedding_losses = []

    ssim_scores = []

    pbar = tqdm(dataloader, desc=f"Epoch {epoch_idx + 1} Training", leave=False)

    for im in pbar:
        im = im.float().to(device)
        optimizer.zero_grad()

        # Forward pass
        x_recon, total_loss, logs, _ = model(im)

        # Individual losses
        recon_loss = logs['recon_loss']
        vq_loss = logs['vq_loss']
        embedding_loss = logs['embedding_loss']
        commitment_loss = logs['commitment_loss']

        # Backprop
        total_loss.backward()
        optimizer.step()

        # Calculate SSIM
        with torch.no_grad():
            # Ensure both are in [0,1]
            ssim_val = ssim(x_recon, im, data_range=1.0)
            ssim_scores.append(ssim_val.item())

        # Track metrics
        recon_losses.append(recon_loss.item())
        codebook_losses.append(vq_loss.item())
        embedding_losses.append(embedding_loss.item())
        commitment_losses.append(commitment_loss.item())
        total_losses.append(total_loss.item())

        pbar.set_postfix({
            "Total": f"{total_loss.item():.4f}",
            "Recon": f"{recon_loss.item():.4f}",
            "Codebook": f"{vq_loss.item():.4f}"
        })

    # Summary
    mean_total = np.mean(total_losses)
    ssim_val = np.mean(ssim_scores)
    print(f"Finished Epoch {epoch_idx + 1} | Recon: {np.mean(recon_losses):.4f} | "
          f"Codebook: {np.mean(codebook_losses):.4f} | Commitment: {np.mean(commitment_losses):.4f} | SSIM: {ssim_val:.4f}")

    # Visualization
    if visualize:
        model.eval()
        with torch.no_grad():
            batch = next(iter(dataloader)).float().to(device)
            x_recon, _, _, _ = model(batch)
            batch = batch.cpu().numpy()
            x_recon = x_recon.cpu().numpy()

            # Display first `num_visuals` images
            for i in range(min(num_visuals, batch.shape[0])):
                fig, axes = plt.subplots(1, 2, figsize=(8, 4))
                axes[0].imshow(batch[i, 0], cmap='gray')
                axes[0].set_title('Original')
                axes[0].axis('off')

                axes[1].imshow(x_recon[i, 0], cmap='gray')
                axes[1].set_title('Reconstruction')
                axes[1].axis('off')

                plt.show()
                save_plot(fig, save_dir=f"outputs/train/epoch_{epoch_idx+1}", filename=f"recon_imag_{i}.png")

    return mean_total, ssim_val

def safe_torch_save(model_state, path):
    if os.path.exists(path):
        os.remove(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    dir_ = os.path.dirname(path)
    with tempfile.NamedTemporaryFile(delete=False, dir=dir_) as tmp:
        torch.save(model_state, tmp.name)
        tmp.flush()
        os.fsync(tmp.fileno())
    os.replace(tmp.name, path)

def train_vqvae(model, dataloader, optimizer, device, val_loader, num_epochs=20,
                visualize_every=5, checkpoint_dir="checkpoints"):
    """
    Train VQVAE over multiple epochs.

    Parameters
    ----------
    model : nn.Module
        VQVAE model
    dataloader : DataLoader
        Dataset loader
    optimizer : torch.optim.Optimizer
        Optimizer
    criterion : loss function
        Typically nn.MSELoss()
    config : dict
        Dictionary with weights for reconstruction, codebook, commitment losses
    device : torch.device
        CPU or CUDA
    num_epochs : int
        Total number of epochs
    visualize_every : int
        Frequency (in epochs) to visualize reconstructions
    checkpoint_dir : str
        Directory to save model checkpoints
    """

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    epoch_losses = []
    ssim_scores = []

    val_epoch_losses = []
    val_ssim_scores = []


    for epoch in range(num_epochs):
        # Decide whether to visualize this epoch
        visualize = (epoch % visualize_every == 0)
        # visualize = False
        # Train one epoch
        mean_loss, ssim_val = train_for_one_epoch(epoch, model, dataloader, optimizer, device,
                                        visualize=visualize)
        epoch_losses.append(mean_loss)
        ssim_scores.append(ssim_val)

        val_total_loss, val_recon, val_vq, val_commit, val_ssim = validate_vqvae(model, val_loader, device)

        print(f"Epoch {epoch + 1}/{num_epochs} | "
              f"Train Loss={mean_loss:.4f} | "
              f"Val Total={val_total_loss:.4f} Val Recon={val_recon:.4f} | Val VQ={val_vq:.4f} | Val Commit={val_commit:.4f} | Val ssim={val_ssim:.4f} ")

        val_epoch_losses.append(val_total_loss)
        val_ssim_scores.append(val_ssim)

        # Save checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f"vqvae_epoch_{epoch + 1}.pt")
        try:
            safe_torch_save(model, checkpoint_path)
        except Exception as e:
            print(f"Warning: Failed to save checkpoint ({e})")
        print(f"Saved checkpoint to {checkpoint_path}")

    # --- Plot: Training vs Validation Loss ---
    fig1 = plt.figure(figsize=(7, 5))
    plt.plot(range(1, num_epochs + 1), epoch_losses, 'o-', label='Train Loss')
    plt.plot(range(1, num_epochs + 1), val_epoch_losses, 's--', label='Validation Loss')
    plt.title("Training vs Validation Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # --- Plot: Training vs Validation SSIM ---
    fig2 = plt.figure(figsize=(7, 5))
    plt.plot(range(1, num_epochs + 1), ssim_scores, 'o-', label='Train SSIM')
    plt.plot(range(1, num_epochs + 1), val_ssim_scores, 's--', label='Validation SSIM')
    plt.title("Training vs Validation SSIM per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("SSIM")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    save_plot(fig1, save_dir="outputs/plots", filename="vqvae_epoch_losses.png")
    save_plot(fig2, save_dir="outputs/plots", filename="ssim_epoch_losses.png")

    return epoch_losses


if __name__ == "__main__":

    model = VQVAE(
            image_channels=1,
            hidden_channels=128,
            embedding_dim=128,
            num_embeddings=512,
            n_down=3,
            commitment_cost=0.25,
            use_ema=False,
            use_batchnorm=False,
            activation=nn.ReLU,
            kernel_size=4,
            num_residual_blocks=1,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    test = "keras_slices_train"
    validate = "keras_slices_validate"

    test_path = os.path.join(path, test)
    validate_path = os.path.join(path, validate)

    transform = transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    data = HipMRIProstateDataset(root_dir=test_path, transform=transform)
    val_data = HipMRIProstateDataset(root_dir=validate_path, transform=transform)

    dataloader = DataLoader(data, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=32, shuffle=False)

    num_epochs = 20
    epoch_losses = train_vqvae(model, dataloader, optimizer, device, val_dataloader,
                               num_epochs=num_epochs, visualize_every=1,
                               checkpoint_dir="vqvae_checkpoints")