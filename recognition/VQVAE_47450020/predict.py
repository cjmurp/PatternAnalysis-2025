"""
Usage of model. Useful visualisations.
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torchmetrics.functional import structural_similarity_index_measure as ssim
import os
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from modules import VQVAE
from dataset import HipMRIProstateDataset

path = "./../../../keras_slices_data/"
# path = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data"

def evaluate_vqvae(model, test_loader, device, save_dir="results", num_visuals=5):
    """
    Evaluate a trained VQ-VAE model on a test dataset.

    Parameters
    ----------
    model : torch.nn.Module
        The trained VQ-VAE model.
    test_loader : torch.utils.data.DataLoader
        DataLoader containing test data.
    device : torch.device
        CPU or CUDA device.
    save_dir : str
        Directory where visualizations and plots will be saved.
    num_visuals : int
        Number of reconstruction examples to visualize.
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    total_loss, total_recon, total_vq, total_commit, total_ssim = 0, 0, 0, 0, 0
    ssim_values, recon_losses = [], []

    with torch.no_grad():
        for batch in test_loader:
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            batch = batch.float().to(device)
            if batch.dim() == 3:
                batch = batch.unsqueeze(0)

            # Forward pass
            x_recon, total_l, logs, _ = model(batch)
            recon_loss = logs['recon_loss']
            vq_loss = logs['vq_loss']
            commit_loss = logs['commitment_loss']

            # Clamp and compute SSIM
            x_recon_clamped = torch.clamp(x_recon, -1, 1)
            batch_clamped = torch.clamp(batch, -1, 1)

            x_recon_clamped = torch.clamp(x_recon, -1, 1)
            batch_clamped = torch.clamp(batch, -1, 1)
            ssim_val = ssim(x_recon_clamped, batch_clamped, data_range=1.0)

            total_loss += total_l.item()
            total_recon += recon_loss.item()
            total_vq += vq_loss.item()
            total_commit += commit_loss.item()
            total_ssim += ssim_val
            ssim_values.append(ssim_val.item())
            recon_losses.append(recon_loss.item())

    n = len(test_loader)
    mean_loss = total_loss / n
    mean_ssim = total_ssim / n
    min_ssim = np.min(ssim_values)
    max_ssim = np.max(ssim_values)

    print(f"\nTest Results:")
    print(f"  Avg Total Loss   : {mean_loss:.4f}")
    print(f"  Avg Recon Loss   : {total_recon / n:.4f}")
    print(f"  Avg VQ Loss      : {total_vq / n:.4f}")
    print(f"  Avg Commit Loss  : {total_commit / n:.4f}")
    print(f"  Avg SSIM         : {mean_ssim:.4f}")
    print(f"  Min SSIM         : {min_ssim:.4f}")
    print(f"  Max SSIM         : {max_ssim:.4f}")

    # === Plot SSIM over batches ===
    fig1, ax1 = plt.subplots()
    ax1.plot(ssim_values, label="SSIM per batch")
    ax1.set_xlabel("Batch")
    ax1.set_ylabel("SSIM")
    ax1.set_title("SSIM Scores Across Test Set")
    ax1.grid(True)
    ax1.legend()
    fig1.savefig(os.path.join(save_dir, "test_ssim_curve.png"), dpi=300, bbox_inches='tight')
    plt.close(fig1)

    # === Plot Reconstruction Loss over batches ===
    fig2, ax2 = plt.subplots()
    ax2.plot(recon_losses, label="Reconstruction Loss", color='orange')
    ax2.set_xlabel("Batch")
    ax2.set_ylabel("Loss")
    ax2.set_title("Reconstruction Loss Across Test Set")
    ax2.grid(True)
    ax2.legend()
    fig2.savefig(os.path.join(save_dir, "test_recon_curve.png"), dpi=300, bbox_inches='tight')
    plt.close(fig2)

    # === Visualize some reconstructions ===
    visualize_reconstructions(model, test_loader, device, save_dir, num_visuals)

    return {
        "mean_loss": mean_loss,
        "mean_ssim": mean_ssim,
        "ssim_values": ssim_values,
        "recon_losses": recon_losses
    }


def visualize_reconstructions(model, dataloader, device, save_dir, num_visuals=5):
    """
    Display and save example reconstructions from the test set.
    """
    batch = next(iter(dataloader)).float().to(device)
    with torch.no_grad():
        x_recon, _, _, _ = model(batch)

    x_recon = x_recon.cpu().numpy()
    batch = batch.cpu().numpy()

    for i in range(min(num_visuals, batch.shape[0])):
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(batch[i, 0], cmap='gray')
        axes[0].set_title("Original")
        axes[0].axis("off")

        axes[1].imshow(x_recon[i, 0], cmap='gray')
        axes[1].set_title("Reconstruction")
        axes[1].axis("off")

        fig.savefig(os.path.join(save_dir, f"reconstruction_{i}.png"), dpi=300, bbox_inches='tight')
        plt.close(fig)


def visualize_codebook_usage(model, dataloader, device, save_dir=None, num_visuals=4):
    """
    Visualize which codebook embeddings are used for each patch in the quantized image.

    Parameters
    ----------
    model : nn.Module
        Trained VQ-VAE model
    dataloader : DataLoader
        DataLoader with test or validation samples
    device : torch.device
        Device to run on ('cuda' or 'cpu')
    save_dir : str, optional
        Directory to save images. If None, just show them.
    num_visuals : int
        Number of images from the batch to visualize
    """

    model.eval()
    batch = next(iter(dataloader)).float().to(device)

    with torch.no_grad():
        # Forward pass through encoder & quantizer only
        z_e = model.encoder(batch)
        z_e = model.pre_vq_conv(z_e)
        quantized, _, _, _, _, encoding_indices = model.vq(z_e)
        zq = model.post_vq_conv(quantized)
        # Decode to reconstruction
        x_recon = model.decoder(zq)

    encoding_indices = encoding_indices.detach().cpu().numpy()
    batch_np = batch.cpu().numpy()
    x_recon_np = x_recon.cpu().numpy()

    for i in range(min(num_visuals, batch_np.shape[0])):
        idx_map = encoding_indices[i]  # shape [H', W']

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(batch_np[i, 0], cmap="gray")
        axes[0].set_title("Original")
        axes[0].axis("off")

        axes[1].imshow(x_recon_np[i, 0], cmap="gray")
        axes[1].set_title("Reconstruction")
        axes[1].axis("off")

        # Show encoding index map as a color-coded patch grid
        im = axes[2].imshow(idx_map, cmap="viridis")
        axes[2].set_title("Codebook Index Map")
        axes[2].axis("off")
        fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

        plt.tight_layout()

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"codebook_map_{i}.png")
            plt.savefig(save_path, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved codebook visualization to {save_path}")
        else:
            plt.show()


if __name__ == "__main__":

    model_path = "vqvae_checkpoints/vqvae_epoch_20.pt"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = torch.load(model_path, map_location=device, weights_only=False)
    model.eval()

    test = "keras_slices_test"

    test_path = os.path.join(path, test)

    transform = transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    data = HipMRIProstateDataset(root_dir=test_path, transform=transform)
    test_loader = DataLoader(data, batch_size=1, shuffle=True)

    results = evaluate_vqvae(
        model=model,
        test_loader=test_loader,
        device=device,
        save_dir="outputs/test/test_eval",
        num_visuals=6
    )

    visualize_codebook_usage(model, test_loader, device, save_dir="outputs/test/codebook_vis", num_visuals=4)


