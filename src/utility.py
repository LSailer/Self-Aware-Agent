import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import torchvision.utils as vutils

def visualize_vae_reconstruction(originals, reconstructions, step, save_dir="logs/vae_reconstructions", num_examples=8):
    """
    Saves a comparison grid of original and reconstructed images from the VAE.

    Args:
        originals (torch.Tensor): Batch of original images (B, C, H, W), normalized [0, 1].
        reconstructions (torch.Tensor): Batch of reconstructed images (B, C, H, W), normalized [0, 1].
        step (int): The current simulation step, used for filename.
        save_dir (str): Directory to save the image grid.
        num_examples (int): Number of image pairs to display.
    """
    if originals.shape[0] == 0 or reconstructions.shape[0] == 0:
        print("Warning: No images provided for VAE visualization.")
        return

    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Limit number of examples to show, ensure it doesn't exceed batch size
    num_examples = min(num_examples, originals.shape[0], reconstructions.shape[0])
    if num_examples <= 0:
        return

    # Select examples and move to CPU
    originals = originals[:num_examples].cpu().detach()
    reconstructions = reconstructions[:num_examples].cpu().detach()

    # Create comparison tensor (stack originals and reconstructions)
    comparison = torch.cat([originals, reconstructions])

    # Create a grid of images
    grid = vutils.make_grid(comparison, nrow=num_examples, padding=2, normalize=False) # Already in [0,1]

    # Plot and save
    plt.figure(figsize=(num_examples * 2, 4)) # Adjust figure size as needed
    plt.imshow(np.transpose(grid.numpy(), (1, 2, 0))) # Convert (C, H, W) to (H, W, C)
    plt.title(f"VAE Reconstructions - Step {step}\nTop: Originals, Bottom: Reconstructions")
    plt.axis('off')
    save_path = os.path.join(save_dir, f"reconstruction_step_{step:06d}.png")
    try:
        plt.savefig(save_path)
        # print(f"VAE reconstruction visualization saved to {save_path}")
    except Exception as e:
        print(f"Error saving VAE visualization: {e}")
    plt.close()