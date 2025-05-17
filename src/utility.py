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
                                  Expected to be on CPU.
        reconstructions (torch.Tensor): Batch of reconstructed images (B, C, H, W), normalized [0, 1].
                                        Expected to be on CPU.
        step (int): The current simulation step, used for filename.
        save_dir (str): Directory to save the image grid.
        num_examples (int): Number of image pairs to display.
    """
    if originals.shape[0] == 0 or reconstructions.shape[0] == 0:
        print("Warning: No images provided for VAE visualization.")
        return

    os.makedirs(save_dir, exist_ok=True)
    num_examples_to_show = min(num_examples, originals.shape[0], reconstructions.shape[0])
    if num_examples_to_show <= 0:
        print("Warning: num_examples_to_show is zero or negative in visualize_vae_reconstruction.")
        return

    originals_subset       = originals[:num_examples_to_show]
    reconstructions_subset = reconstructions[:num_examples_to_show]
    comparison = torch.cat([originals_subset, reconstructions_subset])

    try:
        grid = vutils.make_grid(comparison, nrow=num_examples_to_show, padding=2, normalize=False)
    except Exception as e:
        print(f"Error creating VAE visualization grid with vutils.make_grid: {e}")
        return

    plt.figure(figsize=(max(10, num_examples_to_show * 1.5), 4))
    try:
        # hier das .cpu() vor .numpy()
        plt.imshow(np.transpose(grid.cpu().numpy(), (1, 2, 0)))
    except Exception as e:
        print(f"Error during plt.imshow for VAE visualization: {e}")
        plt.close()
        return
        
    plt.title(f"VAE Reconstructions - Step {step}\nTop: Originals, Bottom: VAE Reconstructions")
    plt.axis('off')
    save_path = os.path.join(save_dir, f"vae_reconstruction_step_{step:06d}.png")
    try:
        plt.savefig(save_path)
    except Exception as e:
        print(f"Error saving VAE visualization to {save_path}: {e}")
    finally:
        plt.close()

def visualize_rnn_prediction(actual_next_frames, rnn_predicted_latent_z, vae_decode_function, 
                             step, agent_id, save_dir="logs/rnn_predictions", num_examples=8):
    """
    Visualizes the RNN's prediction of the next frame by decoding its predicted latent state.

    Args:
        actual_next_frames (torch.Tensor): Batch of actual next frames (B, C, H, W), normalized [0,1].
                                           Expected to be on CPU. These are the ground truth o_{t+1}.
        rnn_predicted_latent_z (torch.Tensor): Batch of latent states z_{t+1} predicted by RNN (B, LatentDim).
                                               Expected to be on CPU.
        vae_decode_function (callable): The VAE's decode method (e.g., vae.decode).
        step (int): Current simulation step.
        agent_id (int): The ID of the agent for whom the prediction is being visualized.
        save_dir (str): Directory to save the image grid.
        num_examples (int): Number of image pairs to display.
    """
    import os
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    import torchvision.utils as vutils

    if actual_next_frames.shape[0] == 0 or rnn_predicted_latent_z.shape[0] == 0:
        print(f"Warning: No images or latent codes provided for RNN prediction visualization for Agent {agent_id}.")
        return

    os.makedirs(save_dir, exist_ok=True)
    
    num_examples_to_show = min(num_examples, actual_next_frames.shape[0], rnn_predicted_latent_z.shape[0])
    if num_examples_to_show <= 0:
        print(f"Warning: num_examples_to_show is zero for RNN prediction viz for Agent {agent_id}.")
        return

    actual_frames_subset     = actual_next_frames[:num_examples_to_show]
    predicted_latents_subset = rnn_predicted_latent_z[:num_examples_to_show]

    # Decode the RNN's predicted latent states using the VAE's decoder
    with torch.no_grad():
        try:
            decoded_rnn_predictions = vae_decode_function(predicted_latents_subset)
        except Exception as e:
            print(f"Error decoding RNN predicted latent states for Agent {agent_id}: {e}")
            print(f"  Latent shape: {predicted_latents_subset.shape}, dtype: {predicted_latents_subset.dtype}")
            return

    # ─── Ensure both tensors live on the same device ───
    target_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    actual_frames_subset    = actual_frames_subset.to(target_device)
    decoded_rnn_predictions = decoded_rnn_predictions.to(target_device)
    # ───────────────────────────────────────────────────

    if decoded_rnn_predictions.shape != actual_frames_subset.shape:
        print(f"Warning: Shape mismatch for Agent {agent_id} RNN viz. "
              f"Actuals: {actual_frames_subset.shape}, Decoded Preds: {decoded_rnn_predictions.shape}")
        return

    # Create comparison tensor: [Actual Next Frames, Decoded RNN Predicted Next Frames]
    comparison = torch.cat([actual_frames_subset, decoded_rnn_predictions], dim=0)

    try:
        grid = vutils.make_grid(comparison.cpu(), nrow=num_examples_to_show, padding=2, normalize=False)
    except Exception as e:
        print(f"Error creating RNN prediction visualization grid for Agent {agent_id}: {e}")
        return

    plt.figure(figsize=(max(10, num_examples_to_show * 1.5), 4))
    try:
        plt.imshow(np.transpose(grid.numpy(), (1, 2, 0)))
    except Exception as e:
        print(f"Error during plt.imshow for RNN prediction visualization for Agent {agent_id}: {e}")
        plt.close()
        return
        
    plt.title(f"Agent {agent_id} - RNN Prediction vs Actual Next Frame - Step {step}\n"
              "Top: Actual Next, Bottom: RNN Predicted (decoded)")
    plt.axis('off')
    
    agent_save_dir = os.path.join(save_dir, f"agent_{agent_id}")
    os.makedirs(agent_save_dir, exist_ok=True)
    save_path = os.path.join(agent_save_dir, f"rnn_pred_step_{step:06d}_agent_{agent_id}.png")
    
    try:
        plt.savefig(save_path)
    except Exception as e:
        print(f"Error saving RNN prediction visualization for Agent {agent_id} to {save_path}: {e}")
    finally:
        plt.close()
