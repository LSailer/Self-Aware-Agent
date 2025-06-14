from enum import Enum
import os
import torch
import torch.nn.functional as F
import numpy as np
from models import VAE, RNNModel, SelfModel # Import new models
import matplotlib.pyplot as plt
from torchvision import transforms as T
from collections import deque, namedtuple
import random
import math
from utility import visualize_rnn_prediction, visualize_vae_reconstruction

# Define the Experience tuple for the replay buffer
Experience = namedtuple('Experience',
                        ('raw_image', 'action_key', 'action_array', 'reward', 'next_raw_image', 'done'))

#ENUMs for action selection
class ActionSelection(Enum):
    EPSILON_GREEDY = "epsilon_greedy"
    BOLTZMANN = "boltzmann"
    UCB = "ucb"

class CuriosityDrivenAgent:
    def __init__(self, actions, latent_dim=32, rnn_hidden_dim=256, buffer_size=10000, batch_size=64, device='cpu'):
        """
        Agent implementing VAE, RNN, and Self Model for curiosity-driven exploration.

        Args:
            actions (dict): Mappings from action keys (str) to action vectors (list/np.array).
            latent_dim (int): Dimensionality of the VAE latent space z.
            rnn_hidden_dim (int): Hidden dimension for the RNN model.
            buffer_size (int): Maximum size of the replay buffer.
            batch_size (int): Size of batches sampled from the replay buffer for training.
            device (str): 'cpu' or 'cuda'.
        """
        self.actions = actions
        self.action_dim = len(list(actions.values())[0])
        self.latent_dim = latent_dim
        self.rnn_hidden_dim = rnn_hidden_dim
        self.device = device
        self.batch_size = batch_size

        # --- Initialize Models ---
        self.vae = VAE(latent_dim=latent_dim).to(self.device)
        self.rnn_model = RNNModel(latent_dim=latent_dim, action_dim=self.action_dim, rnn_hidden_dim=rnn_hidden_dim).to(self.device)
        self.self_model = SelfModel(latent_dim=latent_dim, rnn_hidden_dim=rnn_hidden_dim, action_dim=self.action_dim).to(self.device)

        # --- Initialize Optimizers ---
        self.optimizer_vae = torch.optim.Adam(self.vae.parameters(), lr=0.001)
        self.optimizer_rnn = torch.optim.Adam(self.rnn_model.parameters(), lr=0.001)
        self.optimizer_self = torch.optim.Adam(self.self_model.parameters(), lr=0.001)

        # --- Replay Buffer ---
        self.replay_buffer = deque(maxlen=buffer_size)

        # --- Image Transformation ---
        # Transformation pipeline for VAE input/output
        self.transform = T.Compose([
            T.ToPILImage(),         # Convert numpy array (H, W, C) to PIL Image
            T.Resize((64, 64)),     # Resize to 64x64
            T.ToTensor(),           # Convert to tensor and normalize to [0, 1]
        ])

        # --- Agent State ---
        self.current_z = None       # Current latent state z_t
        self.current_h = None       # Current RNN hidden state (h_t, c_t) - Tuple

    def _preprocess_image(self, raw_image):
        """ Preprocess a single raw image for VAE """
        # Add batch dimension, apply transform, move to device
        return self.transform(raw_image).unsqueeze(0).to(self.device) # (1, C, H, W)

    def encode_image(self, raw_image):
        """ Encodes a raw image into latent state z using the VAE """
        processed_image = self._preprocess_image(raw_image)
        with torch.no_grad(): # No need to track gradients during encoding for action selection/storage
            mu, logvar = self.vae.encode(processed_image)
            z = self.vae.reparameterize(mu, logvar)
        return z # Shape: (1, LatentDim)

    def reset_hidden_state(self):
        """ Resets the RNN hidden state (at the start of an episode) """
        # Initialize hidden state for a batch size of 1
        self.current_h = self.rnn_model.init_hidden(batch_size=1, device=self.device)

    def choose_action(self, action_selection=ActionSelection.EPSILON_GREEDY, epsilon=0.2, temperature=1.0, c=1.0):
        if action_selection == ActionSelection.EPSILON_GREEDY:
            return self.choose_action_epsilon(epsilon=epsilon)
        elif action_selection == ActionSelection.BOLTZMANN:
            return self.choose_action_Boltzmann(temperature=temperature)
        else:
            return self.choose_action_UCB(c=c)

    def choose_action_epsilon(self, epsilon=0.2):
        """
        Choose an action based on maximizing the Self-Model prediction error.
        Uses the current latent state z_t and RNN hidden state h_t.

        Args:
            epsilon (float): Probability of choosing a random action.
        Returns:
            tuple: (action_key, action_array)
        """
        if self.current_z is None or self.current_h is None:
            raise ValueError("Agent state (z or h) not initialized. Call encode_image and reset_hidden_state first.")

        if np.random.rand() < epsilon:
            # Random exploration
            action_key = np.random.choice(list(self.actions.keys()))
        else:
            # Exploitation: Predict future prediction errors using SelfModel
            self.self_model.eval() # Set self_model to evaluation mode
            with torch.no_grad():
                future_rewards = []
                # Get the hidden state part 'h' from the tuple (h, c)
                h_state_for_self_model = self.current_h[0] # Shape: (NumLayers, Batch=1, HiddenDim)

                for key, val_array in self.actions.items():
                    action_tensor = torch.tensor(val_array, dtype=torch.float32).unsqueeze(0).to(self.device) # (1, ActionDim)
                    # Predict reward for taking this action given current z and h
                    predicted_reward = self.self_model(self.current_z, h_state_for_self_model, action_tensor)
                    future_rewards.append((predicted_reward.item(), key))


            self.self_model.train() # Set back to training mode
            if not future_rewards: # Handle empty case if needed
                action_key = np.random.choice(list(self.actions.keys()))
            else:
                # Choose action predicted to yield highest prediction error
                _, action_key = max(future_rewards)

        action_array = self.actions[action_key]
        return action_key, np.array(action_array, dtype=np.float32)
    
    def choose_action_Boltzmann(self, temperature=1.0):
        """
        Choose an action using Softmax (Boltzmann) exploration over predicted rewards.

        Args:
            temperature (float): Controls exploration. Higher = more random; lower = more greedy.
        Returns:
            tuple: (action_key, action_array)
        """
        if self.current_z is None or self.current_h is None:
            raise ValueError("Agent state (z or h) not initialized. Call encode_image and reset_hidden_state first.")

        self.self_model.eval()
        with torch.no_grad():
            h_state_for_self_model = self.current_h[0]  # (NumLayers, 1, HiddenDim)

            action_keys = list(self.actions.keys())
            reward_values = []

            for key in action_keys:
                val_array = self.actions[key]
                action_tensor = torch.tensor(val_array, dtype=torch.float32).unsqueeze(0).to(self.device)
                predicted_reward = self.self_model(self.current_z, h_state_for_self_model, action_tensor)
                reward_values.append(predicted_reward.item())

            # Softmax über die Reward-Werte
            reward_tensor = torch.tensor(reward_values, dtype=torch.float32)
            probs = F.softmax(reward_tensor / temperature, dim=0).cpu().numpy()

            # Auswahl einer Aktion basierend auf Softmax-Wahrscheinlichkeiten
            action_index = np.random.choice(len(action_keys), p=probs)
            action_key = action_keys[action_index]

        self.self_model.train()
        action_array = self.actions[action_key]
        return action_key, np.array(action_array, dtype=np.float32)



    def choose_action_UCB(self, c=1.0):
        """
        Choose an action using Upper Confidence Bound (UCB) strategy.

        Args:
            c (float): Exploration constant. Higher = more exploration.
        Returns:
            tuple: (action_key, action_array)
        """
        if self.current_z is None or self.current_h is None:
            raise ValueError("Agent state (z or h) not initialized. Call encode_image and reset_hidden_state first.")

        # Initialisierung der Zählungen, falls nicht vorhanden
        if not hasattr(self, "action_counts"):
            self.action_counts = {key: 1 for key in self.actions}  # starte mit 1 um 0-Division zu vermeiden
            self.total_steps = len(self.actions)

        self.self_model.eval()
        with torch.no_grad():
            h_state_for_self_model = self.current_h[0]
            ucb_scores = []
            action_keys = list(self.actions.keys())

            for key in action_keys:
                val_array = self.actions[key]
                action_tensor = torch.tensor(val_array, dtype=torch.float32).unsqueeze(0).to(self.device)
                predicted_reward = self.self_model(self.current_z, h_state_for_self_model, action_tensor).item()

                # UCB-Wert berechnen
                count = self.action_counts.get(key, 1)
                bonus = c * math.sqrt(math.log(self.total_steps + 1) / count)
                ucb_score = predicted_reward + bonus
                ucb_scores.append((ucb_score, key))

            # Wähle Aktion mit höchstem UCB-Wert
            _, action_key = max(ucb_scores)

        # Zählungen aktualisieren
        self.action_counts[action_key] += 1
        self.total_steps += 1
        self.self_model.train()

        action_array = self.actions[action_key]
        return action_key, np.array(action_array, dtype=np.float32)
    
    def store_experience(self, raw_image, action_key, action_array, reward, next_raw_image, done):
        """ Stores an experience tuple in the replay buffer """
        # Note: We store raw images and re-process/encode them during training updates
        # This is more memory intensive but avoids issues if the VAE changes during training
        experience = Experience(raw_image, action_key, action_array, reward, next_raw_image, done)
        self.replay_buffer.append(experience)

    def calculate_curiosity_reward(self, predicted_next_z, actual_next_z):
        """
        Calculate curiosity reward based on the RNN's prediction error in latent space.
        Args:
            predicted_next_z (torch.Tensor): RNN's prediction of z_{t+1} (Batch, LatentDim).
            actual_next_z (torch.Tensor): VAE's encoding of the actual next image (Batch, LatentDim).
        Returns:
            torch.Tensor: Curiosity reward tensor (Batch, 1).
        """
        # Use Mean Squared Error in the latent space as the reward signal
        # Higher error means higher curiosity reward
        reward = F.mse_loss(predicted_next_z, actual_next_z.detach(), reduction='none').mean(dim=1) # Average MSE across latent dimensions
        # Detach actual_next_z as it's the target«
        return reward.unsqueeze(1) # Shape: (Batch, 1)

    def update_models(self, visualize=False, step=0, log_dir="logs", num_agents=1, visualize_vae_after_steps=2, visualize_rnn_after_steps=2):
        """ Samples a batch from the replay buffer and updates VAE, RNN, and SelfModel """
        if len(self.replay_buffer) < self.batch_size:
            return None # Not enough samples to train yet

        # Sample a batch of experiences
        experiences = random.sample(self.replay_buffer, self.batch_size)
        batch = Experience(*zip(*experiences)) # Transpose the batch

        # --- Prepare Batch Data ---
        # Convert raw images to tensors and encode with VAE
        # Stack raw images and preprocess together for efficiency
        raw_images_np = np.stack(batch.raw_image)
        next_raw_images_np = np.stack(batch.next_raw_image)

        processed_images = torch.stack([self.transform(img) for img in raw_images_np]).to(self.device)
        processed_next_images = torch.stack([self.transform(img) for img in next_raw_images_np]).to(self.device)

        # Encode images to get z_t and z_{t+1}
        mu_t, logvar_t = self.vae.encode(processed_images)
        z_t = self.vae.reparameterize(mu_t, logvar_t) # (Batch, LatentDim)

        mu_t1, logvar_t1 = self.vae.encode(processed_next_images)
        z_t1_actual = self.vae.reparameterize(mu_t1, logvar_t1) # (Batch, LatentDim) - This is the target for RNN

        # Convert actions and rewards to tensors
        actions_array = np.stack(batch.action_array)
        action_t = torch.tensor(actions_array, dtype=torch.float32).to(self.device) # (Batch, ActionDim)

        # --- 1. Update VAE ---
        self.optimizer_vae.zero_grad()
        recon_x, _, mu, logvar = self.vae(processed_images) # Pass current images through VAE
        vae_loss_info = self.vae.loss_function(recon_x, processed_images, mu, logvar)
        vae_loss = vae_loss_info['loss']
        vae_loss.backward()
        self.optimizer_vae.step()
        # --- Periodically Visualize VAE ---
        if visualize:
            visualize_vae_reconstruction(processed_images, recon_x, step)

        # --- 2. Update RNN Model ---
        initial_hidden = self.rnn_model.init_hidden(self.batch_size, self.device)
        self.optimizer_rnn.zero_grad()
        
        # Predict z_{t+1} using the RNN
        predicted_z_t1, _ = self.rnn_model(z_t.detach(), action_t, initial_hidden)
        
        # Calculate prediction error using the new method
        error_metrics = self.calculate_prediction_error(predicted_z_t1, z_t1_actual.detach())
        rnn_loss = error_metrics['actual_error'].mean()
        rnn_loss.backward()
        self.optimizer_rnn.step()

        # --- 3. Calculate Curiosity Reward for SelfModel Training ---
        with torch.no_grad():
            curiosity_reward_batch = error_metrics['actual_error'].unsqueeze(1)

        # --- 4. Update Self Model ---
        self.optimizer_self.zero_grad()
        h_state_for_self_model = initial_hidden[0]
        predicted_reward = self.self_model(z_t.detach(), h_state_for_self_model.detach(), action_t)
        
        # Calculate self model loss using the new method
        self_error_metrics = self.calculate_prediction_error(
            predicted_z_t1.detach(), 
            z_t1_actual.detach(),
            predicted_reward
        )
        self_loss = self_error_metrics['error_loss']
        
        self_loss.backward()
        self.optimizer_self.step()

        # --- 5. Visualize RNN Prediction ---
        # VAE visualization
        if step % visualize_vae_after_steps == 0:  # Visualize every 1000 steps
            if processed_images.nelement() > 0:
                self.vae.eval()
                with torch.no_grad():
                    recon_x_all_vis, _, _, _ = self.vae(processed_images[:min(8*num_agents, processed_images.shape[0])])  # Take max 8 pairs
                visualize_path = os.path.join(log_dir, "vae_reconstructions")
                visualize_vae_reconstruction(processed_images[:min(8*num_agents, processed_images.shape[0])],
                                          recon_x_all_vis,
                                          step,
                                          save_dir=visualize_path)
                self.vae.train()

        # RNN prediction visualization
        if step % visualize_rnn_after_steps == 0 and step > 0:  # Different period for RNN viz
            self.vae.eval()  # VAE decoder will be used in eval mode
            
            if processed_next_images.nelement() > 0 and predicted_z_t1.nelement() > 0:
                rnn_pred_save_dir = os.path.join(log_dir, "rnn_predictions")
                # 1) Ziel-Device bestimmen
                target_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
                # 2) Tensoren auf dasselbe Device verschieben
                actual_frames = processed_next_images.to(target_device)
                pred_frames   = predicted_z_t1.detach().to(target_device)
                # 3) Visualisierung aufrufen
                visualize_rnn_prediction(
                    actual_next_frames     = actual_frames,          # Ground truth o_{t+1}
                    rnn_predicted_latent_z = pred_frames,            # RNN's z_{t+1} prediction
                    vae_decode_function    = self.vae.decode,        # VAE-Decoder
                    step                   = step,
                    agent_id               = 0,                     # Agent ID für Dateinamen
                    save_dir               = rnn_pred_save_dir
                )

        # Return losses for logging
        return {
            'vae_loss': vae_loss.item(),
            'vae_recon_loss': vae_loss_info['Reconstruction_Loss'].item(),
            'vae_kld_loss': vae_loss_info['KLD'].item(),
            'rnn_loss': rnn_loss.item(),
            'self_loss': self_loss.item(),
            'avg_curiosity_reward': curiosity_reward_batch.mean().item()
        }

