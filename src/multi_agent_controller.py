import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms as T
from collections import deque, namedtuple
import numpy as np
import random
import math

from models import VAE, RNNModel, SelfModel
from utility import visualize_vae_reconstruction

# --- Replay Buffer Definition ---
Experience = namedtuple('Experience', (
    'obs_t1', 'obs_t2',             # Observations at time t (Agent 1, Agent 2)
    'action_t1', 'action_t2',       # Actions at time t (Agent 1, Agent 2)
    'reward_t1', 'reward_t2',       # External rewards (mostly 0 here)
    'obs_tp1', 'obs_tp2',           # Observations at time t+1
    'done'                          # Is the state terminal?
))

class MultiAgentController:
    def __init__(self,
                # Models
                vae: VAE,
                rnn_model: RNNModel,
                self_models: list,  # List of SelfModel instances
                controllers: list,  # List of controller instances/parameters
                # Hyperparameters
                num_agents: int,
                action_map: dict,
                latent_dim: int,
                action_dim: int,
                rnn_hidden_dim: int,
                device: torch.device,
                # Training parameters
                replay_buffer_size: int,
                batch_size: int,
                learning_rate_vae: float,
                learning_rate_rnn: float,
                learning_rate_self: float,
                # Epsilon Greedy
                epsilon_start: float,
                epsilon_end: float,
                epsilon_decay: float):
        """
        Initialize the MultiAgentController.

        Args:
            vae: The (shared) VAE instance.
            rnn_model: The (shared) RNNModel instance (Shared Core / Separate Heads).
            self_models: List of SelfModel instances (one per agent).
            controllers: List of controller models/parameters (one per agent).
            num_agents: Number of agents.
            action_map: Dictionary of possible actions.
            latent_dim: Dimension of latent space.
            action_dim: Dimension of action vector per agent.
            rnn_hidden_dim: Dimension of RNN hidden state.
            device: Torch device ('cpu' or 'cuda').
            replay_buffer_size: Maximum size of replay buffer.
            batch_size: Batch size for updates.
            learning_rate_*: Learning rates for optimizers.
            epsilon_*: Parameters for Epsilon-Greedy exploration.
        """
        self.vae = vae
        self.rnn_model = rnn_model
        self.self_models = self_models
        self.controllers = controllers
        self.num_agents = num_agents
        self.action_map = action_map
        self.action_keys_list = list(action_map.keys())
        self.action_arrays_list = [np.array(v, dtype=np.float32)[:action_dim] for v in action_map.values()]
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.rnn_hidden_dim = rnn_hidden_dim
        self.device = device

        self.replay_buffer = deque(maxlen=replay_buffer_size)
        self.batch_size = batch_size

        # Optimizers
        self.optimizer_vae = optim.Adam(self.vae.parameters(), lr=learning_rate_vae)
        self.optimizer_rnn = optim.Adam(self.rnn_model.parameters(), lr=learning_rate_rnn)
        self.optimizer_self_models = [optim.Adam(sm.parameters(), lr=learning_rate_self) for sm in self.self_models]

        # Epsilon Greedy state
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.current_step = 0  # Internal counter for epsilon decay

        # Image preprocessing (consistent with VAE training)
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((64, 64)),  # Must match VAE input size
            T.ToTensor(),
        ])

        # Initial shared hidden state (will be set externally or initialized here)
        # Important: Batch size is 1 for step-by-step processing
        self.h_t = self.rnn_model.init_hidden(batch_size=1, device=self.device)
        # Current latent states (will be set externally or initialized here)
        self.z_t_all = [torch.zeros(1, self.latent_dim).to(self.device) for _ in range(self.num_agents)]

    def preprocess_observation(self, obs_np):
        """Applies transformation to a single NumPy observation."""
        if obs_np is None:
            return None
        try:
            return self.transform(obs_np)
        except Exception as e:
            print(f"Error preprocessing observation: {e}")
            return torch.zeros((3, 64, 64))  # Placeholder

    def set_initial_state(self, initial_obs_list):
        """Sets the initial latent state and RNN hidden state."""
        initial_obs_processed = [self.preprocess_observation(obs) for obs in initial_obs_list]
        if any(o is None for o in initial_obs_processed):
            raise ValueError("Initial observation preprocessing failed.")

        initial_obs_batch = torch.stack(initial_obs_processed).to(self.device)
        self.vae.eval()
        with torch.no_grad():
            mu_t, logvar_t = self.vae.encode(initial_obs_batch)
            self.z_t_all = [self.vae.reparameterize(mu_t[i:i+1], logvar_t[i:i+1]) for i in range(self.num_agents)]
        self.vae.train()
        self.h_t = self.rnn_model.init_hidden(batch_size=1, device=self.device)
        print("MultiAgentController initial state set.")

    def choose_actions(self, epsilon=0.5):
        """Chooses actions for all agents for the current step."""
        self.current_step += 1

        action_keys = [None] * self.num_agents
        action_arrays = [None] * self.num_agents

        # Get relevant part of shared hidden state
        #h_t_last_layer = self.h_t[0][-1]  # Shape: (1, HiddenDim)
        h_t_last_layer = self.h_t[0][-1, :, :].detach()


        for i in range(self.num_agents):
            if random.random() < epsilon:
                # Random exploration
                rand_idx = random.randrange(len(self.action_keys_list))
                action_keys[i] = self.action_keys_list[rand_idx]
                action_arrays[i] = self.action_arrays_list[rand_idx]
            else:
                # Choose action based on Self-Model prediction
                self.self_models[i].eval()

                with torch.no_grad():
                    best_action_idx = -1
                    max_predicted_reward = -float('inf')

                    for idx, action_arr in enumerate(self.action_arrays_list):
                        action_tensor = torch.tensor(action_arr, dtype=torch.float32).unsqueeze(0).to(self.device)  # (1, ActionDim)
                        z_t_agent = self.z_t_all[i]  # Current latent state (1, LatentDim)

                        # Self-Model prediction
                        predicted_reward = self.self_models[i](z_t_agent, h_t_last_layer, action_tensor)  # Shape: (1, 1)
                        predicted_reward_scalar = predicted_reward.item()

                        if predicted_reward_scalar > max_predicted_reward:
                            max_predicted_reward = predicted_reward_scalar
                            best_action_idx = idx

                    if best_action_idx != -1:
                        action_keys[i] = self.action_keys_list[best_action_idx]
                        action_arrays[i] = self.action_arrays_list[best_action_idx]
                    else:
                        # Fallback
                        rand_idx = random.randrange(len(self.action_keys_list))
                        action_keys[i] = self.action_keys_list[rand_idx]
                        action_arrays[i] = self.action_arrays_list[rand_idx]

                self.self_models[i].train()

        return action_keys, action_arrays, epsilon  # Return epsilon for logging

    def store_experience(self, obs_t_list, action_arrays_list, reward_list, obs_tp1_list, done):
        """Stores the experience in the replay buffer."""
        if len(obs_t_list) != self.num_agents or len(action_arrays_list) != self.num_agents or \
           len(reward_list) != self.num_agents or len(obs_tp1_list) != self.num_agents:
            print("Warning: Incorrect number of elements provided to store_experience.")
            return

        # Ensure observations are NumPy arrays
        obs_t_np = [np.array(obs) if not isinstance(obs, np.ndarray) else obs for obs in obs_t_list]
        obs_tp1_np = [np.array(obs) if not isinstance(obs, np.ndarray) else obs for obs in obs_tp1_list]

        exp = Experience(obs_t_np[0], obs_t_np[1],
                        action_arrays_list[0], action_arrays_list[1],
                        reward_list[0], reward_list[1],
                        obs_tp1_np[0], obs_tp1_np[1],
                        done)
        self.replay_buffer.append(exp)

    def update_rnn_state(self, action_arrays_list, next_obs_list):
        """Updates the latent state z_t and RNN hidden state h_t."""
        # Encode next observations
        obs_tp1_processed = [self.preprocess_observation(obs) for obs in next_obs_list]
        if any(o is None for o in obs_tp1_processed):
            print(f"Warning: Observation preprocessing failed during RNN state update. State not updated.")
            return False  # Signals error

        obs_tp1_batch = torch.stack(obs_tp1_processed).to(self.device)
        self.vae.eval()
        with torch.no_grad():
            mu_tp1, logvar_tp1 = self.vae.encode(obs_tp1_batch)
            z_tp1_all = [self.vae.reparameterize(mu_tp1[i:i+1], logvar_tp1[i:i+1]) for i in range(self.num_agents)]
        self.vae.train()

        # Calculate next shared hidden state h_{t+1}
        z_t_combined = torch.cat(self.z_t_all, dim=1)  # Use current z_t
        action_t_combined = torch.cat([torch.tensor(a, dtype=torch.float32).unsqueeze(0).to(self.device) for a in action_arrays_list], dim=1)

        self.rnn_model.eval()
        with torch.no_grad():
            _, h_tp1 = self.rnn_model(z_t_combined, action_t_combined, self.h_t)
        self.rnn_model.train()

        # Update states for next step
        self.z_t_all = z_tp1_all
        self.h_t = h_tp1
        return True  # Signals success

    def update_models(self, current_step):
        """Performs coordinated update of all models."""
        if len(self.replay_buffer) < self.batch_size:
            return None  # Not enough data

        # --- 1. Data Sampling and Preparation ---
        experiences = random.sample(self.replay_buffer, self.batch_size)
        batch = Experience(*zip(*experiences))

        # Prepare observations and check for errors
        obs_t1_list = [self.preprocess_observation(obs) for obs in batch.obs_t1]
        obs_t2_list = [self.preprocess_observation(obs) for obs in batch.obs_t2]
        obs_tp1_list = [self.preprocess_observation(obs) for obs in batch.obs_tp1]
        obs_tp2_list = [self.preprocess_observation(obs) for obs in batch.obs_tp2]

        valid_indices = [i for i, (o1, o2, o3, o4) in enumerate(zip(obs_t1_list, obs_t2_list, obs_tp1_list, obs_tp2_list)) 
                        if o1 is not None and o2 is not None and o3 is not None and o4 is not None]
        
        if len(valid_indices) < self.batch_size:
            print(f"Warning: Only {len(valid_indices)} valid experiences in batch for update.")
            if len(valid_indices) == 0:
                return None
            batch_obs_t1 = [obs_t1_list[i] for i in valid_indices]
            batch_obs_t2 = [obs_t2_list[i] for i in valid_indices]
            batch_obs_tp1 = [obs_tp1_list[i] for i in valid_indices]
            batch_obs_tp2 = [obs_tp2_list[i] for i in valid_indices]
            batch_action_t1 = [batch.action_t1[i] for i in valid_indices]
            batch_action_t2 = [batch.action_t2[i] for i in valid_indices]
            current_batch_size = len(valid_indices)
        else:
            batch_obs_t1 = obs_t1_list
            batch_obs_t2 = obs_t2_list
            batch_obs_tp1 = obs_tp1_list
            batch_obs_tp2 = obs_tp2_list
            batch_action_t1 = batch.action_t1
            batch_action_t2 = batch.action_t2
            current_batch_size = self.batch_size

        obs_t_agent1_batch = torch.stack(batch_obs_t1).to(self.device)
        obs_t_agent2_batch = torch.stack(batch_obs_t2).to(self.device)
        obs_t_all = torch.cat((obs_t_agent1_batch, obs_t_agent2_batch), dim=0)

        obs_tp1_agent1_batch = torch.stack(batch_obs_tp1).to(self.device)
        obs_tp1_agent2_batch = torch.stack(batch_obs_tp2).to(self.device)
        obs_tp1_all = torch.cat((obs_tp1_agent1_batch, obs_tp1_agent2_batch), dim=0)

        try:
            action_t1 = torch.tensor(np.array(batch_action_t1, dtype=np.float32), dtype=torch.float32).to(self.device)
            action_t2 = torch.tensor(np.array(batch_action_t2, dtype=np.float32), dtype=torch.float32).to(self.device)
        except ValueError as e:
            print(f"Error converting actions to tensor: {e}")
            return None
        action_t_combined = torch.cat((action_t1, action_t2), dim=1)

        # --- 2. VAE Update ---
        self.vae.train()
        self.optimizer_vae.zero_grad()
        mu_t_all, logvar_t_all = self.vae.encode(obs_t_all)
        z_t_all_vae = self.vae.reparameterize(mu_t_all, logvar_t_all)
        recon_x_all, _, _, _ = self.vae(obs_t_all)
        vae_loss_info = self.vae.loss_function(recon_x_all, obs_t_all, mu_t_all, logvar_t_all)
        vae_loss = vae_loss_info['loss']
        if torch.isnan(vae_loss) or torch.isinf(vae_loss):
            print("Warning: VAE loss is NaN or Inf. Skipping VAE update.")
            vae_loss = torch.tensor(0.0)
        else:
            vae_loss.backward()
            self.optimizer_vae.step()

        # Extract latent states
        z_t1 = z_t_all_vae[:current_batch_size]
        z_t2 = z_t_all_vae[current_batch_size:]
        z_t_combined = torch.cat((z_t1, z_t2), dim=1).detach()

        self.vae.eval()
        with torch.no_grad():
            mu_tp1_all, logvar_tp1_all = self.vae.encode(obs_tp1_all)
            z_tp1_all_vae = self.vae.reparameterize(mu_tp1_all, logvar_tp1_all)
            z_tp1_actual_1 = z_tp1_all_vae[:current_batch_size]
            z_tp1_actual_2 = z_tp1_all_vae[current_batch_size:]
        self.vae.train()

        # --- 3. RNN Update ---
        self.rnn_model.train()
        self.optimizer_rnn.zero_grad()
        initial_hidden = self.rnn_model.init_hidden(current_batch_size, self.device)
        predicted_z_tp1_list, _ = self.rnn_model(z_t_combined, action_t_combined, initial_hidden)
        pred_z_tp1_1 = predicted_z_tp1_list[0]
        pred_z_tp1_2 = predicted_z_tp1_list[1]
        loss_rnn_head1 = F.mse_loss(pred_z_tp1_1, z_tp1_actual_1.detach())
        loss_rnn_head2 = F.mse_loss(pred_z_tp1_2, z_tp1_actual_2.detach())
        rnn_loss_combined = loss_rnn_head1 + loss_rnn_head2
        actual_error_1 = torch.tensor(0.0)
        actual_error_2 = torch.tensor(0.0)

        if torch.isnan(rnn_loss_combined) or torch.isinf(rnn_loss_combined):
            print("Warning: RNN loss is NaN or Inf. Skipping RNN update.")
            rnn_loss_combined = torch.tensor(0.0)
            loss_rnn_head1 = torch.tensor(0.0)
            loss_rnn_head2 = torch.tensor(0.0)
            with torch.no_grad():
                if not torch.isnan(pred_z_tp1_1).any() and not torch.isnan(z_tp1_actual_1).any():
                    actual_error_1 = F.mse_loss(pred_z_tp1_1.detach(), z_tp1_actual_1.detach(), reduction='none').mean(dim=1)
                if not torch.isnan(pred_z_tp1_2).any() and not torch.isnan(z_tp1_actual_2).any():
                    actual_error_2 = F.mse_loss(pred_z_tp1_2.detach(), z_tp1_actual_2.detach(), reduction='none').mean(dim=1)
        else:
            rnn_loss_combined.backward()
            self.optimizer_rnn.step()
            with torch.no_grad():
                actual_error_1 = F.mse_loss(pred_z_tp1_1.detach(), z_tp1_actual_1.detach(), reduction='none').mean(dim=1)
                actual_error_2 = F.mse_loss(pred_z_tp1_2.detach(), z_tp1_actual_2.detach(), reduction='none').mean(dim=1)

        # --- 4. Self-Model Updates ---
        h_t_for_self_model = initial_hidden[0][-1].detach()

        # Update Self-Model 1
        self.self_models[0].train()
        self.optimizer_self_models[0].zero_grad()
        predicted_reward_1 = self.self_models[0](z_t1.detach(), h_t_for_self_model, action_t1.detach())
        loss_self1 = F.mse_loss(predicted_reward_1.squeeze(), actual_error_1.detach())
        if torch.isnan(loss_self1) or torch.isinf(loss_self1):
            print("Warning: SelfModel 1 loss is NaN or Inf. Skipping update.")
            loss_self1 = torch.tensor(0.0)
        else:
            loss_self1.backward()
            self.optimizer_self_models[0].step()

        # Update Self-Model 2
        self.self_models[1].train()
        self.optimizer_self_models[1].zero_grad()
        predicted_reward_2 = self.self_models[1](z_t2.detach(), h_t_for_self_model, action_t2.detach())
        loss_self2 = F.mse_loss(predicted_reward_2.squeeze(), actual_error_2.detach())
        if torch.isnan(loss_self2) or torch.isinf(loss_self2):
            print("Warning: SelfModel 2 loss is NaN or Inf. Skipping update.")
            loss_self2 = torch.tensor(0.0)
        else:
            loss_self2.backward()
            self.optimizer_self_models[1].step()

        # Optional: VAE visualization after update
        if current_step % 1000 == 0:  # Visualize every 1000 steps
            if obs_t_all.nelement() > 0:
                self.vae.eval()
                with torch.no_grad():
                    recon_x_all_vis, _, _, _ = self.vae(obs_t_all[:min(8*self.num_agents, obs_t_all.shape[0])])  # Take max 8 pairs
                visualize_vae_reconstruction(obs_t_all[:min(8*self.num_agents, obs_t_all.shape[0])],
                                          recon_x_all_vis,
                                          current_step,
                                          save_dir="logs/multi_agent/vae_reconstructions")
                self.vae.train()

        # --- 5. Return losses for logging ---
        return {
            'vae_loss': vae_loss.item(),
            'vae_recon_loss': vae_loss_info['Reconstruction_Loss'].item(),
            'vae_kld_loss': vae_loss_info['KLD'].item(),
            'rnn_loss': rnn_loss_combined.item(),
            'rnn_loss_head1': loss_rnn_head1.item(),
            'rnn_loss_head2': loss_rnn_head2.item(),
            'self_loss_1': loss_self1.item(),
            'self_loss_2': loss_self2.item(),
            'avg_curiosity_reward_1': actual_error_1.mean().item() if actual_error_1.numel() > 0 and not torch.isnan(actual_error_1.mean()) else 0.0,
            'avg_curiosity_reward_2': actual_error_2.mean().item() if actual_error_2.numel() > 0 and not torch.isnan(actual_error_2.mean()) else 0.0,
        }