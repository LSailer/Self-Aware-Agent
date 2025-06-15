import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms as T
import numpy as np
import os
import math

from common.utility import visualize_rnn_prediction, visualize_vae_reconstruction
from models.networks import VAE, RNNModel, SelfModel
from common.replay_buffer import ReplayBuffer
from common.enums import ActionSelection

class SingleAgentController:
    def __init__(self, config, action_map, device):
        """
        Initializes the single-agent controller.

        Args:
            config: A configuration object with all hyperparameters.
            action_map (dict): A mapping from action keys to action vectors.
            device (torch.device): The device to run models on ('cpu' or 'cuda').
        """
        self.config = config
        self.action_map = action_map
        self.action_keys = list(action_map.keys())
        self.action_arrays = [np.array(v, dtype=np.float32) for v in action_map.values()]
        self.device = device
        self.current_step = 0

        # --- Initialize Models (using unified model classes) ---
        self.vae = VAE(latent_dim=config.LATENT_DIM).to(device)
        self.rnn = RNNModel(
            latent_dim=config.LATENT_DIM, action_dim=config.ACTION_DIM,
            rnn_hidden_dim=config.RNN_HIDDEN_DIM, num_agents=1
        ).to(device)
        self.self_model = SelfModel(
            latent_dim=config.LATENT_DIM, rnn_hidden_dim=config.RNN_HIDDEN_DIM,
            action_dim=config.ACTION_DIM
        ).to(device)

        # --- Initialize Optimizers ---
        self.optimizer_vae = optim.Adam(self.vae.parameters(), lr=config.LEARNING_RATE_VAE)
        self.optimizer_rnn = optim.Adam(self.rnn.parameters(), lr=config.LEARNING_RATE_RNN)
        self.optimizer_self = optim.Adam(self.self_model.parameters(), lr=config.LEARNING_RATE_SELF)

        # --- Replay Buffer ---
        self.replay_buffer = ReplayBuffer(config.REPLAY_BUFFER_SIZE)

        # --- Agent State ---
        self.current_z = None
        self.current_h = self.rnn.init_hidden(batch_size=1, device=self.device)
        
        # State for UCB action selection
        self.action_counts = {key: 1 for key in self.action_keys}
        self.total_steps = len(self.action_keys)

        # --- Image Transformation ---
        self.transform = T.Compose([
            T.ToPILImage(), T.Resize((64, 64)), T.ToTensor(),
        ])

    def set_initial_state(self, initial_obs_list: list):
        """Sets the initial latent state z_t from the first observation."""
        initial_obs_np = initial_obs_list[0]
        initial_obs_tensor = self.transform(initial_obs_np).unsqueeze(0).to(self.device)
        self.vae.eval()
        with torch.no_grad():
            mu, logvar = self.vae.encode(initial_obs_tensor)
            self.current_z = self.vae.reparameterize(mu, logvar)
        self.vae.train()
        self.current_h = self.rnn.init_hidden(batch_size=1, device=self.device)
        print("SingleAgentController initial state set.")

    def _choose_action_epsilon_greedy(self):
        """Epsilon-Greedy action selection."""
        if np.random.rand() < self.config.EPSILON_GREEDY:
            return np.random.randint(len(self.action_keys))
        else:
            self.self_model.eval()
            with torch.no_grad():
                rewards = []
                # Pass the raw 3D hidden state tensor (h from (h,c) tuple)
                raw_h_tensor = self.current_h[0]
                for action_arr in self.action_arrays:
                    action_tensor = torch.tensor(action_arr, dtype=torch.float32).unsqueeze(0).to(self.device)
                    pred_reward = self.self_model(self.current_z, raw_h_tensor, action_tensor)
                    rewards.append(pred_reward.item())
                action_idx = np.argmax(rewards)
            self.self_model.train()
            return action_idx

    def _choose_action_boltzmann(self):
        """Boltzmann (softmax) action selection."""
        self.self_model.eval()
        with torch.no_grad():
            # Pass the raw 3D hidden state tensor (h from (h,c) tuple)
            raw_h_tensor = self.current_h[0]
            reward_values = []
            for action_arr in self.action_arrays:
                action_tensor = torch.tensor(action_arr, dtype=torch.float32).unsqueeze(0).to(self.device)
                pred_reward = self.self_model(self.current_z, raw_h_tensor, action_tensor)
                reward_values.append(pred_reward.item())
            
            reward_tensor = torch.tensor(reward_values, dtype=torch.float32)
            probs = F.softmax(reward_tensor / self.config.TEMPERATURE, dim=0).cpu().numpy()
            action_idx = np.random.choice(len(self.action_keys), p=probs)
        self.self_model.train()
        return action_idx
        
    def _choose_action_ucb(self):
        """Upper Confidence Bound (UCB) action selection."""
        self.self_model.eval()
        with torch.no_grad():
            # Pass the raw 3D hidden state tensor (h from (h,c) tuple)
            raw_h_tensor = self.current_h[0]
            ucb_scores = []
            for i, action_arr in enumerate(self.action_arrays):
                action_tensor = torch.tensor(action_arr, dtype=torch.float32).unsqueeze(0).to(self.device)
                predicted_reward = self.self_model(self.current_z, raw_h_tensor, action_tensor).item()
                
                count = self.action_counts[self.action_keys[i]]
                bonus = self.config.UCB_C * math.sqrt(math.log(self.total_steps + 1) / count)
                ucb_scores.append(predicted_reward + bonus)
            
            action_idx = np.argmax(ucb_scores)
        
        self.action_counts[self.action_keys[action_idx]] += 1
        self.total_steps += 1
        self.self_model.train()
        return action_idx

    def choose_actions(self, obs_list: list) -> (list, list): # type: ignore
        """Chooses an action based on the configured policy."""
        if self.config.ACTION_SELECTION_TYPE == ActionSelection.EPSILON_GREEDY:
            action_idx = self._choose_action_epsilon_greedy()
        elif self.config.ACTION_SELECTION_TYPE == ActionSelection.BOLTZMANN:
            action_idx = self._choose_action_boltzmann()
        elif self.config.ACTION_SELECTION_TYPE == ActionSelection.UCB:
            action_idx = self._choose_action_ucb()
        else:
            raise ValueError(f"Unknown action selection type: {self.config.ACTION_SELECTION_TYPE}")

        action_key = self.action_keys[action_idx]
        action_array = self.action_arrays[action_idx]
        return [action_key], [action_array]

    def store_experience(self, obs_t, actions_t, rewards_t, obs_tp1, done):
        """Adds an experience to the replay buffer."""
        self.replay_buffer.add(obs_t, actions_t, rewards_t, obs_tp1, done)

    def update_rnn_state(self, action_arrays_list: list, next_obs_list: list):
        """Updates the agent's latent state z and RNN hidden state h."""
        next_obs_np = next_obs_list[0]
        action_array = action_arrays_list[0]
        
        next_obs_tensor = self.transform(next_obs_np).unsqueeze(0).to(self.device)
        self.vae.eval()
        with torch.no_grad():
            mu_tp1, logvar_tp1 = self.vae.encode(next_obs_tensor)
            z_tp1 = self.vae.reparameterize(mu_tp1, logvar_tp1)
        self.vae.train()
        
        # For single agent, the RNN expects the z and action to be concatenated
        z_t_combined = self.current_z
        action_tensor_combined = torch.tensor(action_array, dtype=torch.float32).unsqueeze(0).to(self.device)

        self.rnn.eval()
        with torch.no_grad():
            # The RNN needs a combined z and action tensor for its input
            # For num_agents=1, the RNN input size is (latent_dim + action_dim)
            rnn_input_z = z_t_combined
            rnn_input_a = action_tensor_combined
            
            # The controller passes the separate z and a to the model, which handles concatenation
            predicted_z_list, h_tp1 = self.rnn(rnn_input_z, rnn_input_a, self.current_h)
        self.rnn.train()
        
        self.current_z = z_tp1
        self.current_h = h_tp1
        return True

    def update_models(self) -> dict:
        """Samples a batch and updates all models."""
        if len(self.replay_buffer) < self.config.BATCH_SIZE:
            return None

        batch = self.replay_buffer.sample(self.config.BATCH_SIZE)
        batch_size = len(batch.done)

        obs_t_batch_np = np.array([obs[0] for obs in batch.obs_t])
        obs_tp1_batch_np = np.array([obs[0] for obs in batch.obs_tp1])
        action_t_batch = torch.tensor(np.array([a[0] for a in batch.actions_t]), dtype=torch.float32).to(self.device)
        
        obs_t_tensor = torch.stack([self.transform(img) for img in obs_t_batch_np]).to(self.device)
        obs_tp1_tensor = torch.stack([self.transform(img) for img in obs_tp1_batch_np]).to(self.device)

        self.optimizer_vae.zero_grad()
        recon_x, _, mu_t, logvar_t = self.vae(obs_t_tensor)
        vae_loss_info = self.vae.loss_function(recon_x, obs_t_tensor, mu_t, logvar_t)
        vae_loss = vae_loss_info['loss'] / self.config.BATCH_SIZE
        vae_loss.backward()
        self.optimizer_vae.step()
        
        z_t = self.vae.reparameterize(mu_t, logvar_t)
        with torch.no_grad():
            mu_tp1, logvar_tp1 = self.vae.encode(obs_tp1_tensor)
            z_tp1_actual = self.vae.reparameterize(mu_tp1, logvar_tp1)

        self.optimizer_rnn.zero_grad()
        initial_hidden = self.rnn.init_hidden(self.config.BATCH_SIZE, self.device)
        predicted_z_tp1_list, _ = self.rnn(z_t.detach(), action_t_batch, initial_hidden)
        predicted_z_tp1 = predicted_z_tp1_list[0]
        
        rnn_loss = F.mse_loss(predicted_z_tp1, z_tp1_actual.detach())
        rnn_loss.backward()
        self.optimizer_rnn.step()
        
        with torch.no_grad():
            curiosity_reward = F.mse_loss(predicted_z_tp1, z_tp1_actual, reduction='none').mean(dim=1)

        self.optimizer_self.zero_grad()
        # Pass the raw 3D hidden state tensor, not the pre-processed last layer
        h_for_self_model = initial_hidden[0]
        
        predicted_reward = self.self_model(z_t.detach(), h_for_self_model.detach(), action_t_batch.detach())
        self_loss = F.mse_loss(predicted_reward.squeeze(), curiosity_reward.detach())
        self_loss.backward()
        self.optimizer_self.step()
                # Visualization
        if self.current_step  % self.config.VAE_VISUALIZE_AFTER_STEPS == 0:
             visualize_path = os.path.join(self.config.LOG_DIR, "vae_reconstructions")
             visualize_vae_reconstruction(obs_t_tensor, recon_x.detach(), self.current_step, save_dir=visualize_path)

        if self.current_step  % self.config.RNN_VISUALIZE_AFTER_STEPS == 0:
            i=0
            rnn_pred_save_dir = os.path.join(self.config.LOG_DIR, "rnn_predictions")
            visualize_rnn_prediction(obs_tp1_tensor[i*batch_size:(i+1)*batch_size], predicted_z_tp1_list[i].detach(), self.vae.decode, self.current_step, i, save_dir=rnn_pred_save_dir)

        return {
            'vae_loss': vae_loss.item(),
            'vae_recon_loss': vae_loss_info['Reconstruction_Loss'].item() / self.config.BATCH_SIZE,
            'vae_kld_loss': vae_loss_info['KLD'].item() / self.config.BATCH_SIZE,
            'rnn_loss': rnn_loss.item(),
            'self_loss': self_loss.item(),
            'avg_curiosity_reward': curiosity_reward.mean().item()
        }
