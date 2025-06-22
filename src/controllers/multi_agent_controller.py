import math
import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms as T
import numpy as np
import os
import random

from models.networks import VAE, RNNModel, SelfModel
from common.replay_buffer import ReplayBuffer
from common.enums import ActionSelection
from common.utility import visualize_vae_reconstruction, visualize_rnn_prediction


class MultiAgentController:
    def __init__(self, config, action_map, device):
        """
        Initializes the multi-agent controller.

        Args:
            config: A configuration object with all hyperparameters.
            action_map (dict): A mapping from action keys to action vectors.
            device (torch.device): The device to run models on ('cpu' or 'cuda').
        """
        self.config = config
        self.num_agents = config.NUM_AGENTS
        self.action_map = action_map
        self.action_keys = list(action_map.keys())
        self.action_arrays = [np.array(v, dtype=np.float32) for v in action_map.values()]
        self.device = device
        self.current_step = 0

        # --- Initialize Models ---
        self.vae = VAE(latent_dim=config.LATENT_DIM).to(device)
        self.rnn = RNNModel(
            latent_dim=config.LATENT_DIM,
            action_dim=config.ACTION_DIM,
            rnn_hidden_dim=config.RNN_HIDDEN_DIM,
            num_agents=self.num_agents
        ).to(device)
        self.self_models = [
            SelfModel(
                latent_dim=config.LATENT_DIM,
                rnn_hidden_dim=config.RNN_HIDDEN_DIM,
                action_dim=config.ACTION_DIM
            ).to(device) for _ in range(self.num_agents)
        ]

        # --- Initialize Optimizers ---
        self.optimizer_vae = optim.Adam(self.vae.parameters(), lr=config.LEARNING_RATE_VAE)
        self.optimizer_rnn = optim.Adam(self.rnn.parameters(), lr=config.LEARNING_RATE_RNN)
        self.optimizer_self = [optim.Adam(sm.parameters(), lr=config.LEARNING_RATE_SELF) for sm in self.self_models]

        # --- Replay Buffer ---
        self.replay_buffer = ReplayBuffer(config.REPLAY_BUFFER_SIZE)

        # --- Agent State ---
        self.z_t_all = [torch.zeros(1, config.LATENT_DIM).to(device) for _ in range(self.num_agents)]
        self.h_t = self.rnn.init_hidden(batch_size=1, device=self.device)

        # --- Image Transformation ---
        self.transform = T.Compose([
            T.ToPILImage(), T.Resize((64, 64)), T.ToTensor(),
        ])

        self.action_counts = [
            {key: 1 for key in self.action_keys} for _ in range(self.num_agents)
        ]
        self.total_steps = [1 for _ in range(self.num_agents)]




    


    def set_initial_state(self, initial_obs_list: list):
        """Sets the initial latent states z_t from the first observations."""
        initial_obs_tensors = torch.stack([self.transform(obs) for obs in initial_obs_list]).to(self.device)
        self.vae.eval()
        with torch.no_grad():
            mu, logvar = self.vae.encode(initial_obs_tensors)
            z_all = self.vae.reparameterize(mu, logvar)
            self.z_t_all = [z_all[i:i+1] for i in range(self.num_agents)]
        self.vae.train()
        self.h_t = self.rnn.init_hidden(batch_size=1, device=self.device)
        print("MultiAgentController initial state set.")

    def _choose_action_epsilon_greedy(self, agent_idx, z_t, h_t):
        """Epsilon-Greedy action selection for one agent."""
        if np.random.rand() < self.config.EPSILON_GREEDY:
            return np.random.randint(len(self.action_keys))
        else:
            self.self_models[agent_idx].eval()
            with torch.no_grad():
                rewards = []
                for action_arr in self.action_arrays:
                    action_tensor = torch.tensor(action_arr, dtype=torch.float32).unsqueeze(0).to(self.device)
                    pred_reward = self.self_models[agent_idx](z_t, h_t, action_tensor)
                    rewards.append(pred_reward.item())
                action_idx = np.argmax(rewards)
            self.self_models[agent_idx].train()
            return action_idx


    def _choose_action_boltzmann(self, agent_idx, z_t, h_t):
        """Boltzmann (softmax) action selection for one agent."""
        self.self_models[agent_idx].eval()
        with torch.no_grad():
            reward_values = []
            for action_arr in self.action_arrays:
                action_tensor = torch.tensor(action_arr, dtype=torch.float32).unsqueeze(0).to(self.device)
                pred_reward = self.self_models[agent_idx](z_t, h_t, action_tensor)
                reward_values.append(pred_reward.item())

            reward_tensor = torch.tensor(reward_values, dtype=torch.float32)
            probs = F.softmax(reward_tensor / self.config.TEMPERATURE, dim=0).cpu().numpy()
            action_idx = np.random.choice(len(self.action_keys), p=probs)
        self.self_models[agent_idx].train()
        return action_idx

        
    def _choose_action_ucb(self, agent_idx, z_t, h_t):
        self.self_models[agent_idx].eval()
        with torch.no_grad():
            ucb_scores = []
            for i, action_arr in enumerate(self.action_arrays):
                action_tensor = torch.tensor(action_arr, dtype=torch.float32).unsqueeze(0).to(self.device)
                predicted_reward = self.self_models[agent_idx](z_t, h_t, action_tensor).item()
                count = self.action_counts[agent_idx][self.action_keys[i]]
                bonus = self.config.UCB_C * math.sqrt(math.log(self.total_steps[agent_idx] + 1) / count)
                ucb_scores.append(predicted_reward + bonus)

            action_idx = np.argmax(ucb_scores)
        
        self.action_counts[agent_idx][self.action_keys[action_idx]] += 1
        self.total_steps[agent_idx] += 1
        self.self_models[agent_idx].train()
        return action_idx

    
    def choose_actions(self, obs_list: list):
        """Chooses actions for all agents based on the configured policy."""
        action_keys_list = [None] * self.num_agents
        action_arrays_list = [None] * self.num_agents

        h_t = self.h_t[0]  # LSTM hidden state

        for i in range(self.num_agents):
            z_t = self.z_t_all[i]
            if self.config.ACTION_SELECTION_TYPE == ActionSelection.EPSILON_GREEDY:
                action_idx = self._choose_action_epsilon_greedy(i, z_t, h_t)
            elif self.config.ACTION_SELECTION_TYPE == ActionSelection.BOLTZMANN:
                action_idx = self._choose_action_boltzmann(i, z_t, h_t)
            elif self.config.ACTION_SELECTION_TYPE == ActionSelection.UCB:
                action_idx = self._choose_action_ucb(i, z_t, h_t)
            else:
                raise ValueError(f"Unknown action selection type: {self.config.ACTION_SELECTION_TYPE}")
            
            action_keys_list[i] = self.action_keys[action_idx]
            action_arrays_list[i] = self.action_arrays[action_idx]

        return action_keys_list, action_arrays_list

    
    

    def store_experience(self, obs_t, actions_t, rewards_t, obs_tp1, done):
        """Adds an experience to the replay buffer."""
        self.replay_buffer.add(obs_t, actions_t, rewards_t, obs_tp1, done)

    def update_rnn_state(self, action_arrays_list: list, next_obs_list: list):
        """Updates the agent's latent state z and RNN hidden state h."""
        next_obs_tensors = torch.stack([self.transform(obs) for obs in next_obs_list]).to(self.device)
        self.vae.eval()
        with torch.no_grad():
            mu_tp1, logvar_tp1 = self.vae.encode(next_obs_tensors)
            z_tp1_all = [self.vae.reparameterize(mu_tp1[i:i+1], logvar_tp1[i:i+1]) for i in range(self.num_agents)]
        self.vae.train()

        z_t_combined = torch.cat(self.z_t_all, dim=1)
        action_t_combined = torch.cat([torch.tensor(a, dtype=torch.float32).unsqueeze(0).to(self.device) for a in action_arrays_list], dim=1)

        self.rnn.eval()
        with torch.no_grad():
            _, h_tp1 = self.rnn(z_t_combined, action_t_combined, self.h_t)
        self.rnn.train()

        self.z_t_all = z_tp1_all
        self.h_t = h_tp1
        return True

    def update_models(self) -> dict:
        """Samples a batch and updates all models."""
        if len(self.replay_buffer) < self.config.BATCH_SIZE:
            return None

        batch = self.replay_buffer.sample(self.config.BATCH_SIZE)
        batch_size = len(batch.done)
        
        # Prepare batch data
        obs_t_all_agents = [item for sublist in batch.obs_t for item in sublist]
        obs_tp1_all_agents = [item for sublist in batch.obs_tp1 for item in sublist]
        
        obs_t_tensor = torch.stack([self.transform(img) for img in obs_t_all_agents]).to(self.device)
        obs_tp1_tensor = torch.stack([self.transform(img) for img in obs_tp1_all_agents]).to(self.device)

        
        
        actions_t_list = [torch.tensor(np.array([a[i] for a in batch.actions_t]), dtype=torch.float32).to(self.device) for i in range(self.num_agents)]
        action_t_combined = torch.cat(actions_t_list, dim=1)
        
        # --- 1. Update VAE ---
        self.optimizer_vae.zero_grad()
        recon_x, _, mu_t, logvar_t = self.vae(obs_t_tensor)
        vae_loss_info = self.vae.loss_function(recon_x, obs_t_tensor, mu_t, logvar_t)
        vae_loss = vae_loss_info['loss'] / batch_size
        vae_loss.backward()
        self.optimizer_vae.step()
        
        z_t_all = self.vae.reparameterize(mu_t, logvar_t)
        z_t_list = [z_t_all[i*batch_size:(i+1)*batch_size] for i in range(self.num_agents)]
        z_t_combined = torch.cat(z_t_list, dim=1)
        
        with torch.no_grad():
            mu_tp1, logvar_tp1 = self.vae.encode(obs_tp1_tensor)
            z_tp1_all_actual = self.vae.reparameterize(mu_tp1, logvar_tp1)
            z_tp1_actual_list = [z_tp1_all_actual[i*batch_size:(i+1)*batch_size] for i in range(self.num_agents)]

        # --- 2. Update RNN ---
        self.optimizer_rnn.zero_grad()
        initial_hidden = self.rnn.init_hidden(batch_size, self.device)
        predicted_z_tp1_list, _ = self.rnn(z_t_combined.detach(), action_t_combined, initial_hidden)
        
        rnn_losses = [F.mse_loss(pred, actual.detach()) for pred, actual in zip(predicted_z_tp1_list, z_tp1_actual_list)]
        total_rnn_loss = sum(rnn_losses)
        total_rnn_loss.backward()
        self.optimizer_rnn.step()
        
        # --- 3. Update Self-Models ---
        # Correctly pass the 3D hidden state tensor from the initial hidden state
        h_for_self_model = initial_hidden[0]
        losses = {}
        with torch.no_grad():
            curiosity_rewards = [F.mse_loss(pred, actual, reduction='none').mean(dim=1) for pred, actual in zip(predicted_z_tp1_list, z_tp1_actual_list)]

        for i in range(self.num_agents):
            self.optimizer_self[i].zero_grad()
            predicted_reward = self.self_models[i](z_t_list[i].detach(), h_for_self_model.detach(), actions_t_list[i].detach())
            self_loss = F.mse_loss(predicted_reward.squeeze(), curiosity_rewards[i].detach())
            self_loss.backward()
            self.optimizer_self[i].step()
            losses[f'self_loss_{i}'] = self_loss.item()
            losses[f'avg_curiosity_reward_{i}'] = curiosity_rewards[i].mean().item()
            losses[f'rnn_loss_head_{i}'] = rnn_losses[i].item()

        # Visualization
        if self.current_step  % self.config.VAE_VISUALIZE_AFTER_STEPS == 0:
             visualize_path = os.path.join(self.config.LOG_DIR, "vae_reconstructions")
             visualize_vae_reconstruction(obs_t_tensor, recon_x.detach(), self.current_step, save_dir=visualize_path)

        if self.current_step  % self.config.RNN_VISUALIZE_AFTER_STEPS == 0:
            rnn_pred_save_dir = os.path.join(self.config.LOG_DIR, "rnn_predictions")

            # Stelle sicher, dass batch_size * num_agents == 16 (oder entsprechende Gesamtzahl)
            for i in range(self.num_agents):
                # Wähle jedes i-te Bild, beginnend bei i
                obs_for_agent_i = obs_tp1_tensor[i*batch_size:(i+1)*batch_size]           # z.B. i=0 → 0,2,4,...
                obs_for_agent_i2= obs_for_agent_i[i::self.num_agents]
                z_for_agent_i = predicted_z_tp1_list[i].detach()[i::self.num_agents]  # falls vorher kombiniert

                visualize_rnn_prediction(
                    obs_for_agent_i2,
                    z_for_agent_i,
                    self.vae.decode,
                    self.current_step,
                    i,
                    save_dir=rnn_pred_save_dir
                )
                


        # Return loss dictionary for logging
        return {
            'vae_loss': vae_loss.item(),
            'vae_recon_loss': vae_loss_info['Reconstruction_Loss'].item() / batch_size,
            'vae_kld_loss': vae_loss_info['KLD'].item() / batch_size,
            'rnn_loss': total_rnn_loss.item(),
            **losses
        }
