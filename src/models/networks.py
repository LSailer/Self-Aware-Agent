# src/models/networks.py
"""
This module contains the core neural network models used by the agent(s):
- VAE: The vision model for encoding observations.
- RNNModel: The world model (memory) for predicting future states.
- SelfModel: The model for predicting the world model's error (curiosity).

These models are designed to be generic and support both single-agent and
multi-agent configurations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    """
    Convolutional Variational Autoencoder (VAE).
    Encodes a 64x64 image into a latent vector z and decodes z back.
    This model is generic and processes batches of images, regardless of
    which agent they come from.
    """
    def __init__(self, input_channels=3, latent_dim=32, hidden_dims=[32, 64, 128, 256]):
        super(VAE, self).__init__()
        # ... (Implementation is already generic and suitable for reuse)
        self.latent_dim = latent_dim
        modules = []
        in_channels = input_channels
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim
        self.encoder = nn.Sequential(*modules)
        self.final_feature_map_size = 4
        self.flattened_size = hidden_dims[-1] * (self.final_feature_map_size ** 2)
        self.fc_mu = nn.Linear(self.flattened_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flattened_size, latent_dim)
        modules = []
        self.decoder_input = nn.Linear(latent_dim, self.flattened_size)
        hidden_dims.reverse()
        in_channels = hidden_dims[0]
        for i in range(len(hidden_dims)):
            out_channels_decoder = hidden_dims[i+1] if i < len(hidden_dims) - 1 else hidden_dims[-1]
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_channels, out_channels_decoder, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(out_channels_decoder),
                    nn.LeakyReLU())
            )
            in_channels = out_channels_decoder
        modules.append(nn.Sequential(nn.ConvTranspose2d(hidden_dims[-1], input_channels, kernel_size=3, stride=1, padding=1), nn.Sigmoid()))
        self.decoder = nn.Sequential(*modules)

    def encode(self, x):
        result = self.encoder(x)
        result = torch.flatten(result, start_dim=1)
        mu = self.fc_mu(result)
        logvar = self.fc_logvar(result)
        return mu, logvar

    def decode(self, z):
        result = self.decoder_input(z)
        result = result.view(-1, self.encoder[-1][0].out_channels, self.final_feature_map_size, self.final_feature_map_size)
        return self.decoder(result)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed_x = self.decode(z)
        return reconstructed_x, x, mu, logvar

    def loss_function(self, recon_x, x, mu, logvar, kld_weight=1.0):
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss': recon_loss.detach(), 'KLD': kld_loss.detach()}


class RNNModel(nn.Module):
    """
    Recurrent Neural Network (LSTM) with a shared core and separate prediction heads.
    This model is generic and supports both single- and multi-agent setups by
    adjusting the `num_agents` parameter during initialization.
    """
    def __init__(self, latent_dim=32, action_dim=4, rnn_hidden_dim=256, num_rnn_layers=1, num_agents=2):
        super(RNNModel, self).__init__()
        self.num_agents = num_agents
        self.rnn_hidden_dim = rnn_hidden_dim
        self.num_rnn_layers = num_rnn_layers

        # The input dimension is dynamically calculated based on the number of agents.
        # For num_agents=1, this will be (latent_dim + action_dim).
        # For num_agents=2, this will be (latent_dim + action_dim) * 2.
        self.core_input_dim = (latent_dim + action_dim) * num_agents
        self.lstm_core = nn.LSTM(self.core_input_dim, rnn_hidden_dim, num_rnn_layers, batch_first=True)

        # A prediction head is created for each agent.
        # For num_agents=1, this will be a list with one head.
        self.prediction_heads = nn.ModuleList([
            nn.Linear(rnn_hidden_dim, latent_dim) for _ in range(num_agents)
        ])

    def forward(self, z_combined, action_combined, hidden_state):
        """
        Forward pass through the shared core RNN and separate prediction heads.
        Returns a list of predicted latent states, one for each agent.
        """
        lstm_input = torch.cat((z_combined, action_combined), dim=1)
        lstm_input = lstm_input.unsqueeze(1)
        lstm_out, next_hidden_state = self.lstm_core(lstm_input, hidden_state)
        lstm_out_squeezed = lstm_out.squeeze(1)

        predicted_z_t1_list = [head(lstm_out_squeezed) for head in self.prediction_heads]
        return predicted_z_t1_list, next_hidden_state

    def init_hidden(self, batch_size, device):
        """Initializes the hidden state for the LSTM core."""
        hidden = torch.zeros(self.num_rnn_layers, batch_size, self.rnn_hidden_dim).to(device)
        cell = torch.zeros(self.num_rnn_layers, batch_size, self.rnn_hidden_dim).to(device)
        return (hidden, cell)


class SelfModel(nn.Module):
    """
    Predicts the curiosity reward (world model's prediction error).
    This model is generic and is instantiated separately for each agent. It takes
    an agent's specific state (z_agent), the shared RNN state (h_shared), and
    the agent's action as input.
    """
    def __init__(self, latent_dim=32, rnn_hidden_dim=256, action_dim=4, fc_hidden_dim=128):
        super(SelfModel, self).__init__()
        input_dim = latent_dim + rnn_hidden_dim + action_dim
        self.fc_layers = nn.Sequential(
            nn.Linear(input_dim, fc_hidden_dim),
            nn.ReLU(),
            nn.Linear(fc_hidden_dim, fc_hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(fc_hidden_dim // 2, 1) # Output is a single value: the predicted error
        )

    def forward(self, z_agent, h_shared, action_agent):
        # The hidden state from the last layer of the shared RNN is used.
        last_layer_h_shared = h_shared[-1]
        combined = torch.cat([z_agent, last_layer_h_shared, action_agent], dim=1)
        predicted_reward = self.fc_layers(combined)
        return predicted_reward
