import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the VAE (Vision Model 'V')
class VAE(nn.Module):
    def __init__(self, input_channels=3, latent_dim=32, hidden_dims=[32, 64, 128, 256]):
        """
        Convolutional Variational Autoencoder (VAE).
        Encodes a 64x64 image into a latent vector z and decodes z back into an image.
        In the multi-agent setting, processes batches containing images from both agents.

        Args:
            input_channels (int): Number of channels in the input image (e.g., 3 for RGB).
            latent_dim (int): Dimensionality of the latent space vector z.
            hidden_dims (list): List of channel sizes for convolutional layers.
        """
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        modules = []

        # --- Encoder ---
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
        self.final_feature_map_size = 4  # 64 / (2^4) = 4
        self.flattened_size = hidden_dims[-1] * (self.final_feature_map_size ** 2)

        # Fully connected layers for mean and log variance
        self.fc_mu = nn.Linear(self.flattened_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flattened_size, latent_dim)

        # --- Decoder ---
        modules = []
        self.decoder_input = nn.Linear(latent_dim, self.flattened_size)

        hidden_dims.reverse()  # Reverse for decoder [256, 128, 64, 32]

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

        # Final layer to reconstruct the image
        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(hidden_dims[-1], input_channels, kernel_size=3, stride=1, padding=1),
                nn.Sigmoid())
        )

        self.decoder = nn.Sequential(*modules)
        hidden_dims.reverse()  # Reverse back for consistency

    def encode(self, x):
        """
        Encodes the input image batch into latent space parameters.
        In multi-agent setting, x contains images from both agents.

        Args:
            x (torch.Tensor): Input image tensor (Batch, Channels, Height, Width).
        Returns:
            tuple: (mu, logvar) tensors.
        """
        result = self.encoder(x)
        result = torch.flatten(result, start_dim=1)
        mu = self.fc_mu(result)
        logvar = self.fc_logvar(result)
        return mu, logvar

    def decode(self, z):
        """
        Decodes the latent vector batch z back into images.
        In multi-agent setting, z contains latent vectors from both agents.

        Args:
            z (torch.Tensor): Latent vector (Batch, LatentDim).
        Returns:
            torch.Tensor: Reconstructed image tensor (Batch, Channels, Height, Width).
        """
        result = self.decoder_input(z)
        result = result.view(-1, self.encoder[-1][0].out_channels, self.final_feature_map_size, self.final_feature_map_size)
        result = self.decoder(result)
        return result

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample z from N(mu, var).
        Args:
            mu (torch.Tensor): Mean of the latent Gaussian.
            logvar (torch.Tensor): Log variance of the latent Gaussian.
        Returns:
            torch.Tensor: Sampled latent vector z.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        """
        Forward pass through the VAE (encode, reparameterize, decode).
        In multi-agent setting, processes batches containing images from both agents.

        Args:
            x (torch.Tensor): Input image tensor (Batch, C, H, W).
        Returns:
            tuple: (reconstructed_x, x, mu, logvar)
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed_x = self.decode(z)
        return reconstructed_x, x, mu, logvar

    def loss_function(self, recon_x, x, mu, logvar, kld_weight=1.0):
        """
        Calculates VAE loss = Reconstruction Loss + KLD Loss.
        Args:
            recon_x: Reconstructed image.
            x: Original input image.
            mu: Latent mean.
            logvar: Latent log variance.
            kld_weight: Weight for the KL divergence term.
        Returns:
            dict: Dictionary containing total loss, reconstruction loss, and KLD.
        """
        batch_size = x.shape[0]
        if batch_size == 0:
            return {'loss': torch.tensor(0.0), 'Reconstruction_Loss': torch.tensor(0.0), 'KLD': torch.tensor(0.0)}

        recon_loss = F.mse_loss(recon_x.view(batch_size, -1), x.view(batch_size, -1), reduction='sum') / batch_size
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size

        loss = recon_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss': recon_loss.detach(), 'KLD': kld_loss.detach()}


# Define the RNN Model (Memory Model 'M') - Shared Core, Separate Heads
class RNNModel(nn.Module):
    def __init__(self, latent_dim=32, action_dim=4, rnn_hidden_dim=256, num_rnn_layers=1, num_agents=2):
        """
        Recurrent Neural Network (LSTM) model with shared core and separate heads for multi-agent.
        Predicts the next latent state z_{t+1} for each agent given combined current latent states z_t,
        combined actions a_t, and the shared hidden state h_t.

        Args:
            latent_dim (int): Dimensionality of the VAE latent vector z (per agent).
            action_dim (int): Dimensionality of the action vector a (per agent).
            rnn_hidden_dim (int): Number of hidden units in the LSTM core.
            num_rnn_layers (int): Number of LSTM layers in the core.
            num_agents (int): Number of agents (default: 2).
        """
        super(RNNModel, self).__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.rnn_hidden_dim = rnn_hidden_dim
        self.num_rnn_layers = num_rnn_layers
        self.num_agents = num_agents

        # --- Shared LSTM Core ---
        # Input size is concatenation of latent states and actions from all agents
        self.core_input_dim = (latent_dim + action_dim) * num_agents
        self.lstm_core = nn.LSTM(self.core_input_dim, rnn_hidden_dim, num_rnn_layers, batch_first=True)

        # --- Separate Prediction Heads ---
        # One head per agent to predict their next latent state
        self.prediction_heads = nn.ModuleList([
            nn.Linear(rnn_hidden_dim, latent_dim) for _ in range(num_agents)
        ])

    def forward(self, z_combined, action_combined, hidden_state):
        """
        Forward pass through the shared core RNN and separate prediction heads.

        Args:
            z_combined (torch.Tensor): Concatenated current latent states of all agents
                                     Shape: (Batch, latent_dim * num_agents).
            action_combined (torch.Tensor): Concatenated current actions of all agents
                                          Shape: (Batch, action_dim * num_agents).
            hidden_state (tuple): Previous hidden state (h_0, c_0) of the LSTM core.
                                Each has shape (NumLayers, Batch, HiddenDim).
                                Pass None for the initial state.

        Returns:
            tuple: (predicted_z_t1_list, next_hidden_state)
                   predicted_z_t1_list (list): List containing the predicted next latent state
                                              tensor for each agent [(Batch, LatentDim), ...].
                   next_hidden_state (tuple): Next shared hidden state (h_n, c_n).
        """
        # Prepare input for LSTM core
        lstm_input = torch.cat((z_combined, action_combined), dim=1)
        lstm_input = lstm_input.unsqueeze(1)  # Add sequence length dimension

        # Shared LSTM core pass
        lstm_out, next_hidden_state = self.lstm_core(lstm_input, hidden_state)
        lstm_out_squeezed = lstm_out.squeeze(1)  # Shape: (Batch, HiddenDim)

        # Separate prediction heads pass
        predicted_z_t1_list = []
        for i in range(self.num_agents):
            predicted_z = self.prediction_heads[i](lstm_out_squeezed)
            predicted_z_t1_list.append(predicted_z)

        return predicted_z_t1_list, next_hidden_state

    def init_hidden(self, batch_size, device):
        """
        Initializes the hidden state of the LSTM core.
        Args:
            batch_size (int): The batch size.
            device: The torch device ('cpu' or 'cuda').
        Returns:
            tuple: Initial hidden state (hidden_state, cell_state).
        """
        hidden_state = torch.zeros(self.num_rnn_layers, batch_size, self.rnn_hidden_dim).to(device)
        cell_state = torch.zeros(self.num_rnn_layers, batch_size, self.rnn_hidden_dim).to(device)
        return (hidden_state, cell_state)


# Define the Self Model (predicts curiosity reward)
class SelfModel(nn.Module):
    def __init__(self, latent_dim=32, rnn_hidden_dim=256, action_dim=4, fc_hidden_dim=128, output_dim=1):
        """
        A self-model that predicts curiosity rewards (prediction error of the associated RNN head).
        Takes agent's current latent state z_t, the shared RNN hidden state h_t,
        and agent's action a_t as input.

        Args:
            latent_dim (int): Dimensionality of the VAE latent vector z.
            rnn_hidden_dim (int): Dimensionality of the shared RNN hidden state h.
            action_dim (int): Dimensionality of the action vector a.
            fc_hidden_dim (int): Hidden dimension for fully connected layers.
            output_dim (int): Output dimension (1 for reward/error prediction).
        """
        super(SelfModel, self).__init__()

        # Input size is concatenation of agent's z_t, shared h_t (last layer), and agent's action a_t
        input_dim = latent_dim + rnn_hidden_dim + action_dim

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(input_dim, fc_hidden_dim),
            nn.ReLU(),
            nn.Linear(fc_hidden_dim, fc_hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(fc_hidden_dim // 2, output_dim)  # Output predicted reward/error
        )

    def forward(self, z_agent, h_shared, action_agent):
        """
        Forward pass through the self-model for a specific agent.

        Args:
            z_agent (torch.Tensor): Current latent state of the specific agent (Batch, LatentDim).
            h_shared (torch.Tensor): Current shared RNN hidden state (NumLayers, Batch, HiddenDim).
                                   We use the output of the last layer.
            action_agent (torch.Tensor): Action tensor of the specific agent (Batch, ActionDim).

        Returns:
            torch.Tensor: Predicted curiosity reward/error for this agent (Batch, OutputDim).
        """
        # Extract the hidden state from the last layer of the shared RNN
        #last_layer_h_shared = h_shared[-1]  # Shape: (Batch, HiddenDim)
        try:
            # Concatenate agent's z_t, shared h_t (last layer), and agent's action a_t
            combined = torch.cat([z_agent, h_shared, action_agent], dim=1)
        except Exception as e:
            print(f"Error concatenating z_agent, last_layer_h_shared, and action_agent: {e}")
            print(f"  Input z_agent shape: {z_agent.shape}, ndim: {z_agent.ndim}")
            print(f"  Input h_shared shape: {h_shared.shape}, ndim: {h_shared.ndim}")
            print(f"  Input action_agent shape: {action_agent.shape}, ndim: {action_agent.ndim}")
            raise e
        # Process the combined features through fully connected layers
        predicted_reward = self.fc_layers(combined)
        return predicted_reward

