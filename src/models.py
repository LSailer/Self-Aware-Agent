import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the VAE (Vision Model 'V')
class VAE(nn.Module):
    def __init__(self, input_channels=3, latent_dim=32, hidden_dims=[32, 64, 128, 256]):
        """
        Convolutional Variational Autoencoder (VAE).
        Encodes a 64x64 image into a latent vector z and decodes z back into an image.

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
                    nn.Conv2d(in_channels, out_channels=h_dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        # Calculate the flattened size after convolutions (assuming 64x64 input)
        # 64 -> 32 -> 16 -> 8 -> 4. So final feature map size is 4x4
        self.final_feature_map_size = 4
        self.flattened_size = hidden_dims[-1] * (self.final_feature_map_size ** 2)

        # Fully connected layers for mean and log variance
        self.fc_mu = nn.Linear(self.flattened_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flattened_size, latent_dim)

        # --- Decoder ---
        modules = []
        self.decoder_input = nn.Linear(latent_dim, self.flattened_size)

        hidden_dims.reverse() # Reverse for decoder [256, 128, 64, 32]

        in_channels = hidden_dims[0] # Start with the last encoder hidden dim
        # Start from the second element since the first is handled by decoder_input
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_channels, hidden_dims[i+1], kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i+1]),
                    nn.LeakyReLU())
            )
            in_channels = hidden_dims[i+1]

        # Final layer to reconstruct the image
        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(hidden_dims[-1], hidden_dims[-1], kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(hidden_dims[-1]),
                nn.LeakyReLU(),
                nn.Conv2d(hidden_dims[-1], out_channels=input_channels, kernel_size=3, padding=1),
                nn.Sigmoid()) # Use Sigmoid for output normalized to [0, 1]
        )

        self.decoder = nn.Sequential(*modules)
        hidden_dims.reverse() # Reverse back for consistency if needed elsewhere

    def encode(self, x):
        """
        Encodes the input image into latent space parameters (mean and log variance).
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
        Decodes the latent vector z back into an image.
        Args:
            z (torch.Tensor): Latent vector (Batch, LatentDim).
        Returns:
            torch.Tensor: Reconstructed image tensor (Batch, Channels, Height, Width).
        """
        result = self.decoder_input(z)
        result = result.view(-1, self.encoder[-1][0].out_channels, self.final_feature_map_size, self.final_feature_map_size) # Reshape to match decoder input
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
        Args:
            x (torch.Tensor): Input image tensor.
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
        recon_loss = F.mse_loss(recon_x, x, reduction='sum') / x.shape[0] # Per batch item MSE
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.shape[0] # Per batch item KLD

        loss = recon_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss': recon_loss.detach(), 'KLD': kld_loss.detach()}


# Define the RNN Model (Memory Model 'M')
class RNNModel(nn.Module):
    def __init__(self, latent_dim=32, action_dim=4, rnn_hidden_dim=256, num_rnn_layers=1):
        """
        Recurrent Neural Network (LSTM) model.
        Predicts the next latent state z_{t+1} given current z_t, action a_t, and hidden state h_t.

        Args:
            latent_dim (int): Dimensionality of the VAE latent vector z.
            action_dim (int): Dimensionality of the action vector a.
            rnn_hidden_dim (int): Number of hidden units in the LSTM.
            num_rnn_layers (int): Number of LSTM layers.
        """
        super(RNNModel, self).__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.rnn_hidden_dim = rnn_hidden_dim
        self.num_rnn_layers = num_rnn_layers

        # LSTM layer
        # Input size is concatenation of latent z and action a
        self.lstm = nn.LSTM(latent_dim + action_dim, rnn_hidden_dim, num_rnn_layers, batch_first=True)

        # Output layer to predict the next latent state z_{t+1}
        # Predicts the mean of the next latent state directly for simplicity
        # (Could be extended to predict distribution parameters like MDN-RNN)
        self.fc_out = nn.Linear(rnn_hidden_dim, latent_dim)

    def forward(self, z, action, hidden_state):
        """
        Forward pass through the RNN model.
        Args:
            z (torch.Tensor): Current latent state (Batch, LatentDim).
            action (torch.Tensor): Current action (Batch, ActionDim).
            hidden_state (tuple): Previous hidden state (h_0, c_0) of the LSTM.
                                  Each has shape (NumLayers, Batch, HiddenDim).
                                  Pass None for the initial state.
        Returns:
            tuple: (predicted_next_z, next_hidden_state)
                   predicted_next_z (Batch, LatentDim)
                   next_hidden_state (tuple): (h_n, c_n)
        """
        # Ensure inputs are correctly shaped for batch_first=True LSTM
        # LSTM expects input shape (Batch, SeqLen, InputDim)
        # Here, SeqLen is 1 as we process one step at a time
        z = z.unsqueeze(1)         # (Batch, 1, LatentDim)
        action = action.unsqueeze(1) # (Batch, 1, ActionDim)

        # Concatenate z and action to form LSTM input
        lstm_input = torch.cat((z, action), dim=2) # (Batch, 1, LatentDim + ActionDim)

        # Pass through LSTM
        lstm_out, next_hidden_state = self.lstm(lstm_input, hidden_state)
        # lstm_out shape: (Batch, 1, HiddenDim)

        # Predict next latent state z_{t+1} from LSTM output
        # Squeeze the sequence length dimension
        predicted_next_z = self.fc_out(lstm_out.squeeze(1)) # (Batch, LatentDim)

        return predicted_next_z, next_hidden_state

    def init_hidden(self, batch_size, device):
        """
        Initializes the hidden state of the LSTM.
        Args:
            batch_size (int): The batch size.
            device: The torch device ('cpu' or 'cuda').
        Returns:
            tuple: Initial hidden state (h_0, c_0).
        """
        # The hidden state is a tuple (h_0, c_0)
        # h_0 shape: (num_layers * num_directions, batch, hidden_size)
        # c_0 shape: (num_layers * num_directions, batch, hidden_size)
        h_0 = torch.zeros(self.num_rnn_layers, batch_size, self.rnn_hidden_dim).to(device)
        c_0 = torch.zeros(self.num_rnn_layers, batch_size, self.rnn_hidden_dim).to(device)
        return (h_0, c_0)


# Define the Self Model (predicts curiosity reward)
class SelfModel(nn.Module):
    def __init__(self, latent_dim=32, rnn_hidden_dim=256, action_dim=4, fc_hidden_dim=128, output_dim=1):
        """
        A self-model that predicts curiosity rewards.
        Takes current latent state z_t, RNN hidden state h_t, and action a_t as input.

        Args:
            latent_dim (int): Dimensionality of the VAE latent vector z.
            rnn_hidden_dim (int): Dimensionality of the RNN hidden state h.
            action_dim (int): Dimensionality of the action vector a.
            fc_hidden_dim (int): Hidden dimension for fully connected layers.
            output_dim (int): Output dimension (1 for reward prediction).
        """
        super(SelfModel, self).__init__()

        # Input size is concatenation of z_t, h_t (only the hidden state h, not cell state c), and action a_t
        # h_t shape is (NumLayers, Batch, HiddenDim), we take the last layer's output
        input_dim = latent_dim + rnn_hidden_dim + action_dim

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(input_dim, fc_hidden_dim),
            nn.ReLU(),
            nn.Linear(fc_hidden_dim, fc_hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(fc_hidden_dim // 2, output_dim)  # Output predicted reward
        )

    def forward(self, z, h, action):
        """
        Forward pass through the self-model.
        Args:
            z (torch.Tensor): Current latent state (Batch, LatentDim).
            h (torch.Tensor): Current RNN hidden state (NumLayers, Batch, HiddenDim).
                               We use the output of the last layer.
            action (torch.Tensor): Action tensor (Batch, ActionDim).
        Returns:
            torch.Tensor: Predicted curiosity reward (Batch, OutputDim).
        """
        # Extract the hidden state from the last layer of the RNN
        # h shape is (num_layers, batch, hidden_dim) -> take h[-1]
        last_layer_h = h[-1] # Shape: (Batch, HiddenDim)

        # Concatenate z_t, h_t (last layer), and action a_t
        combined = torch.cat([z, last_layer_h, action], dim=1)

        # Process the combined features through fully connected layers
        predicted_reward = self.fc_layers(combined)
        return predicted_reward

