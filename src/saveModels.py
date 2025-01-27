import torch
import torch.nn as nn

class WorldModel(nn.Module):
    def __init__(self, input_channels=3, action_dim=4, hidden_dim=64, output_channels=3):
        """
        A world model that predicts the next state from the current state and action.
        Args:
            input_channels (int): Number of input channels (e.g., 3 for RGB images).
            hidden_dim (int): Hidden dimension for intermediate layers.
            output_channels (int): Number of output channels (e.g., 3 for RGB images).
        """
        super(WorldModel, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1),  # Output: 16x64x64
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 64-2/2=16x32x32

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # Output: 32x 32 x 32
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 32x16x16
        )
        
        # Dynamically compute the input size for the fully connected layer
        self.flattened_size = 32 * 16 * 16  # 32x16x16 = 8192

        # Fully Connected Layers
        self.fc_layers = nn.Sequential(
            nn.Linear(self.flattened_size +  action_dim, hidden_dim),  # Combine image and action
            nn.ReLU(),
            nn.Linear(hidden_dim, self.flattened_size),  # Ausgangsgröße für Deconv
            nn.ReLU()
        )

        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),  # Output: 16xH/2xW/2
            nn.ReLU(),
            nn.ConvTranspose2d(16, output_channels, kernel_size=2, stride=2),  # Output: output_channelsxHxW
            nn.Sigmoid()  # Normalize to [0, 1]
        )

    def forward(self, image, action):
        """
        Forward pass of the world model.
        Args:
            image (torch.Tensor): Input image tensor of shape (batch_size, C, H, W).
            action (torch.Tensor): Action tensor of shape (batch_size, action_dim).
        Returns:
            torch.Tensor: Predicted next image of shape (batch_size, C, H, W).
        """
        # Pass the image through the convolutional layers
        conv_output = self.conv_layers(image)  # Shape: (batch_size, 32, H/4, W/4)

        # Flatten the convolutional output
        flattened_conv = conv_output.view(conv_output.size(0), -1)  # Shape: (batch_size, 32 * H/4 * W/4)

        # Concatenate the flattened convolutional output with the action tensor
        combined = torch.cat([flattened_conv, action], dim=1)  # Shape: (batch_size, 32 * H/4 * W/4 + action_dim)

        # Pass through the fully connected layers
        fc_output = self.fc_layers(combined)  # Shape: (batch_size, 32 * H/4 * W/4)

        # Reshape the output to match the deconvolution input
        reshaped = fc_output.view(conv_output.size(0), 32,  16, 16)  # Shape: (batch_size, 32, H/4, W/4)

        # Pass through the deconvolutional layers
        predicted_image = self.deconv_layers(reshaped)  # Shape: (batch_size, output_channels, H, W)
        return predicted_image


class SelfModel(nn.Module):
    def __init__(self, input_channels=3, action_dim=4, hidden_dim=64, output_dim=1):
        """
        A self-model that predicts curiosity rewards from camera images.
        Args:
            input_channels (int): Number of input channels (e.g., 3 for RGB images).
            hidden_dim (int): Hidden dimension for fully connected layers.
            output_dim (int): Output dimension (e.g., 1 for reward prediction).
        """
        super(SelfModel, self).__init__()

        # Convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1),  # Output: 16x64x64
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 16x32x32

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1), #16 Input, 32 Output Channels, # Output: 32x32x32
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 32x16x16
        )

        # Dynamically compute the input size for the fully connected layer
        self.flattened_size = 32 * 16 * 16  # 32x16x16 = 8192

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(self.flattened_size + action_dim, hidden_dim),  # Combine image and action
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)  # Output reward
        )

    def forward(self, image, action):
            """
            Forward pass through the self-model.
            Args:
                image (torch.Tensor): Input image of shape (batch_size, channels, height, width).
                action (torch.Tensor): Input action tensor of shape (batch_size, action_dim).
            Returns:
                torch.Tensor: Predicted curiosity reward.
            """
            # Process the image through convolutional layers
            conv_output = self.conv_layers(image)  # Shape: (batch_size, 32, 16, 16)

            # Flatten the convolutional output
            flattened_conv = conv_output.view(conv_output.size(0), -1)  # Shape: (batch_size, 8192)

            # Concatenate the flattened convolutional output with the action tensor
            combined = torch.cat([flattened_conv, action], dim=1)  # Shape: (batch_size, 8192 + action_dim)

            # Process the combined features through fully connected layers
            predicted_reward = self.fc_layers(combined)
            return predicted_reward



