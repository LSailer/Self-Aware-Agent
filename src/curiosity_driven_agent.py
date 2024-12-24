import torch
import numpy as np
from models import WorldModel, SelfModel
import matplotlib.pyplot as plt
from torchvision import transforms as T

class CuriosityDrivenAgent:
    def __init__(self, actions):
        self.world_model = WorldModel()  # Adjust input size for images
        self.self_model = SelfModel()  
        self.optimizer_world = torch.optim.Adam(self.world_model.parameters(), lr=0.001)
        self.optimizer_self = torch.optim.Adam(self.self_model.parameters(), lr=0.001)
        self.history_buffer = []
        self.world_losses = []
        self.self_losses = []
        self.actions = actions

        # Define the transformation pipeline
        self.transform = T.Compose([
            T.ToPILImage(),  # Convert numpy array to PIL Image
            T.Resize((64, 64)),  # Resize to 64x64
            T.ToTensor(),  # Convert to tensor and normalize to [0, 1]
        ])

    def preprocess_camera_image(self, image):
        """
        Preprocess raw RGB camera image into tensor with the correct shape for the SelfModel.
        Args:
            image (numpy.ndarray): Raw RGB image (H, W, C).
        Returns:
            torch.Tensor: Preprocessed image tensor of shape (1, C, H, W).
        """
        # Apply transformations: resize, normalize, and add batch dimension
        image_tensor = self.transform(image)  # Output shape: (C, H, W)
        return image_tensor.unsqueeze(0)  # Add batch dimension: (1, C, H, W)

    def choose_action(self, epsilon=0.1):
        """
        Choose an action based on the self-model's predicted reward or random exploration.
        Args:
            epsilon (float): Probability of choosing a random action for exploration.
        Returns:
            action_key (str): Selected action name.
            action (torch.Tensor): Action tensor.
        """

        if self.last_processed_image.shape[1:] != (3, 64, 64):  # Ensure proper dimensions
            raise ValueError(f"Incorrect image shape: {self.last_processed_image.shape}")

        if np.random.rand() < epsilon:
            # Random exploration
            action_key = np.random.choice(list(self.actions.keys()))

        # Exploitation: Predict future rewards
        future_rewards = []
        for action_key, action in self.actions.items():
            action_tensor = torch.tensor(action, dtype=torch.float32).unsqueeze(0)  # Shape: (1, action_dim)
            predicted_reward = self.self_model(self.last_processed_image, action_tensor)
            future_rewards.append((predicted_reward.item(), action_key))
        _, action_key = max(future_rewards)  # Maximize predicted reward

        action_array = self.actions[action_key]
        return action_key, action_array

    def train_world_model(self, processed_image, action, next_processed_image):
        """
        Train the world model to predict the next image state.
        Args:
            processed_image (torch.Tensor): Current image tensor of shape (batch_size, C, H, W).
            action (torch.Tensor): Action tensor of shape (batch_size, action_dim).
            next_processed_image (torch.Tensor): Target next image tensor of shape (batch_size, C, H, W).
        Returns:
            float: Loss value.
        """
        # Debugging: Check shapes
        print(f"Processed image shape: {processed_image.shape}")
        print(f"Action tensor shape: {action.shape}")
        print(f"Next processed image shape: {next_processed_image.shape}")

        if processed_image.dim() != 4 or next_processed_image.dim() != 4 or action.dim() != 2:
            raise ValueError("Inputs to world model must have correct dimensions.")
        
        # Forward pass through the world model
        predicted_next_image = self.world_model(processed_image, action)

        # Calculate the loss
        loss = torch.nn.functional.mse_loss(predicted_next_image, next_processed_image)

        # Optimize the world model
        self.optimizer_world.zero_grad()
        loss.backward()
        self.optimizer_world.step()
        return loss.item()

    def train_self_model(self, processed_image, action, target_reward):
        """
        Train the self model to predict the reward for a given state and action.
        Args:
            processed_image (torch.Tensor): Input image of shape (batch_size, channels, height, width).
            action (torch.Tensor): Action tensor of shape (batch_size, action_dim).
            target_reward (torch.Tensor): Ground truth reward tensor of shape (batch_size, 1).
        Returns:
            float: Loss value for training step.
        """
        # Validate input shapes (debugging step)
        assert processed_image.dim() == 4, f"processed_image should be 4D, got {processed_image.shape}"
        assert action.dim() == 2, f"action should be 2D, got {action.shape}"
        assert target_reward.dim() == 2, f"target_reward should be 2D, got {target_reward.shape}"

        # Forward pass through the self model
        predicted_reward = self.self_model(processed_image, action)  # Shape: (batch_size, 1)

        # Calculate loss (Mean Squared Error)
        loss = torch.nn.functional.mse_loss(predicted_reward, target_reward)

        # Optimize the self model
        self.optimizer_self.zero_grad()
        loss.backward()
        self.optimizer_self.step()

        return loss.item()

    def calculate_curiosity_reward(self, predicted_next_image, actual_next_image):
        """
        Calculate the curiosity reward based on the difference between predicted and actual next images.
        Args:
            predicted_next_image (torch.Tensor): Predicted image tensor of shape (batch_size, C, H, W).
            actual_next_image (torch.Tensor): Actual image tensor of shape (batch_size, C, H, W).
        Returns:
            float: The curiosity reward.
        """
        # Calculate the L2 norm (Euclidean distance) between the predicted and actual images
        difference = predicted_next_image - actual_next_image
        curiosity_reward = torch.norm(difference, p=2).item()  # Compute the L2 norm as a scalar value
        max_norm = torch.sqrt(torch.tensor(predicted_next_image.numel()))  # Normalize by max possible norm
        return (curiosity_reward / max_norm).item()