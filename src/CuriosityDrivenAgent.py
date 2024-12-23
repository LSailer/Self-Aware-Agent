import torch
import numpy as np
from models import WorldModel, SelfModel
import matplotlib.pyplot as plt

class CuriosityDrivenAgent:
    def __init__(self, state_dim, action_dim, hidden_dim, actions):
        self.world_model = WorldModel(state_dim + action_dim, hidden_dim, state_dim)  # Fix input size
        self.self_model = SelfModel(state_dim + action_dim, hidden_dim, 1)  # Fix input size
        self.actions = actions
        self.optimizer_world = torch.optim.Adam(self.world_model.parameters(), lr=0.001)
        self.optimizer_self = torch.optim.Adam(self.self_model.parameters(), lr=0.001)
        self.belief = torch.zeros(state_dim)
        self.history_buffer = []
        self.world_losses = []
        self.self_losses = []

    def update_belief(self, observation, action):
        input_tensor = torch.cat([self.belief, action], dim=-1).unsqueeze(0)
        predicted_next_state = self.world_model(input_tensor)
        self.belief = observation * 0.5 + predicted_next_state.squeeze(0) * 0.5

    def choose_action(self, epsilon=0.1):
        if np.random.rand() < epsilon:
            action_key = np.random.choice(list(self.actions.keys()))
        else:
            future_losses = []
            for action_key, action in self.actions.items():
                action_tensor = torch.tensor(action, dtype=torch.float32)
                input_tensor = torch.cat([self.belief, action_tensor], dim=-1).unsqueeze(0)
                predicted_loss = self.self_model(input_tensor)
                future_losses.append((predicted_loss.item(), action_key))
            _, action_key = max(future_losses)
        return action_key, torch.tensor(self.actions[action_key], dtype=torch.float32)

    def train_world_model(self, input_tensor, next_state):
        predicted_next_state = self.world_model(input_tensor)
        loss = torch.nn.functional.mse_loss(predicted_next_state, next_state)

        self.optimizer_world.zero_grad()
        loss.backward()
        self.optimizer_world.step()

        self.world_losses.append(loss.item())
        return loss.item()

    def train_self_model(self, input_tensor, target_tensor):
        predicted_reward = self.self_model(input_tensor)
        loss = torch.nn.functional.mse_loss(predicted_reward, target_tensor)

        self.optimizer_self.zero_grad()
        loss.backward()
        self.optimizer_self.step()

        self.self_losses.append(loss.item())
        return loss.item()

    def calculate_curiosity_reward(self, predicted_next_state, actual_next_state):
        return torch.norm(predicted_next_state - actual_next_state, p=2).item()

    def observe_and_act(self, state, epsilon=0.1):
        action_key, action = self.choose_action(epsilon)
        return action_key, action

    def plot_losses(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.world_losses, label='World Model Loss')
        plt.plot(self.self_losses, label='Self Model Loss')
        plt.xlabel('Training Steps')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training Losses Over Time')
        plt.show()

# Local Test
if __name__ == "__main__":
    state_dim = 6
    action_dim = 3
    hidden_dim = 64
    actions = {
        "forward": [1, 0, 0],
        "backward": [-1, 0, 0],
        "left": [0, -1, 0],
        "right": [0, 1, 0]
    }

    agent = CuriosityDrivenAgent(state_dim, action_dim, hidden_dim, actions)

    # Test update_belief
    observation = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=torch.float32)
    action = torch.tensor([1, 0, 0], dtype=torch.float32)
    agent.update_belief(observation, action)
    print(f"Updated belief: {agent.belief}")

    # Test choose_action
    action_key, action_tensor = agent.choose_action(epsilon=0.1)
    print(f"Chosen action: {action_key}, Action tensor: {action_tensor}")


    for step in range(100):  # Beispielsweise 100 Schritte
        action_key, action_tensor = agent.choose_action(epsilon=0.1)
        input_tensor = torch.cat([observation, action_tensor], dim=-1).unsqueeze(0)
        next_state = torch.tensor([0.2, 0.3, 0.4, 0.5, 0.6, 0.7], dtype=torch.float32).unsqueeze(0)

        # Trainiere das World Model
        world_loss = agent.train_world_model(input_tensor, next_state)

        # Trainiere das Self Model
        target_tensor = torch.tensor([1.0], dtype=torch.float32).unsqueeze(0)
        self_loss = agent.train_self_model(input_tensor, target_tensor)

        print(f"Step {step}, World Model Loss: {world_loss}, Self Model Loss: {self_loss}")

    # Plot losses
    agent.plot_losses()