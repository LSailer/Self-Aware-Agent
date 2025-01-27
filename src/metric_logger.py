import csv
from collections import deque
import os
from matplotlib import pyplot as plt
import numpy as np
import torch

class MetricLogger:
    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        self.metrics = deque(maxlen=100)  # Store recent metrics for trend analysis
        os.makedirs(log_dir, exist_ok=True)
        self.fields = [
            "Step",
            "Position_X",
            "Position_Y",
            "Position_Z",
            "Velocity_X",
            "Velocity_Y",
            "Velocity_Z",
            "Orientation",
            "Angular_Velocity_X",
            "Angular_Velocity_Y",
            "Angular_Velocity_Z",
            "Action_Type",
            "Action_Vector",
            "Curiosity_Reward",
            "World_Loss",
            "Self_Loss",
            "Reward_Trend"
        ]
        self._init_csv()
        self.steps = []
        self.rewards = []
        self.self_losses = []
        self.world_losses = []
        self.actions = []

    def _init_csv(self):
        """Initialize the CSV file with headers."""
        with open(os.path.join(self.log_dir, "metrics_log.csv"), mode="w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fields)
            writer.writeheader()
        

    def log_metrics(
        self,
        step,
        position,
        velocity,
        orientation,
        angular_velocity,
        action_type,
        action_vector,
        curiosity_reward,
        world_loss,
        self_loss
    ):
        """Log metrics to CSV and update trends."""
        self.metrics.append(curiosity_reward)
        reward_trend = sum(self.metrics) / len(self.metrics)  # Calculate rolling average

        log_entry = {
            "Step": step,
            "Position_X": position[0],
            "Position_Y": position[1],
            "Position_Z": position[2],
            "Velocity_X": velocity[0],
            "Velocity_Y": velocity[1],
            "Velocity_Z": velocity[2],
            "Orientation": orientation,
            "Angular_Velocity_X": angular_velocity[0],
            "Angular_Velocity_Y": angular_velocity[1],
            "Angular_Velocity_Z": angular_velocity[2],
            "Action_Type": action_type,
            "Action_Vector": action_vector,
            "Curiosity_Reward": curiosity_reward,
            "World_Loss": world_loss,
            "Self_Loss": self_loss,
            "Reward_Trend": reward_trend,
        }

        with open(os.path.join(self.log_dir, "metrics_log.csv"), mode="a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fields)
            writer.writerow(log_entry)

        # Store metrics
        self.steps.append(step)
        self.rewards.append(curiosity_reward)
        self.self_losses.append(self_loss)
        self.world_losses.append(world_loss)
        self.actions.append(action_type)


    def plot_metrics(self):
        """Generate and save plots for metrics."""
        # Reward Plot
        plt.figure(figsize=(10, 6))
        plt.plot(self.steps, self.rewards, label="Reward", color="blue")
        plt.xlabel("Steps")
        plt.ylabel("Reward")
        plt.title("Reward over Steps")
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(self.log_dir, "reward_plot.png"))
        plt.close()

        # Loss Plot
        plt.figure(figsize=(10, 6))
        plt.plot(self.steps, self.self_losses, label="Self Loss", color="green")
        plt.plot(self.steps, self.world_losses, label="World Loss", color="red")
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.title("Losses over Steps")
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(self.log_dir, "loss_plot.png"))
        plt.close()

        # Action Histogram
        plt.figure(figsize=(10, 6))
        plt.hist(self.actions, bins=range(len(set(self.actions)) + 1), align="left", rwidth=0.8, color="purple")
        plt.xlabel("Action Key")
        plt.ylabel("Frequency")
        plt.title("Action Histogram")
        plt.xticks(range(len(set(self.actions))), sorted(set(self.actions)))
        plt.grid(axis="y")
        plt.savefig(os.path.join(self.log_dir, "action_histogram.png"))
        plt.close()

    def close(self):
        """Handle any cleanup if necessary."""
        print(f"Metrics logged to {self.log_file}")
