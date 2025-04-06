import csv
from collections import deque
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd # Import pandas for easier CSV reading/plotting

class MetricLogger:
    def __init__(self, log_dir="logs", csv_filename="metrics_log_vae_rnn.csv", plot_filename_base="plot_vae_rnn"):
        """
        Initializes the MetricLogger.

        Args:
            log_dir (str): Directory to save logs and plots.
            csv_filename (str): Name of the CSV log file.
            plot_filename_base (str): Base name for saved plot files.
        """
        self.log_dir = log_dir
        self.csv_filepath = os.path.join(log_dir, csv_filename)
        self.plot_filename_base = plot_filename_base
        self.metrics_buffer = deque(maxlen=100) # For rolling averages/trends if needed
        os.makedirs(log_dir, exist_ok=True)

        # Define CSV fields - Added VAE losses and interaction flag
        self.fields = [
            "Step",
            "Position_X", "Position_Y", "Position_Z",
            "Velocity_X", "Velocity_Y", "Velocity_Z",
            "Orientation_W", "Orientation_X", "Orientation_Y", "Orientation_Z", # Assuming quaternion
            "Angular_Velocity_X", "Angular_Velocity_Y", "Angular_Velocity_Z",
            "Action_Type",
            "Action_Vector_0", "Action_Vector_1", "Action_Vector_2", "Action_Vector_3", # Assuming action_dim=4
            "Curiosity_Reward",
            "World_Loss", # Now RNN Loss
            "Self_Loss",
            "VAE_Loss",
            "VAE_KLD_Loss",
            "Is_Interacting" # 0 or 1 flag
        ]
        self._init_csv()

        # Store metrics in lists for plotting (consider memory for very long runs)
        self.steps = []
        self.rewards = []
        self.self_losses = []
        self.world_losses = [] # RNN losses
        self.vae_losses = []
        self.vae_kld_losses = []
        self.actions = []
        self.interactions = [] # Store interaction flags

    def _init_csv(self):
        """Initialize the CSV file with headers if it doesn't exist."""
        # Check if file exists and is empty or doesn't exist
        write_header = not os.path.exists(self.csv_filepath) or os.path.getsize(self.csv_filepath) == 0
        try:
            with open(self.csv_filepath, mode="a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.fields)
                if write_header:
                    writer.writeheader()
        except IOError as e:
             print(f"Error initializing CSV {self.csv_filepath}: {e}")


    def log_metrics(
        self,
        step,
        env, # Pass environment to get state here
        action_type,
        action_vector,
        curiosity_reward,
        world_loss, # RNN loss
        self_loss,
        vae_loss,
        vae_kld_loss,
        is_interacting # Boolean or 0/1
    ):
        """Log metrics to CSV and store for plotting."""
        try:
            state = env.get_state()
            agent_pos = state["agent"]["position"]
            agent_vel = state["agent"]["velocity"]
            agent_ori = state["agent"]["orientation"] # Assuming quaternion (w, x, y, z)
            agent_ang_vel = state["agent"]["angular_velocity"]
        except Exception as e:
            print(f"Error getting state from env in logger: {e}")
            agent_pos = (0,0,0)
            agent_vel = (0,0,0)
            agent_ori = (1,0,0,0) # Default quaternion
            agent_ang_vel = (0,0,0)

        # Ensure action_vector has the expected length (e.g., 4)
        action_vec_padded = list(action_vector) + [0] * (4 - len(action_vector))


        log_entry = {
            "Step": step,
            "Position_X": agent_pos[0], "Position_Y": agent_pos[1], "Position_Z": agent_pos[2],
            "Velocity_X": agent_vel[0], "Velocity_Y": agent_vel[1], "Velocity_Z": agent_vel[2],
            "Orientation_W": agent_ori[3], "Orientation_X": agent_ori[0], # Order might depend on pybullet output
            "Orientation_Y": agent_ori[1], "Orientation_Z": agent_ori[2],
            "Angular_Velocity_X": agent_ang_vel[0], "Angular_Velocity_Y": agent_ang_vel[1], "Angular_Velocity_Z": agent_ang_vel[2],
            "Action_Type": action_type,
            "Action_Vector_0": action_vec_padded[0], "Action_Vector_1": action_vec_padded[1],
            "Action_Vector_2": action_vec_padded[2], "Action_Vector_3": action_vec_padded[3],
            "Curiosity_Reward": curiosity_reward,
            "World_Loss": world_loss,
            "Self_Loss": self_loss,
            "VAE_Loss": vae_loss,
            "VAE_KLD_Loss": vae_kld_loss,
            "Is_Interacting": 1 if is_interacting else 0
        }

        # Append to CSV
        try:
            # Check header again just before writing row if lists are empty (can happen if init failed)
            write_header = not os.path.exists(self.csv_filepath) or os.path.getsize(self.csv_filepath) == 0
            with open(self.csv_filepath, mode="a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.fields)
                if write_header:
                     writer.writeheader()
                writer.writerow(log_entry)
        except IOError as e:
             print(f"Error writing to CSV {self.csv_filepath}: {e}")


        # Store metrics for plotting
        self.steps.append(step)
        self.rewards.append(curiosity_reward)
        self.self_losses.append(self_loss)
        self.world_losses.append(world_loss)
        self.vae_losses.append(vae_loss)
        self.vae_kld_losses.append(vae_kld_loss)
        self.actions.append(action_type)
        self.interactions.append(1 if is_interacting else 0)

    def plot_metrics(self, rolling_window=100):
        """Generate and save plots for metrics using pandas for easier handling."""
        if not self.steps:
            print("No metrics logged to plot.")
            return

        # Create DataFrame for easier plotting and rolling averages
        df = pd.DataFrame({
            'Step': self.steps,
            'Reward': self.rewards,
            'Self_Loss': self.self_losses,
            'World_Loss': self.world_losses, # RNN Loss
            'VAE_Loss': self.vae_losses,
            'VAE_KLD_Loss': self.vae_kld_losses,
            'Action': self.actions,
            'Interaction': self.interactions
        })

        # --- Plotting ---
        # Use a standard, available style
        try:
            # plt.style.use('seaborn-v0_8-darkgrid') # Use this if available
            plt.style.use('ggplot') # Fallback style
        except:
             print("Plot style 'seaborn-v0_8-darkgrid' not found, using default.")
             # Keep default style if seaborn styles are unavailable

        # 1. Reward Plot (with rolling average)
        plt.figure(figsize=(12, 7))
        plt.plot(df['Step'], df['Reward'], label="Curiosity Reward (Raw)", alpha=0.6, color="lightblue")
        plt.plot(df['Step'], df['Reward'].rolling(window=rolling_window, min_periods=1).mean(), label=f"Reward (Avg {rolling_window} steps)", color="blue")
        plt.xlabel("Steps")
        plt.ylabel("Reward")
        plt.title("Curiosity Reward over Steps")
        plt.legend()
        plt.grid(True) # Ensure grid is on
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, f"{self.plot_filename_base}_reward.png"))
        plt.close()

        # 2. Loss Plot (Self, World/RNN, VAE)
        plt.figure(figsize=(12, 7))
        plt.plot(df['Step'], df['Self_Loss'].rolling(window=rolling_window, min_periods=1).mean(), label=f"Self Loss (Avg {rolling_window})", color="green")
        plt.plot(df['Step'], df['World_Loss'].rolling(window=rolling_window, min_periods=1).mean(), label=f"World(RNN) Loss (Avg {rolling_window})", color="red")
        plt.plot(df['Step'], df['VAE_Loss'].rolling(window=rolling_window, min_periods=1).mean(), label=f"VAE Loss (Avg {rolling_window})", color="purple")
        # Optional: Plot KLD separately if scale is different
        # plt.plot(df['Step'], df['VAE_KLD_Loss'].rolling(window=rolling_window, min_periods=1).mean(), label=f"VAE KLD (Avg {rolling_window})", color="orange")
        plt.xlabel("Steps")
        plt.ylabel("Loss (Rolling Average)")
        plt.title("Model Losses over Steps")
        # plt.ylim(0, max(1.0, df['Self_Loss'].quantile(0.98), df['World_Loss'].quantile(0.98))) # Adjust ylim if needed
        plt.legend()
        plt.grid(True) # Ensure grid is on
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, f"{self.plot_filename_base}_losses.png"))
        plt.close()

        # 3. Action Histogram
        plt.figure(figsize=(10, 6))
        # Ensure actions are treated as strings for categorical plotting
        action_counts = df['Action'].astype(str).value_counts()
        action_counts.plot(kind='bar', color="purple")
        plt.xlabel("Action Key")
        plt.ylabel("Frequency")
        plt.title("Action Histogram (Total Counts)")
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis="y")
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, f"{self.plot_filename_base}_action_hist.png"))
        plt.close()

        # 4. Interaction Frequency Plot
        plt.figure(figsize=(12, 7))
        interaction_freq = df['Interaction'].rolling(window=rolling_window, min_periods=1).mean()
        plt.plot(df['Step'], interaction_freq, label=f"Interaction Freq. (Avg {rolling_window} steps)", color="orange")
        plt.xlabel("Steps")
        plt.ylabel("Interaction Frequency (Rolling Average)")
        plt.title("Agent Interaction Frequency over Steps")
        plt.ylim(0, 1.05) # Frequency is between 0 and 1, add slight margin
        plt.legend()
        plt.grid(True) # Ensure grid is on
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, f"{self.plot_filename_base}_interaction_freq.png"))
        plt.close()

        print(f"Plots saved to {self.log_dir}")

    def close(self):
        """Handle any cleanup if necessary."""
        print(f"Metrics logged to {self.csv_filepath}")

