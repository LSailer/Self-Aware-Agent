import csv
from collections import deque
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd # Import pandas for easier CSV reading/plotting

class MetricLogger:
    def __init__(self, log_dir="logs", csv_filename="metrics_log_multi_agent.csv", plot_filename_base="multi_agent_plot"):
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
        os.makedirs(log_dir, exist_ok=True)

        # Define CSV fields
        self.fields = [
            "Step", "Agent_ID",
            "Position_X", "Position_Y", "Position_Z",
            "Velocity_X", "Velocity_Y", "Velocity_Z",
            "Orientation_W", "Orientation_X", "Orientation_Y", "Orientation_Z",
            "Angular_Velocity_X", "Angular_Velocity_Y", "Angular_Velocity_Z",
            "Action_Type",
            "Action_Vector_0", "Action_Vector_1", "Action_Vector_2", "Action_Vector_3",
            "Curiosity_Reward",
            "World_Loss",
            "Self_Loss",
            "VAE_Loss",
            "VAE_KLD_Loss",
            "Is_Interacting_Object",
            "Is_Interacting_With_Other_Agent"
        ]
        
        # Stores all metric entries as dictionaries
        self.all_metrics_data = [] 
        self._init_csv()

    def _init_csv(self):
        """Initialize the CSV file with headers if it doesn't exist."""
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
        agent_id, 
        action_type,
        action_vector,
        curiosity_reward, 
        world_loss,       
        self_loss,        
        vae_loss,         
        vae_kld_loss,     
        is_interacting_object,
        is_interacting_with_other_agent = None
    ):
        """Log metrics to a list and then to CSV."""
        # Ensure action_vector has the expected length (e.g., 4)
        action_vec_padded = list(action_vector) + [0] * (max(0, 4 - len(action_vector))) # Ensure at least 4 elements, pad with 0 if less

        log_entry = {
            "Step": step,
            "Agent_ID": agent_id,
            "Action_Type": action_type,
            "Action_Vector_0": action_vec_padded[0], "Action_Vector_1": action_vec_padded[1],
            "Action_Vector_2": action_vec_padded[2], "Action_Vector_3": action_vec_padded[3],
            "Curiosity_Reward": curiosity_reward,
            "World_Loss": world_loss,
            "Self_Loss": self_loss,
            "VAE_Loss": vae_loss,
            "VAE_KLD_Loss": vae_kld_loss,
            "Is_Interacting_Object": 1 if is_interacting_object else 0,
            "Is_Interacting_With_Other_Agent": 1 if is_interacting_with_other_agent else 0
        }
        
        self.all_metrics_data.append(log_entry)

        # Append to CSV
        try:
            write_header = not os.path.exists(self.csv_filepath) or os.path.getsize(self.csv_filepath) == 0
            with open(self.csv_filepath, mode="a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.fields)
                if write_header and not self.all_metrics_data: # Only write header if file is new AND no data yet (first log)
                     writer.writeheader()
                writer.writerow(log_entry)
        except IOError as e:
             print(f"Error writing to CSV {self.csv_filepath}: {e}")

    def plot_metrics(self, rolling_window=100):
        """Generate and save plots for metrics using pandas for easier handling."""
        if not self.all_metrics_data:
            print("No metrics logged to plot.")
            return

        df = pd.DataFrame(self.all_metrics_data)

        # Use a standard, available style
        try:
            plt.style.use('seaborn-v0_8-darkgrid') 
        except:
            print("Plot style 'seaborn-v0_8-darkgrid' not found, using default.")
            plt.style.use('ggplot') 

        # 1. Curiosity Reward Plot (Separate for each agent)
        plt.figure(figsize=(12, 7))
        color_map_reward = {0: "lightblue", 1: "lightcoral"}
        line_color_map_reward = {0: "blue", 1: "red"}
        for agent_id_val in sorted(df['Agent_ID'].unique()):
            agent_df = df[df['Agent_ID'] == agent_id_val].sort_values(by='Step')
            plt.plot(agent_df['Step'], agent_df['Curiosity_Reward'], 
                     label=f"Agent {agent_id_val} Curiosity Reward (Raw)", alpha=0.5, color=color_map_reward.get(agent_id_val, "gray"))
            plt.plot(agent_df['Step'], agent_df['Curiosity_Reward'].rolling(window=rolling_window, min_periods=1).mean(), 
                     label=f"Agent {agent_id_val} Curiosity Reward (Avg {rolling_window})", color=line_color_map_reward.get(agent_id_val, "black"))
        plt.xlabel("Steps")
        plt.ylabel("Reward")
        plt.title("Agent Curiosity Rewards over Steps")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, f"{self.plot_filename_base}_agent_rewards.png"))
        plt.close()

        # 2. Loss Plots (Self, World/RNN, VAE)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 14), sharex=True)
        
        # Subplot 1: Self and World/RNN Losses
        # World Loss (RNN Loss) - Shared, plot once
        df_world_loss = df[df['Agent_ID'] == 0][['Step', 'World_Loss']].drop_duplicates().sort_values(by='Step')
        ax1.plot(df_world_loss['Step'], df_world_loss['World_Loss'].rolling(window=rolling_window, min_periods=1).mean(),
                 label=f"World(RNN) Loss (Avg {rolling_window})", color="red")

        color_map_self_loss = {0: "green", 1: "darkorange"} 
        for agent_id_val in sorted(df['Agent_ID'].unique()):
            agent_df = df[df['Agent_ID'] == agent_id_val].sort_values(by='Step')
            ax1.plot(agent_df['Step'], agent_df['Self_Loss'].rolling(window=rolling_window, min_periods=1).mean(),
                     label=f"Agent {agent_id_val} Self Loss (Avg {rolling_window})", color=color_map_self_loss.get(agent_id_val, "black"))
        
        ax1.set_ylabel("Loss (Rolling Average)")
        ax1.set_title("Self Model and World Model (RNN) Losses")
        ax1.legend()
        ax1.grid(True)
        
        # Subplot 2: VAE Losses - Shared, plot once
        df_vae_loss = df[df['Agent_ID'] == 0][['Step', 'VAE_Loss', 'VAE_KLD_Loss']].drop_duplicates().sort_values(by='Step')
        ax2.plot(df_vae_loss['Step'], df_vae_loss['VAE_Loss'].rolling(window=rolling_window, min_periods=1).mean(),
                 label=f"Total VAE Loss (Avg {rolling_window})", color="purple")
        ax2.plot(df_vae_loss['Step'], df_vae_loss['VAE_KLD_Loss'].rolling(window=rolling_window, min_periods=1).mean(),
                 label=f"VAE KLD Loss (Avg {rolling_window})", color="magenta") # Changed color for KLD
        
        # Calculate and plot reconstruction loss (Total VAE Loss - KLD Loss)
        # Ensure 'VAE_Loss' and 'VAE_KLD_Loss' are numeric and handle potential NaNs
        df_vae_loss['VAE_Loss'] = pd.to_numeric(df_vae_loss['VAE_Loss'], errors='coerce')
        df_vae_loss['VAE_KLD_Loss'] = pd.to_numeric(df_vae_loss['VAE_KLD_Loss'], errors='coerce')
        recon_loss = df_vae_loss['VAE_Loss'] - df_vae_loss['VAE_KLD_Loss']
        ax2.plot(df_vae_loss['Step'], recon_loss.rolling(window=rolling_window, min_periods=1).mean(), 
                 label=f"VAE Recon Loss (Avg {rolling_window})", color="teal") # Changed color for Recon
        
        ax2.set_xlabel("Steps")
        ax2.set_ylabel("Loss (Rolling Average)")
        ax2.set_title("VAE Loss Components")
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, f"{self.plot_filename_base}_losses.png"))
        plt.close()

        # 3. Action Histogram (Combined for now, can be separated if needed)
        plt.figure(figsize=(10, 6))
        # To get a representative action distribution, we can concatenate actions from all agents
        # or plot them separately if action_types differ significantly per agent.
        # For now, let's plot combined action types.
        action_counts = df['Action_Type'].astype(str).value_counts()
        action_counts.plot(kind='bar', color="slategrey")
        plt.xlabel("Action Key")
        plt.ylabel("Frequency")
        plt.title("Action Histogram (All Agents, Total Counts)")
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis="y")
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, f"{self.plot_filename_base}_action_hist.png"))
        plt.close()

        # 4. Interaction with Objects (Combined - Any Agent with Any Object)
        # Group by Step and check if any agent is interacting with an object
        df_obj_interaction_combined = df.groupby('Step')['Is_Interacting_Object'].any().astype(int).reset_index()
        df_obj_interaction_combined.rename(columns={'Is_Interacting_Object': 'Any_Agent_Interacting_Object'}, inplace=True)
        df_obj_interaction_combined = df_obj_interaction_combined.sort_values(by='Step')

        plt.figure(figsize=(12, 7))
        rolling_mean_combined_obj = df_obj_interaction_combined['Any_Agent_Interacting_Object'].rolling(window=rolling_window, min_periods=1).mean()
        plt.plot(df_obj_interaction_combined['Step'], rolling_mean_combined_obj, label=f"Any Agent-Object Interaction Freq. (Avg {rolling_window} steps)", color="saddlebrown")
        plt.xlabel("Steps")
        plt.ylabel("Frequency (Rolling Average)")
        plt.title("Combined Agent-Object Interaction Frequency")
        plt.ylim(0, 1.05)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, f"{self.plot_filename_base}_obj_interaction_combined.png"))
        plt.close()

        # 5. Interaction with Objects (Separate for each agent)
        plt.figure(figsize=(12, 7))
        color_map_obj_interaction = {0: "darkgoldenrod", 1: "olive"}
        for agent_id_val in sorted(df['Agent_ID'].unique()):
            agent_df = df[df['Agent_ID'] == agent_id_val].sort_values(by='Step')
            interaction_freq = agent_df['Is_Interacting_Object'].rolling(window=rolling_window, min_periods=1).mean()
            plt.plot(agent_df['Step'], interaction_freq, 
                     label=f"Agent {agent_id_val} Object Interaction Freq. (Avg {rolling_window} steps)", 
                     color=color_map_obj_interaction.get(agent_id_val, "black"))
        plt.xlabel("Steps")
        plt.ylabel("Frequency (Rolling Average)")
        plt.title("Individual Agent-Object Interaction Frequency")
        plt.ylim(0, 1.05)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, f"{self.plot_filename_base}_obj_interaction_separate.png"))
        plt.close()

        # 6. Agent-Agent Interaction Frequency
        # This metric is shared, so we can filter by one agent or aggregate
        df_agent_interaction = df[df['Agent_ID'] == 0][['Step', 'Is_Interacting_With_Other_Agent']].copy().drop_duplicates().sort_values(by='Step')

        plt.figure(figsize=(12, 7))
        rolling_mean_agent_agent = df_agent_interaction['Is_Interacting_With_Other_Agent'].rolling(window=rolling_window, min_periods=1).mean()
        plt.plot(df_agent_interaction['Step'], rolling_mean_agent_agent, label=f"Agent-Agent Interaction Freq. (Avg {rolling_window} steps)", color="darkcyan")
        plt.xlabel("Steps")
        plt.ylabel("Interaction Frequency (Rolling Average)")
        plt.title("Agent-Agent Interaction Frequency")
        plt.ylim(0, 1.05)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, f"{self.plot_filename_base}_agent_agent_interaction_freq.png"))
        plt.close()

        print(f"Plots saved to {self.log_dir}")

    def close(self):
        """Handle any cleanup if necessary."""
        print(f"Metrics logged to {self.csv_filepath}")