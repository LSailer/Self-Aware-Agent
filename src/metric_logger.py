import csv
from collections import deque
from torch.utils.tensorboard import SummaryWriter

class MetricLogger:
    def __init__(self, log_file="metrics_log.csv", tensorboard_log_dir="runs/metrics"):
        self.log_file = log_file
        self.metrics = deque(maxlen=100)  # Store recent metrics for trend analysis
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
        # Initialize TensorBoard writer
        self.writer = SummaryWriter(log_dir=tensorboard_log_dir)

    def _init_csv(self):
        """Initialize the CSV file with headers."""
        with open(self.log_file, mode="w", newline="") as f:
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

        with open(self.log_file, mode="a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fields)
            writer.writerow(log_entry)

        # Log scalar metrics to TensorBoard
        self.writer.add_scalar("Curiosity_Reward", curiosity_reward, step)
        self.writer.add_scalar("Reward_Trend", reward_trend, step)
        self.writer.add_scalar("World_Loss", world_loss, step)
        self.writer.add_scalar("Self_Loss", self_loss, step)

        # Log histograms for actions and rewards
        self.writer.add_histogram("Actions/Action_Vector", action_vector, step)
        self.writer.add_histogram("Rewards/Curiosity_Reward", curiosity_reward, step)

        # Subcategory for agent state
        self.writer.add_scalar("Agent/Position_X", position[0], step)
        self.writer.add_scalar("Agent/Position_Y", position[1], step)
        self.writer.add_scalar("Agent/Position_Z", position[2], step)
        self.writer.add_scalar("Agent/Velocity_X", velocity[0], step)
        self.writer.add_scalar("Agent/Velocity_Y", velocity[1], step)
        self.writer.add_scalar("Agent/Velocity_Z", velocity[2], step)

    def close(self):
        """Handle any cleanup if necessary."""
        print(f"Metrics logged to {self.log_file}")
