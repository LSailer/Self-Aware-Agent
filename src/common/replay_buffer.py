from collections import deque, namedtuple
import random
import numpy as np

Experience = namedtuple('Experience',
                        ('obs_t', 'actions_t', 'rewards_t', 'obs_tp1', 'done'))

class ReplayBuffer:
    """A simple replay buffer for storing and sampling experiences."""

    def __init__(self, capacity: int):
        """
        Initialize the ReplayBuffer.

        Args:
            capacity (int): The maximum number of experiences to store.
        """
        self.buffer = deque(maxlen=capacity)

    def add(self, obs_t: list, actions_t: list, rewards_t: list, obs_tp1: list, done: bool):
        """
        Add a new experience to the buffer.

        Args:
            obs_t (list): List of observations for each agent at time t.
            actions_t (list): List of actions for each agent at time t.
            rewards_t (list): List of rewards for each agent at time t.
            obs_tp1 (list): List of next observations for each agent at time t+1.
            done (bool): Whether the episode has terminated.
        """
        # Note: We store raw numpy arrays for observations to save memory,
        # they will be converted to tensors during the model update step.
        exp = Experience(obs_t, actions_t, rewards_t, obs_tp1, done)
        self.buffer.append(exp)

    def sample(self, batch_size: int) -> Experience:
        """
        Randomly sample a batch of experiences from the buffer.

        Args:
            batch_size (int): The number of experiences to sample.

        Returns:
            Experience: A new Experience object where each field contains a
                        batch of the corresponding data.
        """
        if batch_size > len(self.buffer):
            raise ValueError(f"Cannot sample {batch_size} experiences, buffer only contains {len(self.buffer)}.")

        experiences = random.sample(self.buffer, batch_size)

        # Transpose the batch of experiences.
        # This converts a list of Experience tuples into a single Experience
        # tuple where each field contains a list of all corresponding values.
        # For example, batch.obs_t will be a list of all obs_t lists from the sampled experiences.
        batch = Experience(*zip(*experiences))
        return batch

    def __len__(self) -> int:
        """Return the current size of the buffer."""
        return len(self.buffer)
