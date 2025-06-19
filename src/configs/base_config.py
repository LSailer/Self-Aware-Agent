# configs/base_config.py
"""
Base configuration file holding all shared parameters for simulations.
Specific configurations for single-agent or multi-agent setups will
import from this file and override parameters as needed.
"""

from common.enums import ActionSelection

# --- Simulation Parameters ---
MAX_STEPS = 200
BATCH_SIZE = 16
UPDATE_EVERY_N_STEPS = 4
REPLAY_BUFFER_SIZE = 10000
USE_GUI = False # Set to True to watch the simulation in PyBullet's GUI

# --- Action Selection Parameters ---
# Options: "epsilon_greedy", "boltzmann", "ucb"
ACTION_SELECTION_TYPE = ActionSelection.EPSILON_GREEDY
EPSILON_GREEDY = 0.3  # Epsilon for exploration in epsilon-greedy
TEMPERATURE = 1.0     # Temperature for Boltzmann exploration
UCB_C = 1.0           # Exploration constant for UCB

# --- Model Hyperparameters ---
LATENT_DIM = 32
ACTION_DIM = 4  # Dimension of action vector per agent (vx, vy, 0, torque_z)
RNN_HIDDEN_DIM = 256
NUM_RNN_LAYERS = 1

# --- Learning Rates ---
LEARNING_RATE_VAE = 0.001
LEARNING_RATE_RNN = 0.001
LEARNING_RATE_SELF = 0.001

# --- Logging and Visualization ---
LOG_DIR_BASE = "logs"
VAE_VISUALIZE_AFTER_STEPS = 20 # How often to save VAE reconstruction images
RNN_VISUALIZE_AFTER_STEPS = 20 # How often to save RNN prediction images
