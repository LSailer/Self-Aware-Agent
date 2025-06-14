from .base_config import *

# --- Agent and Controller Configuration ---
NUM_AGENTS = 1
CONTROLLER_TYPE = "single" # Signals the main script to use the SingleAgentController

# --- Logging Directory ---
# Creates a unique log directory for this experiment type
LOG_DIR = f"{LOG_DIR_BASE}/SingleAgent_Refactored_V1"
