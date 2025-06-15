from .base_config import *

# --- Agent and Controller Configuration ---
NUM_AGENTS = 2
CONTROLLER_TYPE = "multi" # Signals the main script to use the MultiAgentController

# --- Logging Directory ---
# Creates a unique log directory for this experiment type
LOG_DIR = f"{LOG_DIR_BASE}/MultiAgent_Refactored_V1"
