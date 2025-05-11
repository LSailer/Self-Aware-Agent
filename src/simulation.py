import torch
import torch.nn.functional as F
from torchvision import transforms as T
from collections import deque, namedtuple
import numpy as np
import random
import os

from environment import Environment
from models import VAE, RNNModel, SelfModel
from multi_agent_controller import MultiAgentController
from video_recorder import VideoRecorder
from metric_logger import MetricLogger

# --- Constants ---
MAX_STEPS = 5000  # Reduced for initial testing
BATCH_SIZE = 32
UPDATE_EVERY_N_STEPS = 4
REPLAY_BUFFER_SIZE = 10000
LEARNING_RATE_VAE = 0.001
LEARNING_RATE_RNN = 0.001
LEARNING_RATE_SELF = 0.001
LATENT_DIM = 32
ACTION_DIM = 4  # Dimension of action vector per agent (vx, vy, 0, torque_z)
RNN_HIDDEN_DIM = 256
NUM_RNN_LAYERS = 1
NUM_AGENTS = 2
EPSILON_START = 0.95 # Start with more exploration
EPSILON_END = 0.05  # End with less exploration
EPSILON_DECAY = 30000 # Longer decay for epsilon
INTERACTION_DISTANCE_THRESHOLD = 0.8 # For agent-agent interaction
OBJECT_INTERACTION_THRESHOLD = 0.7 # For agent-object interaction (distance to object center)
LOG_DIR = "logs/multi_agent_v3" # Changed log dir to avoid overwriting
VAE_VISUALIZE_AFTER_STEPS= 100
RNN_VISUALIZE_AFTER_STEPS=100
USE_GUI =False

# --- Replay Buffer ---
Experience = namedtuple('Experience', (
    'obs_t1', 'obs_t2',             # Observations at time t (Agent 1, Agent 2)
    'action_t1', 'action_t2',       # Actions at time t (Agent 1, Agent 2)
    'reward_t1', 'reward_t2',       # External rewards (mostly 0 here)
    'obs_tp1', 'obs_tp2',           # Observations at time t+1
    'done'                          # Is the state terminal?
))
replay_buffer = deque(maxlen=REPLAY_BUFFER_SIZE)

# --- Image Preprocessing ---
transform = T.Compose([
    T.ToPILImage(),
    T.Resize((64, 64)),
    T.ToTensor(),  # Converts to [0, 1] and (C, H, W)
])

def preprocess_observation(obs_np):
    """Applies transformation to a single NumPy observation."""
    return transform(obs_np)


def run_multi_agent_simulation():
    # --- Initialization ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Environment
    env = Environment(use_gui=USE_GUI)
    agent_ids = [env.agent_id_1, env.agent_id_2]
    action_map = env.action_map
    print(f"Agent IDs: {agent_ids}")

    # Create models
    vae = VAE(latent_dim=LATENT_DIM).to(device)
    rnn_model = RNNModel(latent_dim=LATENT_DIM, action_dim=ACTION_DIM, 
                        rnn_hidden_dim=RNN_HIDDEN_DIM, num_agents=NUM_AGENTS).to(device)
    self_models = [SelfModel(latent_dim=LATENT_DIM, rnn_hidden_dim=RNN_HIDDEN_DIM, 
                           action_dim=ACTION_DIM).to(device) for _ in range(NUM_AGENTS)]

    # Create controllers (simple linear controllers for now)
    controller_input_dim = LATENT_DIM + RNN_HIDDEN_DIM
    controllers = [torch.nn.Linear(controller_input_dim, ACTION_DIM).to(device) 
                  for _ in range(NUM_AGENTS)]

    # Create MultiAgentController instance
    multi_agent_controller = MultiAgentController(
        vae=vae,
        rnn_model=rnn_model,
        self_models=self_models,
        controllers=controllers,
        num_agents=NUM_AGENTS,
        action_map=action_map,
        latent_dim=LATENT_DIM,
        action_dim=ACTION_DIM,
        rnn_hidden_dim=RNN_HIDDEN_DIM,
        device=device,
        replay_buffer_size=REPLAY_BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        learning_rate_vae=LEARNING_RATE_VAE,
        learning_rate_rnn=LEARNING_RATE_RNN,
        learning_rate_self=LEARNING_RATE_SELF,
        epsilon_start=EPSILON_START,
        epsilon_end=EPSILON_END,
        epsilon_decay=EPSILON_DECAY,
        vae_visualize_after_steps=VAE_VISUALIZE_AFTER_STEPS,
        log_dir=LOG_DIR,
        rnn_visualize_after_steps=RNN_VISUALIZE_AFTER_STEPS
    )

    # Logging & Video
    os.makedirs(LOG_DIR, exist_ok=True)
    
    logger = MetricLogger(log_dir=LOG_DIR, csv_filename="multi_agent_metrics_v2.csv", 
                         plot_filename_base="multi_agent_plot_v2")
    
    # Video recorders: Use the raw image from environment, not the transformed one for VAE
    recorders = [
        VideoRecorder(filename=os.path.join(LOG_DIR, f"agent_{i+1}_video.mp4"), 
                     resolution=(480,640), fps=20) # Use original resolution for recording
        for i in range(NUM_AGENTS)
    ]

    # --- Start Simulation ---
    env.reset()
    # Get initial observations
    obs_t_np_list = [env.get_camera_image(agent_id) for agent_id in agent_ids]
    # Set initial state in controller
    multi_agent_controller.set_initial_state(obs_t_np_list)

    print("Starting multi-agent simulation...")

    for step in range(MAX_STEPS):
        # --- 1. Choose actions for all agents ---
        action_keys, action_arrays_list, current_epsilon = multi_agent_controller.choose_actions()

        # --- 2. Execute actions in environment ---
        for i, agent_id in enumerate(agent_ids):
            env.apply_action(agent_id, action_keys[i])
        env.step_simulation()
        done = False

        # --- 3. Get next observations ---
        obs_tp1_np_list = [env.get_camera_image(agent_id) for agent_id in agent_ids]

        # --- 4. Store experience in controller's replay buffer ---
        multi_agent_controller.store_experience(
            obs_t_list=obs_t_np_list,
            action_arrays_list=action_arrays_list,
            reward_list=[0.0] * NUM_AGENTS, # External rewards
            obs_tp1_list=obs_tp1_np_list,
            done=done
        )

        # --- 5. Update RNN state and latent states in controller ---
        state_updated_successfully = multi_agent_controller.update_rnn_state(action_arrays_list, obs_tp1_np_list)
        
        if not state_updated_successfully:
            print(f"Warning: RNN state update failed at step {step}. Attempting to re-initialize controller state.")
            current_obs_for_reset = [env.get_camera_image(agent_id) for agent_id in agent_ids]
            multi_agent_controller.set_initial_state(current_obs_for_reset)
            obs_t_np_list = current_obs_for_reset
            continue

        # --- IMPORTANT: Set current observation for the *next* iteration ---
        obs_t_np_list = obs_tp1_np_list

        # --- 6. Perform model updates using the controller ---
        loss_dict = None
        if step > multi_agent_controller.batch_size and step % UPDATE_EVERY_N_STEPS == 0:
            loss_dict = multi_agent_controller.update_models(step)

        # --- 7. Logging ---
        if loss_dict:
            current_env_state = env.get_state()

            # Interaction Logic
            agent1_pos_np = np.array(current_env_state['agent_1']['position'])
            agent2_pos_np = np.array(current_env_state['agent_2']['position'])
            cube_pos_np = np.array(current_env_state['cube']['position'])
            cylinder_pos_np = np.array(current_env_state['cylinder']['position'])

            agent1_obj_interaction = (np.linalg.norm(agent1_pos_np - cube_pos_np) < OBJECT_INTERACTION_THRESHOLD or
                                      np.linalg.norm(agent1_pos_np - cylinder_pos_np) < OBJECT_INTERACTION_THRESHOLD)
            
            agent2_obj_interaction = (np.linalg.norm(agent2_pos_np - cube_pos_np) < OBJECT_INTERACTION_THRESHOLD or
                                      np.linalg.norm(agent2_pos_np - cylinder_pos_np) < OBJECT_INTERACTION_THRESHOLD)
            
            agent_agent_interaction = np.linalg.norm(agent1_pos_np - agent2_pos_np) < INTERACTION_DISTANCE_THRESHOLD

            try:
                # Log for Agent 0 (Controller's perspective, maps to env.agent_id_1)
                logger.log_metrics(
                    step=step, agent_id=0,
                    agent_pos=current_env_state['agent_1']['position'],
                    agent_vel=current_env_state['agent_1']['velocity'],
                    agent_ori=current_env_state['agent_1']['orientation'],
                    agent_ang_vel=current_env_state['agent_1']['angular_velocity'],
                    action_type=action_keys[0], action_vector=action_arrays_list[0],
                    curiosity_reward=loss_dict.get('avg_curiosity_reward_1', 0.0),
                    self_loss=loss_dict.get('self_loss_1', 0.0),
                    world_loss=loss_dict.get('rnn_loss', 0.0),
                    vae_loss=loss_dict.get('vae_loss', 0.0),
                    vae_kld_loss=loss_dict.get('vae_kld_loss', 0.0),
                    is_interacting_object=agent1_obj_interaction,
                    is_interacting_with_other_agent=agent_agent_interaction 
                )
                # Log for Agent 1 (Controller's perspective, maps to env.agent_id_2)
                logger.log_metrics(
                    step=step, agent_id=1,
                    agent_pos=current_env_state['agent_2']['position'],
                    agent_vel=current_env_state['agent_2']['velocity'],
                    agent_ori=current_env_state['agent_2']['orientation'],
                    agent_ang_vel=current_env_state['agent_2']['angular_velocity'],
                    action_type=action_keys[1], action_vector=action_arrays_list[1],
                    curiosity_reward=loss_dict.get('avg_curiosity_reward_2', 0.0),
                    self_loss=loss_dict.get('self_loss_2', 0.0),
                    world_loss=loss_dict.get('rnn_loss', 0.0),
                    vae_loss=loss_dict.get('vae_loss', 0.0),
                    vae_kld_loss=loss_dict.get('vae_kld_loss', 0.0),
                    is_interacting_object=agent2_obj_interaction,
                    is_interacting_with_other_agent=agent_agent_interaction 
                )
            except Exception as e:
                print(f"Error during logging at step {step}: {e}")

            print(f"Step {step}/{MAX_STEPS}, Eps: {current_epsilon:.3f} | "
                  f"VAE: {loss_dict.get('vae_loss', 0):.4f} (R:{loss_dict.get('vae_recon_loss',0):.4f}, KLD:{loss_dict.get('vae_kld_loss',0):.4f}), "
                  f"RNN: {loss_dict.get('rnn_loss', 0):.4f} (H1:{loss_dict.get('rnn_loss_head1',0):.4f}, H2:{loss_dict.get('rnn_loss_head2',0):.4f}), "
                  f"Self1: {loss_dict.get('self_loss_1', 0):.4f} (Cur1:{loss_dict.get('avg_curiosity_reward_1',0):.4f}), "
                  f"Self2: {loss_dict.get('self_loss_2', 0):.4f} (Cur2:{loss_dict.get('avg_curiosity_reward_2',0):.4f})")
        
        elif step % 200 == 0:
            print(f"Step {step}/{MAX_STEPS}, Eps: {current_epsilon:.3f} (No model update this step)")

        # --- 8. Video Recording ---
        if obs_t_np_list[0] is not None and obs_t_np_list[1] is not None:
            # Safely get latest curiosity and self-loss for annotation
            curiosity_reward_1_ann = loss_dict.get('avg_curiosity_reward_1', 0.0) if loss_dict else 0.0
            self_loss_1_ann = loss_dict.get('self_loss_1', 0.0) if loss_dict else 0.0
            curiosity_reward_2_ann = loss_dict.get('avg_curiosity_reward_2', 0.0) if loss_dict else 0.0
            self_loss_2_ann = loss_dict.get('self_loss_2', 0.0) if loss_dict else 0.0
            
            # If loss_dict is None, try to get recent values from logger
            if not loss_dict and logger.all_metrics_data:
                last_metrics_agent1 = next((m for m in reversed(logger.all_metrics_data) if m['Agent_ID'] == 0), None)
                last_metrics_agent2 = next((m for m in reversed(logger.all_metrics_data) if m['Agent_ID'] == 1), None)
                if last_metrics_agent1:
                    curiosity_reward_1_ann = last_metrics_agent1.get('Curiosity_Reward', 0.0)
                    self_loss_1_ann = last_metrics_agent1.get('Self_Loss', 0.0)
                if last_metrics_agent2:
                    curiosity_reward_2_ann = last_metrics_agent2.get('Curiosity_Reward', 0.0)
                    self_loss_2_ann = last_metrics_agent2.get('Self_Loss', 0.0)

            try:
                # Annotate and write frame for agent 1
                frame1_annotated = recorders[0].annotate_frame(obs_t_np_list[0].copy(), step, 
                                                             curiosity_reward_1_ann, self_loss_1_ann)
                if frame1_annotated is not None:
                    recorders[0].write_frame(frame1_annotated)
                
                # Annotate and write frame for agent 2
                frame2_annotated = recorders[1].annotate_frame(obs_t_np_list[1].copy(), step, 
                                                             curiosity_reward_2_ann, self_loss_2_ann)
                if frame2_annotated is not None:
                    recorders[1].write_frame(frame2_annotated)
            except Exception as e:
                print(f"Error during video recording at step {step}: {e}")
                print(f"  Frame 1 shape: {obs_t_np_list[0].shape if obs_t_np_list[0] is not None else 'None'}, dtype: {obs_t_np_list[0].dtype if obs_t_np_list[0] is not None else 'None'}")
                print(f"  Frame 2 shape: {obs_t_np_list[1].shape if obs_t_np_list[1] is not None else 'None'}, dtype: {obs_t_np_list[1].dtype if obs_t_np_list[1] is not None else 'None'}")
        else:
            print(f"Skipping video frame at step {step} due to None observation(s).")

        # --- GUI Update (PyBullet specific, if running with GUI) ---
        # time.sleep(1./240.) # Slows down simulation if uncommented

        if step % 100== 0 and step > 0: # Plot metrics periodically
             logger.plot_metrics()

    # --- Cleanup ---
    print("Simulation finished.")
    for recorder in recorders:
        recorder.close()
    env.close()
    logger.plot_metrics() # Final plots
    logger.close()
    print(f"Logs and videos saved to {LOG_DIR}")

if __name__ == "__main__":
    run_multi_agent_simulation()

