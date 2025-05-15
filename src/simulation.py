import torch
from environment import Environment
from curiosity_driven_agent import CuriosityDrivenAgent # Agent now uses VAE/RNN
from video_recorder import VideoRecorder
from metric_logger import MetricLogger
import numpy as np

MAX_STEPS = 300
BATCH_SIZE = 64 # Should match agent's batch_size
UPDATE_EVERY_N_STEPS = 4 # How often to run model updates
INTERACTION_DISTANCE_THRESHOLD = 0.8 # Example distance threshold for interaction
EPSILON_GREEDY = 0.3 # Exploration rate

def check_interaction(env, threshold):
    """Checks if the agent is close to any interactable object."""
    try:
        state      = env.get_state()
        agent_pos  = np.array(state["agent"]["position"])
        cyl_pos    = np.array(state["cylinder"]["position"])
        disk_pos   = np.array(state["disk"]["position"])
        pyr_pos    = np.array(state["pyramid"]["position"])

        dist_cyl   = np.linalg.norm(agent_pos - cyl_pos)
        dist_disk  = np.linalg.norm(agent_pos - disk_pos)
        dist_pyr   = np.linalg.norm(agent_pos - pyr_pos)

        return (dist_cyl  < threshold) or \
               (dist_disk < threshold) or \
               (dist_pyr  < threshold)
    except Exception as e:
        print(f"Error checking interaction: {e}")
        return False



def run_simulation():
    # Use CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize the environment
    env = Environment()
    env.reset()

    # Initialize the agent with VAE/RNN
    agent = CuriosityDrivenAgent(
        actions=env.action_map,
        latent_dim=32,         # Example latent dimension for VAE
        rnn_hidden_dim=256,    # Example hidden dimension for RNN
        buffer_size=50000,     # Example buffer size
        batch_size=BATCH_SIZE,
        device=device
    )

    # Initialize the video recorder
    video_recorder = VideoRecorder(filename="camera_feed_vae_rnn.mp4")

    # Initialize the metric logger
    logger = MetricLogger() # Logger will be updated to handle interaction freq

    # --- Simulation Loop ---
    agent.reset_hidden_state() # Initialize RNN hidden state
    raw_image = env.get_camera_image()
    agent.current_z = agent.encode_image(raw_image) # Encode initial image

    print("Starting simulation...")
    for step in range(MAX_STEPS):
        # 1. Choose action based on current z_t and h_t
        action_key, action_array = agent.choose_action(epsilon=EPSILON_GREEDY) # Adjust epsilon as needed

        # 2. Apply action and step environment
        env.apply_action(action_key)
        env.step_simulation()
        done = False # Assuming non-terminating env for now, or add termination logic

        # 3. Get next state observation
        next_raw_image = env.get_camera_image()

        # 4. Store experience in replay buffer
        # For simplicity, store external reward as 0, focus on curiosity
        external_reward = 0.0
        agent.store_experience(raw_image, action_key, action_array, external_reward, next_raw_image, done)

        # 5. Update agent state (z_t and h_t) for the *next* step's action selection
        # Encode the *next* image to get z_{t+1}
        next_z = agent.encode_image(next_raw_image)
        # Use the RNN to get the next hidden state h_{t+1} based on z_t, a_t, h_t
        action_tensor = torch.tensor(action_array, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            _, next_h = agent.rnn_model(agent.current_z, action_tensor, agent.current_h)

        # Update agent's current state for the next iteration
        agent.current_z = next_z
        agent.current_h = next_h
        raw_image = next_raw_image # Roll over observation

        # 6. Perform model updates periodically
        loss_dict = None
        if step > BATCH_SIZE and step % UPDATE_EVERY_N_STEPS == 0:
            loss_dict = agent.update_models(visualize=step % 1000 == 0, step=step) # Train VAE, RNN, SelfModel

        # 7. Logging
        if loss_dict:
            # Check for interaction
            is_interacting = check_interaction(env, INTERACTION_DISTANCE_THRESHOLD)

            # Log metrics including interaction status
            logger.log_metrics(
                step=step,
                env=env, # Pass env to get state inside logger if needed, or pass state directly
                action_type=action_key,
                action_vector=action_array,
                # Use logged average curiosity from the update step
                curiosity_reward=loss_dict.get('avg_curiosity_reward', 0.0),
                world_loss=loss_dict.get('rnn_loss', 0.0), # Use RNN loss as "world loss"
                self_loss=loss_dict.get('self_loss', 0.0),
                vae_loss=loss_dict.get('vae_loss', 0.0),
                vae_kld_loss=loss_dict.get('vae_kld_loss', 0.0),
                is_interacting=is_interacting # Pass interaction status
            )
            print(f"Step {step}: Action: {action_key}, VAE Loss: {loss_dict.get('vae_loss', 0):.4f}, RNN Loss: {loss_dict.get('rnn_loss', 0):.4f}, Self Loss: {loss_dict.get('self_loss', 0):.4f}, Curiosity: {loss_dict.get('avg_curiosity_reward', 0):.4f}, Interacting: {is_interacting}")
        elif step % 100 == 0:
            print(f"Step {step}: Action: {action_key}")


        # 8. Video Recording (Annotate with available losses)
        # Get latest losses from logger if not updated this step
        current_rnn_loss = logger.world_losses[-1] if logger.world_losses else 0.0
        current_self_loss = logger.self_losses[-1] if logger.self_losses else 0.0
        current_curiosity = logger.rewards[-1] if logger.rewards else 0.0

        try:
            # Use next_raw_image for annotation as it's the result of the action
            annotated_frame = video_recorder.annotate_frame(
                next_raw_image, step, current_curiosity, current_self_loss # Pass relevant metrics
            )
            video_recorder.write_frame(annotated_frame)
        except Exception as e:
            print(f"Error annotating/writing frame at step {step}: {e}")


        # Check for termination condition if applicable
        # if done:
        #     env.reset()
        #     agent.reset_hidden_state()
        #     raw_image = env.get_camera_image()
        #     agent.current_z = agent.encode_image(raw_image)


    # --- Cleanup ---
    print("Simulation finished.")
    video_recorder.close()
    env.close()
    logger.plot_metrics() # Generate plots, including interaction frequency

    print("Logs and video saved.")

if __name__ == "__main__":
    run_simulation()

