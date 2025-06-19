import argparse
import importlib
import os
import torch
import pybullet as p

from envs.environment import Environment
from controllers.single_agent_controller import SingleAgentController
from controllers.multi_agent_controller import MultiAgentController
from common.metric_logger import MetricLogger
from common.video_recorder import VideoRecorder

def check_agent_object_interaction(env, agent_id):
    """Checks if a specific agent is in contact with any object."""
    state = env.get_state()
    for obj_name in state['objects']:
        obj_id = getattr(env, f"{obj_name}_id")
        contacts = p.getContactPoints(bodyA=agent_id, bodyB=obj_id)
        if contacts:
            return True
    return False

def check_agent_agent_interaction(env):
    """Checks if any two agents are in contact with each other."""
    if len(env.agent_ids) < 2:
        return False
    for i in range(len(env.agent_ids)):
        for j in range(i + 1, len(env.agent_ids)):
            contacts = p.getContactPoints(bodyA=env.agent_ids[i], bodyB=env.agent_ids[j])
            if contacts:
                return True
    return False

def run(config):
    """
    Sets up and runs the entire simulation based on the provided config.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Setup Environment, Logger, and Recorders ---
    os.makedirs(config.LOG_DIR, exist_ok=True)
    env = Environment(use_gui=config.USE_GUI, num_agents=config.NUM_AGENTS)
    logger = MetricLogger(log_dir=config.LOG_DIR, csv_filename="metrics.csv", plot_filename_base="plots")
    recorders = [
        VideoRecorder(
            filename=os.path.join(config.LOG_DIR, f"agent_{i}_video.mp4"),
            resolution=(640,480)
        ) for i in range(config.NUM_AGENTS)
    ]

    # --- Instantiate Controller based on Config ---
    if config.CONTROLLER_TYPE == "single":
        controller = SingleAgentController(config, env.action_map, device)
    elif config.CONTROLLER_TYPE == "multi":
        controller = MultiAgentController(config, env.action_map, device)
    else:
        raise ValueError(f"Unknown controller type in config: {config.CONTROLLER_TYPE}")

    print(f"Initialized {config.CONTROLLER_TYPE} controller for {config.NUM_AGENTS} agent(s).")

    # --- Main Simulation Loop ---
    last_loss_dict = {}
    try:
        env.reset()
        obs_t_list = [env.get_camera_image(agent_id) for agent_id in env.agent_ids]
        controller.set_initial_state(obs_t_list)

        print("Starting simulation...")
        for step in range(config.MAX_STEPS):
            # 1. Choose actions using the controller's unified API
            controller.current_step = step
            action_keys, action_arrays = controller.choose_actions(obs_t_list)

            # 2. Apply actions in the environment
            for i, agent_id in enumerate(env.agent_ids):
                env.apply_action(agent_id, action_keys[i])
            env.step_simulation()
            done = False # Or implement a real termination condition

            # 3. Get next observations
            obs_tp1_list = [env.get_camera_image(agent_id) for agent_id in env.agent_ids]

            # 4. Store experience in the controller's buffer
            controller.store_experience(obs_t_list, action_arrays, [0.0]*config.NUM_AGENTS, obs_tp1_list, done)

            # 5. Update controller's internal state (RNN hidden state)
            controller.update_rnn_state(action_arrays, obs_tp1_list)
            obs_t_list = obs_tp1_list

            # 6. Periodically update models
            loss_dict = None
            if step > config.BATCH_SIZE and step % config.UPDATE_EVERY_N_STEPS == 0:
                loss_dict = controller.update_models()

                if loss_dict:
                    agent_agent_interaction = check_agent_agent_interaction(env)

                    for i, agent_id in enumerate(env.agent_ids):
                        agent_obj_interaction = check_agent_object_interaction(env, agent_id)
                        last_loss_dict = loss_dict
                        logger.log_metrics(
                            step=step, agent_id=i,
                            action_type=action_keys[i],
                            # Use .get() for safe access to loss values
                            curiosity_reward=loss_dict.get(f'avg_curiosity_reward_{i}', loss_dict.get('avg_curiosity_reward', 0.0)),
                            self_loss=loss_dict.get(f'self_loss_{i}', loss_dict.get('self_loss', 0.0)),
                            world_loss=loss_dict.get('rnn_loss', 0.0),
                            vae_loss=loss_dict.get('vae_loss', 0.0),
                            vae_kld_loss=loss_dict.get('vae_kld_loss', 0.0),
                            is_interacting_object=agent_obj_interaction,
                            is_interacting_with_other_agent=agent_agent_interaction
                        )
                
                print(f"Step {step}/{config.MAX_STEPS} | VAE Loss: {loss_dict.get('vae_loss', 0):.4f} | RNN Loss: {loss_dict.get('rnn_loss', 0):.4f}")

                # 7. Video Recording
            for i in range(config.NUM_AGENTS):
                cur_reward = last_loss_dict.get(f'avg_curiosity_reward_{i}', last_loss_dict.get('avg_curiosity_reward', 0.0)) if last_loss_dict else 0.0
                self_loss = last_loss_dict.get(f'self_loss_{i}', last_loss_dict.get('self_loss', 0.0)) if last_loss_dict else 0.0
                
                annotated_frame = recorders[i].annotate_frame(obs_tp1_list[i], step, cur_reward, self_loss)
                if annotated_frame is not None:
                    recorders[i].write_frame(annotated_frame)
            
            # Periodically generate plots
            if step % 5000 == 0 and step > 0:
                logger.plot_metrics()

    finally:
        # --- Cleanup ---
        print("Simulation finished. Cleaning up...")
        for recorder in recorders:
            recorder.close()
        env.close()
        logger.plot_metrics() # Final plots
        logger.close()
        print(f"Logs, plots, and videos saved to {config.LOG_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run curiosity-driven agent simulations.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the configuration file (e.g., configs.single_agent_config)"
    )
    args = parser.parse_args()

    # Dynamically import the specified config module
    try:
        config_module = importlib.import_module(args.config)
    except ImportError:
        raise ImportError(f"Could not import config file: {args.config}. Make sure the path is correct.")

    run(config_module)