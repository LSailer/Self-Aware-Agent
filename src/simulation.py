import torch
import numpy as np
import pybullet as p
from environment import Environment
from curiosity_driven_agent import ActionSelection, CuriosityDrivenAgent
from video_recorder import VideoRecorder
from metric_logger import MetricLogger

MAX_STEPS = 1000
BATCH_SIZE = 16
EPSILON_GREEDY = 0.3
UPDATE_EVERY_N_STEPS = 4
VISUALIZE_VAE_AFTER_STEPS = 100
VISUALIZE_RNN_AFTER_STEPS = 100
LOG_DIR = "logs/SingleAgent_V1"
ACTION_SELECTION = ActionSelection.EPSILON_GREEDY
TEMPERATURE = 1.0
C = 1.0
USE_GUI = True

def check_interaction(env):
    """
    Gibt True zurück, sobald der Agent in der Simulation Kontakt
    mit Zylinder, Scheibe oder Pyramide hat.
    """
    # Prüfe echte Kontaktpunkte in PyBullet
    for obj_id in (env.cylinder_id, env.disk_id, env.pyramid_id, env.sphere_id):
        contacts = p.getContactPoints(bodyA=env.agent_id, bodyB=obj_id)
        if contacts:
            return True
    return False

def run_simulation():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    env = Environment(use_gui=USE_GUI)
    env.reset()

    agent = CuriosityDrivenAgent(
        actions=env.action_map,
        latent_dim=64,
        rnn_hidden_dim=256,
        buffer_size=50000,
        batch_size=BATCH_SIZE,
        device=device
    )

    recorder = VideoRecorder('camera_feed.mp4')
    logger   = MetricLogger(log_dir=LOG_DIR)

    agent.reset_hidden_state()
    raw               = env.get_camera_image()
    agent.current_z   = agent.encode_image(raw)

    print("Starting simulation...")
    for step in range(MAX_STEPS):
        action_key, action_array = agent.choose_action(
            action_selection=ACTION_SELECTION,
            epsilon=EPSILON_GREEDY,
            temperature=TEMPERATURE,
            c=C
        )

        env.apply_action(action_key)
        env.step_simulation()

        next_raw = env.get_camera_image()
        agent.store_experience(raw, action_key, action_array, 0.0, next_raw, False)

        # encode & update hidden state
        next_z = agent.encode_image(next_raw)
        with torch.no_grad():
            _, next_h = agent.rnn_model(
                agent.current_z,
                torch.tensor(action_array).float().unsqueeze(0).to(device),
                agent.current_h
            )
        agent.current_z, agent.current_h, raw = next_z, next_h, next_raw

        loss_dict = None
        if step > BATCH_SIZE and step % UPDATE_EVERY_N_STEPS == 0:
            loss_dict = agent.update_models(
                visualize=step % 1000 == 0,
                step=step,
                log_dir=LOG_DIR,
                num_agents=1,
                visualize_vae_after_steps=VISUALIZE_VAE_AFTER_STEPS,
                visualize_rnn_after_steps=VISUALIZE_RNN_AFTER_STEPS
            )

        if loss_dict:
            interacting = check_interaction(env)
            logger.log_metrics(
                step, 0, action_key, action_array,
                curiosity_reward=loss_dict.get('avg_curiosity_reward', 0),
                world_loss=loss_dict.get('rnn_loss', 0),
                self_loss=loss_dict.get('self_loss', 0),
                vae_loss=loss_dict.get('vae_loss', 0),
                vae_kld_loss=loss_dict.get('vae_kld_loss', 0),
                is_interacting_object=interacting
            )
            print(f"Step {step}: Action {action_key}, interacting={interacting}")
            # Video-Annotation
            frame = recorder.annotate_frame(
                next_raw, step,
                loss_dict.get('avg_curiosity_reward', 0),
                0
            )
            recorder.write_frame(frame)
        elif step % 100 == 0:
            print(f"Step {step}: Action {action_key}")


    print("Simulation finished.")
    recorder.close()
    env.close()
    logger.plot_metrics()

if __name__ == '__main__':
    run_simulation()
