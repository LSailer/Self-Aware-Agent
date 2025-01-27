import pybullet as p
import pybullet_data
import torch
<<<<<<< Updated upstream
import numpy as np
from models import WorldModel, SelfModel
from environment import setup_environment, randomize_positions, setup_objects
import matplotlib.pyplot as plt
=======
from environment import Environment
from curiosity_driven_agent import CuriosityDrivenAgent
from video_recorder import VideoRecorder
from metric_logger import MetricLogger

MAX_STEPS = 3000

def log_simulation_metrics(logger, step, env, action_type, action_vector, curiosity_reward, world_loss, self_loss):
    state = env.get_state()
    agent_position = state["agent"]["position"]
    agent_velocity = state["agent"]["velocity"]
    agent_orientation = state["agent"]["orientation"]
    agent_angular_velocity = state["agent"]["angular_velocity"]

    logger.log_metrics(
        step=step,
        position=agent_position,
        velocity=agent_velocity,
        orientation=agent_orientation,
        angular_velocity=agent_angular_velocity,
        action_type=action_type,
        action_vector=action_vector,
        curiosity_reward=curiosity_reward,
        world_loss=world_loss,
        self_loss=self_loss
    )
>>>>>>> Stashed changes


def run_simulation():
    """
    Runs the simulation with adjusted Newtonian physics for better agent movement.
    """
    # Setup environment
    plane_id, wall_ids = setup_environment()
    ball_start_pos, cube_start_pos = randomize_positions()
    sphere_id, cube_id = setup_objects(ball_start_pos, cube_start_pos)

    # Adjust dynamics for improved Newtonian physics
    p.changeDynamics(sphere_id, -1, mass=1.0, lateralFriction=0.3, restitution=0.0, linearDamping=0.0, angularDamping=0.0)
    p.changeDynamics(cube_id, -1, mass=1.0, lateralFriction=0.3, restitution=0.0, linearDamping=0.0, angularDamping=0.0)

    # Initialize models and optimizers
    state_dim = 6
    hidden_dim = 64
    action_dim = 3
    world_model = WorldModel(state_dim + action_dim, hidden_dim, state_dim)
    self_model = SelfModel(state_dim + action_dim, hidden_dim, 1)
    world_optimizer = torch.optim.Adam(world_model.parameters(), lr=0.001)
    self_optimizer = torch.optim.Adam(self_model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()

    # Define actions
    actions = {
        "forward": [1, 0.0, 0.0],
        "backward": [-1, 0.0, 0.0],
        "left": [0.0, -1, 0.0],
        "right": [0.0, 1, 0.0],
        "rotate_left": [0.0, 0.0, -0.5],
        "rotate_right": [0.0, 0.0, 0.5],
    }

    # Simulation loop
    world_losses = []
    self_losses = []

    for step in range(240):
        print(f"Step {step}: Starting simulation step.")

        # Get current state
        cube_pos, _ = p.getBasePositionAndOrientation(cube_id)
        sphere_pos, _ = p.getBasePositionAndOrientation(sphere_id)
        state = torch.tensor(list(cube_pos) + list(sphere_pos), dtype=torch.float32).unsqueeze(0)

        # Evaluate candidate actions
        candidate_action_keys = list(actions.keys())
        future_losses = []

        for action_key in candidate_action_keys:
            action = torch.tensor(actions[action_key], dtype=torch.float32)
            input_to_self_model = torch.cat([state.squeeze(0), action], dim=-1).unsqueeze(0)
            predicted_loss = self_model(input_to_self_model)
            future_losses.append(predicted_loss.item())
            print(f"Step {step}: Action '{action_key}' predicted loss: {predicted_loss.item()}")

        # Select best action
        best_action_key = candidate_action_keys[np.argmax(future_losses)]
        best_action = actions[best_action_key]
        print(f"Step {step}: Selected action '{best_action_key}' with predicted loss {max(future_losses)}.")

        # Apply action to the sphere
        if "rotate" in best_action_key:
            p.resetBaseVelocity(sphere_id, angularVelocity=[0, 0, best_action[2]])
        else:
            force_scale = 10.0  # Scale forces for more significant movement
            best_action_scaled = [force_scale * val for val in best_action]
            p.applyExternalForce(sphere_id, -1, best_action_scaled, [0, 0, 0], p.WORLD_FRAME)

        # Apply force to cube
        cube_force = [-best_action[0], -best_action[1], 0.0]
        p.applyExternalForce(cube_id, -1, cube_force, [0, 0, 0], p.WORLD_FRAME)

        # Step simulation
        p.stepSimulation()

        # Compute world loss
        next_cube_pos, _ = p.getBasePositionAndOrientation(cube_id)
        next_sphere_pos, _ = p.getBasePositionAndOrientation(sphere_id)
        next_state = torch.tensor(list(next_cube_pos) + list(next_sphere_pos), dtype=torch.float32).unsqueeze(0)
        predicted_next_state = world_model(torch.cat([state.squeeze(0), torch.tensor(best_action, dtype=torch.float32)], dim=-1).unsqueeze(0))
        world_loss = criterion(predicted_next_state, next_state)
        print(f"Step {step}: World loss: {world_loss.item()}.")

        world_optimizer.zero_grad()
        world_loss.backward()
        world_optimizer.step()

        # Compute self loss
        predicted_loss = self_model(torch.cat([state.squeeze(0), torch.tensor(best_action, dtype=torch.float32)], dim=-1).unsqueeze(0))
        self_loss = criterion(predicted_loss, world_loss.detach())
        print(f"Step {step}: Self loss: {self_loss.item()}.")

        self_optimizer.zero_grad()
        self_loss.backward()
        self_optimizer.step()

        # Log losses
        world_losses.append(world_loss.item())
        self_losses.append(self_loss.item())

    # Plot losses
    plt.figure(figsize=(10, 5))
    plt.plot(world_losses, label="World Loss")
    plt.plot(self_losses, label="Self Loss")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    # Disconnect PyBullet
    p.disconnect()

if __name__ == "__main__":
    run_simulation()
