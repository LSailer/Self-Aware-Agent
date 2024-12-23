import pybullet as p
import torch
from environment import Environment
from CuriosityDrivenAgent import CuriosityDrivenAgent
import matplotlib.pyplot as plt
import numpy as np

MAX_STEPS = 100
ACTION_SET = {"forward": [1, 0, 0], "backward": [-1, 0, 0], "left": [0, -1, 0], "right": [0, 1, 0]}

def initialize_camera(env):
    agent_pos, _ = p.getBasePositionAndOrientation(env.agent_id)
    view_matrix = p.computeViewMatrix(
        cameraEyePosition=[agent_pos[0], agent_pos[1] - 2, agent_pos[2] + 2],
        cameraTargetPosition=[agent_pos[0], agent_pos[1], agent_pos[2]],
        cameraUpVector=[0, 0, 1]
    )
    projection_matrix = p.computeProjectionMatrixFOV(
        fov=60, aspect=1.0, nearVal=0.1, farVal=10.0
    )
    return view_matrix, projection_matrix

def update_camera(env):
    agent_pos, _ = p.getBasePositionAndOrientation(env.agent_id)
    view_matrix = p.computeViewMatrix(
        cameraEyePosition=[agent_pos[0], agent_pos[1] - 2, agent_pos[2] + 2],
        cameraTargetPosition=[agent_pos[0], agent_pos[1], agent_pos[2]],
        cameraUpVector=[0, 0, 1]
    )
    return view_matrix

def process_camera_data(rgb_img):
    rgb_array = np.array(rgb_img, dtype=np.uint8).reshape(480, 640, 4)
    rgb_image = rgb_array[:, :, :3]
    normalized_rgb = rgb_image / 255.0
    return torch.tensor(normalized_rgb, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)

def run_simulation():
    env = Environment()
    state_dim = 6
    action_dim = 3
    hidden_dim = 64
    agent = CuriosityDrivenAgent(state_dim, action_dim, hidden_dim, ACTION_SET)

    view_matrix, projection_matrix = initialize_camera(env)

    for step in range(MAX_STEPS):
        view_matrix = update_camera(env)
        width, height, rgb_img, _, _ = p.getCameraImage(
            width=640, height=480, viewMatrix=view_matrix, projectionMatrix=projection_matrix
        )
        camera_input = process_camera_data(rgb_img)

        state = env.get_state()
        agent_pos = torch.tensor(state["agent"]["position"], dtype=torch.float32)
        cube_pos = torch.tensor(state["cube"]["position"], dtype=torch.float32)
        full_state = torch.cat([agent_pos, cube_pos])

        action_key, action = agent.choose_action()
        env.apply_action(action_key)
        env.step_simulation()

        next_state = env.get_state()
        next_agent_pos = torch.tensor(next_state["agent"]["position"], dtype=torch.float32)
        next_cube_pos = torch.tensor(next_state["cube"]["position"], dtype=torch.float32)
        next_full_state = torch.cat([next_agent_pos, next_cube_pos])

        input_tensor = torch.cat([full_state, action], dim=-1).unsqueeze(0)
        next_state_tensor = next_full_state.unsqueeze(0)

        world_loss = agent.train_world_model(input_tensor, next_state_tensor)
        print(f"Step {step}, World model loss: {world_loss}")

        predicted_next_state = agent.world_model(input_tensor)
        curiosity_reward = agent.calculate_curiosity_reward(predicted_next_state, next_state_tensor)
        print(f"Step {step}, Curiosity reward: {curiosity_reward}")

        target_tensor = torch.tensor([curiosity_reward], dtype=torch.float32).unsqueeze(0)
        self_loss = agent.train_self_model(input_tensor, target_tensor)
        print(f"Step {step}, Self model loss: {self_loss}")

    env.close()

if __name__ == "__main__":
    run_simulation()
