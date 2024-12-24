import cv2
import torch
from environment import Environment
from curiosity_driven_agent import CuriosityDrivenAgent
from video_recorder import VideoRecorder
from metric_logger import MetricLogger

MAX_STEPS = 100

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


def run_simulation():
    # Initialize the environment
    env = Environment()
    env.reset()

    # Initialize the agent
    agent = CuriosityDrivenAgent(env.action_map)

    # Initialize the video recorder
    video_recorder = VideoRecorder(filename="camera_feed.mp4")

    logger = MetricLogger()

    for step in range(MAX_STEPS):
        # Get the current camera image
        raw_camera_image = env.get_camera_image()


        print(f"Raw image shape: {raw_camera_image.shape}")  # Debug: Check the shape

        # Preprocess the camera image and store it in the agent
        processed_image = agent.preprocess_camera_image(raw_camera_image)  # Shape: (1, 3, 64, 64)
        print(f"Processed image shape: {processed_image.shape}")  # Debug: Check the shape
        agent.last_processed_image = processed_image

        # Choose an action
        action_key, action_array = agent.choose_action(epsilon=0.1)  # Choose action
        action_tensor = torch.tensor(action_array, dtype=torch.float32).unsqueeze(0) 

        # Apply the action in the environment
        env.apply_action(action_key)
        env.step_simulation()

        # Get the next camera image
        next_raw_image = env.get_camera_image()
        next_processed_image = agent.preprocess_camera_image(next_raw_image)

        # Train the world model
        world_loss = agent.train_world_model(agent.last_processed_image, action_tensor, next_processed_image)

        # Predict the next image using the world model
        predicted_next_image = agent.world_model(agent.last_processed_image, action_tensor)

        # Calculate the curiosity reward
        curiosity_reward = agent.calculate_curiosity_reward(predicted_next_image, next_processed_image)

        # Train the self-model
        target_tensor = torch.tensor([[curiosity_reward]], dtype=torch.float32)  # Shape: (batch_size, 1)
        self_loss = agent.train_self_model(next_processed_image, action_tensor, target_tensor)
        print(f"Step: {step}")
        print(f"Self loss: {self_loss}")
        print(f"World loss: {world_loss}")
        print(f"Curiosity reward: {curiosity_reward}")

        # FIXME: Fix this bug to log the metrics and save the video
        # # Save the frame to the video
        # annotated_frame = video_recorder.annotate_frame(raw_camera_image, step, env.get_state()["agent"]["position"], curiosity_reward, self_loss)
        # video_recorder.write_frame(annotated_frame)

        # log_simulation_metrics(
        #         logger=logger,
        #         step=step,
        #         env=env,
        #         action_type=action_key,
        #         action_vector=action_vector,
        #         curiosity_reward=curiosity_reward,
        #         world_loss=world_loss,
        #         self_loss=self_loss
        #     )
        

    # Release resources
    video_recorder.close()
    env.close()
    print("Simulation finished and video saved.")

if __name__ == "__main__":
    run_simulation()