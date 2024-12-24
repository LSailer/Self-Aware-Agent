import pybullet as p
import pybullet_data
import random
import numpy as np
import cv2

from video_recorder import VideoRecorder

class Environment:
    def __init__(self):
        """Initialize the PyBullet environment."""

        self.action_map = {
            "forward": [50.0, 0, 0, 0],
            "backward": [-50.0, 0, 0, 0],
            "left": [0, -50.0, 0, 0],
            "right": [0, 50.0, 0, 0],
            "rotate_left": [0, 0, 0, 5.0],
            "rotate_right": [0, 0, 0, -5.0],
        }

        # Connect to the PyBullet simulator in GUI mode
        p.connect(p.GUI)
        # Add search path for default PyBullet assets
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        # Load a flat plane
        self.plane_id = p.loadURDF("plane.urdf")
        
        # Initialize agent as a green sphere
        self.agent_id = p.createMultiBody(
            baseMass=1.0,
            baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_SPHERE, radius=0.2),
            baseVisualShapeIndex=p.createVisualShape(p.GEOM_SPHERE, radius=0.2, rgbaColor=[0, 1, 0, 1]),
            basePosition=[0, 0, 0.2]  # Start the agent on the ground
        )
        
        # Initialize a red cube as the target (larger size)
        self.cube_id = p.createMultiBody(
            baseMass=0.1,  # Set small mass to allow pushing
            baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.4, 0.4, 0.4]),
            baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[0.4, 0.4, 0.4], rgbaColor=[1, 0, 0, 1]),
            basePosition=[0.5, 0, 0.4]  # Start the cube directly in front of the agent
        )

        # Set gravity in the environment
        p.setGravity(0, 0, -9.8)

        # Define room dimensions and create walls
        self._create_room()

        # Reset environment state
        self.agent_pos = np.array([0.0, 0.0, 0.2])
        self.agent_orientation = [0, 0, 0, 1]  # Quaternion representing orientation
            # Reduce friction for smoother motion
        p.changeDynamics(self.plane_id, -1, lateralFriction=0.1)
        # Reduce mass for the agent
        p.changeDynamics(self.agent_id, -1, mass=0.5)

    def _create_room(self):
        """Create a 10x10 room with walls."""
        wall_thickness = 0.2
        wall_height = 2.0
        wall_length = 10

        # Define positions and sizes of walls
        walls = [
            {"pos": [0, wall_length / 2, wall_height / 2], "size": [wall_length / 2, wall_thickness / 2, wall_height / 2]},
            {"pos": [0, -wall_length / 2, wall_height / 2], "size": [wall_length / 2, wall_thickness / 2, wall_height / 2]},
            {"pos": [-wall_length / 2, 0, wall_height / 2], "size": [wall_thickness / 2, wall_length / 2, wall_height / 2]},
            {"pos": [wall_length / 2, 0, wall_height / 2], "size": [wall_thickness / 2, wall_length / 2, wall_height / 2]},
        ]
        self.wall_ids = []
        for wall in walls:
            # Create each wall using a box shape
            wall_id = p.createMultiBody(
                baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=wall["size"]),
                baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=wall["size"]),
                basePosition=wall["pos"],
                baseMass=0
            )
            self.wall_ids.append(wall_id)

    def reset(self):
        """Reset the environment to its initial state."""
        # Reset the agent's position
        self.agent_pos = np.array([0.0, 0.0, 0.2])
        self.agent_orientation = [0, 0, 0, 1]
        p.resetBasePositionAndOrientation(self.agent_id, [0, 0, 0.2], self.agent_orientation)
        
        # Place the cube directly in front of the agent
        p.resetBasePositionAndOrientation(self.cube_id, [0.5, 0, 0.4], [0, 0, 0, 1])

    def get_camera_image(self):
        """Capture the current camera view from the agent's perspective."""
        #TODO     # Use `agent_pos` dynamically for the camera's eye and target positions.
    # Ensure matrix updates when the agent moves.

        # Attach the camera to the agent
        camera_offset = [0.0, 0.0, 0.3]  # Camera is slightly above the agent's center
        agent_pos, agent_ori = p.getBasePositionAndOrientation(self.agent_id)

        # Compute forward vector based on agent orientation
        forward_vector = p.getMatrixFromQuaternion(agent_ori)[:3]  # Extract forward vector
        forward_vector = np.array(forward_vector)

        camera_eye = np.array(agent_pos) + np.array(camera_offset)
        camera_target = camera_eye + forward_vector * 2  # Adjust distance to look straight ahead

        # Set the camera view matrix
        view_matrix =  p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=agent_pos,
            distance=1.0,
            yaw=0,  # Adjust yaw for lateral view
            pitch=-30,  # Tilt downward
            roll=0,
            upAxisIndex=2
        )

        # Increase FOV for a wider view
        projection_matrix = p.computeProjectionMatrixFOV(
            fov=90, aspect=1.0, nearVal=0.1, farVal=10.0
        )

        # Capture the camera image
        width, height, rgb_img, _, _ = p.getCameraImage(
            width=640, height=480, viewMatrix=view_matrix, projectionMatrix=projection_matrix
        )
        rgb_array = np.array(rgb_img, dtype=np.uint8).reshape(height, width, 4)
        rgb_image = rgb_array[:, :, :3]  # Strip alpha channel
        return rgb_image

    def get_state(self):
        """Retrieve the current state of the environment."""
        # Get agent's position, orientation, and velocity
        agent_pos, agent_ori = p.getBasePositionAndOrientation(self.agent_id)
        agent_vel, agent_ang_vel = p.getBaseVelocity(self.agent_id)

        # Get cube's position and orientation
        cube_pos, cube_ori = p.getBasePositionAndOrientation(self.cube_id)

        # Compile the state information
        state = {
            "agent": {
                "position": agent_pos,
                "orientation": agent_ori,
                "velocity": agent_vel,
                "angular_velocity": agent_ang_vel,
            },
            "cube": {
                "position": cube_pos,
                "orientation": cube_ori,
            }
        }
        return state

    def apply_action(self, action):
        """Apply an action to the agent."""

        if action in self.action_map:
            force = self.action_map[action]
            print(f"Action: {action}, Force: {force}")

            if force[0] or force[1]:
                agent_pos, agent_ori = p.getBasePositionAndOrientation(self.agent_id)
                forward_vector = p.getMatrixFromQuaternion(agent_ori)[:3]
                force_vector = np.array(forward_vector) * force[0]
                print(f"Agent Position: {agent_pos}, Orientation: {agent_ori}")
                print(f"Force Vector: {force_vector}")

                p.applyExternalForce(
                    objectUniqueId=self.agent_id,
                    linkIndex=-1,
                    forceObj=force_vector.tolist(),
                    posObj=agent_pos,
                    flags=p.WORLD_FRAME
                )

            if force[3]:
                print(f"Applying Torque: {force[3]}")
                p.applyExternalTorque(
                    objectUniqueId=self.agent_id,
                    linkIndex=-1,
                    torqueObj=[0, 0, force[3]],
                    flags=p.WORLD_FRAME
                )
    def step_simulation(self):
        """Step the simulation forward."""
        p.stepSimulation()
        # Debug: Check the velocity of the agent after the simulation step
        linear_velocity, angular_velocity = p.getBaseVelocity(self.agent_id)
        print(f"Linear Velocity: {linear_velocity}, Angular Velocity: {angular_velocity}")

    def close(self):
        """Disconnect the simulation."""
        p.disconnect()

import cv2

# Example integration with Environment class
if __name__ == "__main__":
    env = Environment()
    env.reset()

    video_filename = "camera_feed.mp4"
    recorder = VideoRecorder(filename=video_filename)

    try:
        for _ in range(200):  # Run simulation for 200 steps
            env.apply_action("forward")
            env.step_simulation()
            
            # Get the camera image
            camera_image = env.get_camera_image()

            # Save the frame to the video
            recorder.write_frame(camera_image)

    finally:
        # Release resources
        recorder.close()
        env.close()

