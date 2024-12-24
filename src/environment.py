import pybullet as p
import pybullet_data
import random
import numpy as np
import cv2

class Environment:
    def __init__(self):
        """Initialize the PyBullet environment."""
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
            
        )
        
        # Initialize a red cube as the target (doubled size)
        self.cube_id = p.createMultiBody(
            baseMass=0.1,  # Set small mass to allow pushing
            baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.4, 0.4, 0.4]),
            baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[0.4, 0.4, 0.4], rgbaColor=[1, 0, 0, 1]),
             # Start the cube on the ground
        )

        # Set gravity in the environment
        p.setGravity(0, 0, -9.8)

        # Define room dimensions and create walls
        self._create_room()

        # Reset environment state
        self.agent_pos = np.array([0.0, 0.0, 0.2])
        self.reset()

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
        p.resetBasePositionAndOrientation(self.agent_id, [0, 0, 0.2], [0, 0, 0, 1])
        
        # Place the cube directly in front of the agent
        p.resetBasePositionAndOrientation(self.cube_id, [0.5, 0, 0.4], [0, 0, 0, 1])

    def get_camera_image(self):
        """Capture the current camera view from the agent's perspective."""
        camera_offset = [0.0, 0.0, 0.3]  # Camera slightly above the agent
        agent_pos, agent_ori = p.getBasePositionAndOrientation(self.agent_id)

        # Adjust forward vector to align with agent direction
        forward_vector = np.array([1, 0, 0])  # Fixed forward direction
        camera_eye = np.array(agent_pos) + np.array(camera_offset)
        camera_target = camera_eye + forward_vector * 2  # Adjust distance to look straight ahead

        # Set the camera view matrix
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=camera_eye.tolist(),
            cameraTargetPosition=camera_target.tolist(),
            cameraUpVector=[0, 0, 1]
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


    def apply_action(self, action):
        """Apply an action to the agent."""
        force_scale = 10.0  # Scale of the applied force
        # Define possible actions and corresponding forces
        action_map = {
            "forward": [force_scale, 0, 0],
            "backward": [-force_scale, 0, 0],
            "left": [0, -force_scale, 0],
            "right": [0, force_scale, 0]
        }
        if action in action_map:
            velocity = action_map[action]
            p.resetBaseVelocity(objectUniqueId=self.agent_id, linearVelocity=velocity)

    def step_simulation(self):
        """Step the simulation forward."""
        p.stepSimulation()

    def close(self):
        """Disconnect the simulation."""
        p.disconnect()

if __name__ == "__main__":
    env = Environment()
    env.reset()

    for _ in range(100):
        env.apply_action("forward")
        env.step_simulation()
        camera_image = env.get_camera_image()
        cv2.imshow("Agent Camera View", camera_image)
        cv2.waitKey(10)

    env.close()
    cv2.destroyAllWindows()
