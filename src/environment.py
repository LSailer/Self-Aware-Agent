import pybullet as p
import pybullet_data
import random
import numpy as np

class Environment:
    def __init__(self):
        """Initialize the PyBullet environment."""
        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.plane_id = p.loadURDF("plane.urdf")
        # Initialize agent
        self.agent_id = p.createMultiBody(
            baseMass=1.0,
            baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_SPHERE, radius=0.2),
            baseVisualShapeIndex=p.createVisualShape(p.GEOM_SPHERE, radius=0.2, rgbaColor=[0, 1, 0, 1]),
            basePosition=[0, 0, 0.5]
        )
        # Initialize cube (target)
        self.cube_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.1]),
            baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.1], rgbaColor=[1, 0, 0, 1]),
            basePosition=[1, 1, 0.5])
        
        p.setGravity(0, 0, -9.8)


        # Define room dimensions
        self._create_room()

        # Reset environment state
        self.agent_pos = np.array([0.0, 0.0, 0.5])
        self.reset()

    def _create_room(self):
        """Create a 10x10 room with walls."""
        wall_thickness = 0.2
        wall_height = 2.0
        wall_length = 10

        walls = [
            {"pos": [0, wall_length / 2, wall_height / 2], "size": [wall_length / 2, wall_thickness / 2, wall_height / 2]},
            {"pos": [0, -wall_length / 2, wall_height / 2], "size": [wall_length / 2, wall_thickness / 2, wall_height / 2]},
            {"pos": [-wall_length / 2, 0, wall_height / 2], "size": [wall_thickness / 2, wall_length / 2, wall_height / 2]},
            {"pos": [wall_length / 2, 0, wall_height / 2], "size": [wall_thickness / 2, wall_length / 2, wall_height / 2]},
        ]
        self.wall_ids = []
        for wall in walls:
            wall_id = p.createMultiBody(
                baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=wall["size"]),
                baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=wall["size"]),
                basePosition=wall["pos"],
                baseMass=0
            )
            self.wall_ids.append(wall_id)

    def reset(self):
        """Reset the environment to its initial state."""
        self.agent_pos = np.array([0.0, 0.0, 0.5])
        p.resetBasePositionAndOrientation(self.cube_id, [1, 1, 0.5], [0, 0, 0, 1])
        p.resetBasePositionAndOrientation(self.plane_id, [0, 0, 0], [0, 0, 0, 1])

    def get_state(self):
        """Retrieve the current state of the environment."""
        agent_pos, agent_ori = p.getBasePositionAndOrientation(self.agent_id)
        agent_vel, agent_ang_vel = p.getBaseVelocity(self.agent_id)

        cube_pos, cube_ori = p.getBasePositionAndOrientation(self.cube_id)

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
        force_scale = 10.0
        action_map = {
            "forward": [force_scale, 0, 0],
            "backward": [-force_scale, 0, 0],
            "left": [0, -force_scale, 0],
            "right": [0, force_scale, 0]
        }
        if action in action_map:
            force = action_map[action]
            p.applyExternalForce(
                objectUniqueId=self.agent_id,
                linkIndex=-1,
                forceObj=force,
                posObj=p.getBasePositionAndOrientation(self.agent_id)[0],
                flags=p.WORLD_FRAME
            )
    
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
        state = env.get_state()
        print(f"Agent Position: {state['agent']['position']}, Cube Position: {state['cube']['position']}")

    env.close()

# def randomize_positions():
#     """
#     Generates random initial positions for the ball and cube within the room.
#     Ensures they do not spawn too far from each other.
#     Returns:
#         ball_start_pos (list): Position of the ball.
#         cube_start_pos (list): Position of the cube.
#     """
#     ball_start_pos = [random.uniform(-4.5, 4.5), random.uniform(-4.5, 4.5), 0.5]
#     cube_start_pos = [random.uniform(-4.5, 4.5), random.uniform(-4.5, 4.5), 0.5]

#     while np.linalg.norm(np.array(ball_start_pos[:2]) - np.array(cube_start_pos[:2])) > 2.0:
#         cube_start_pos = [random.uniform(-4.5, 4.5), random.uniform(-4.5, 4.5), 0.5]

#     return ball_start_pos, cube_start_pos

# def setup_objects(ball_start_pos, cube_start_pos):
#     """
#     Creates the ball and cube objects in the PyBullet environment.
#     Args:
#         ball_start_pos (list): Initial position of the ball.
#         cube_start_pos (list): Initial position of the cube.
#     Returns:
#         sphere_id (int): ID of the ball.
#         cube_id (int): ID of the cube.
#     """
#     # Ball setup
#     sphere_collision = p.createCollisionShape(p.GEOM_SPHERE, radius=0.25)
#     sphere_visual = p.createVisualShape(p.GEOM_SPHERE, radius=0.25)
#     sphere_id = p.createMultiBody(baseMass=1.0, baseCollisionShapeIndex=sphere_collision, baseVisualShapeIndex=sphere_visual, basePosition=ball_start_pos)
#     p.changeDynamics(sphere_id, -1, mass=1.0, lateralFriction=0.8, restitution=0.0, linearDamping=0.1, angularDamping=0.1)

#     # Cube setup
#     cube_id = p.loadURDF("cube.urdf", cube_start_pos)
#     p.changeDynamics(cube_id, -1, mass=1.0, lateralFriction=0.8, restitution=0.0)

#     return sphere_id, cube_id
