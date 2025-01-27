import pybullet as p
import pybullet_data
import random
import numpy as np

def setup_environment():
    """
    Sets up the PyBullet environment, including walls and plane.
    Returns the IDs of the plane and created walls.
    """
    # Initialize PyBullet
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    plane_id = p.loadURDF("plane.urdf")
    p.setGravity(0, 0, -9.8)

    # Define parameters for the walls
    wall_thickness = 0.2
    wall_height = 2.0
    wall_length = 10

    # Define wall positions and sizes
    walls = [
        {"pos": [0, wall_length / 2, wall_height / 2], "size": [wall_length / 2, wall_thickness / 2, wall_height / 2]},
        {"pos": [0, -wall_length / 2, wall_height / 2], "size": [wall_length / 2, wall_thickness / 2, wall_height / 2]},
        {"pos": [-wall_length / 2, 0, wall_height / 2], "size": [wall_thickness / 2, wall_length / 2, wall_height / 2]},
        {"pos": [wall_length / 2, 0, wall_height / 2], "size": [wall_thickness / 2, wall_length / 2, wall_height / 2]},
    ]

<<<<<<< Updated upstream
    wall_ids = []
    for wall in walls:
        wall_id = p.createMultiBody(
            baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=wall["size"]),
            baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=wall["size"]),
            basePosition=wall["pos"],
            baseMass=0  # Static walls
=======
        # Action Map
        self.action_map = {
            "forward":      [50.0,   0,   0,  0],
            "backward":     [-50.0,  0,   0,  0],
            "left":         [0, -50.0,   0,  0],
            "right":        [0,  50.0,   0,  0],
            "rotate_left":  [0,   0,   0,  5.0],
            "rotate_right": [0,   0,   0, -5.0],
            "stop":         [0,    0,   0,   0],  
        }
    
        # Connect to the PyBullet simulator in GUI mode: p.GUI else: p.DIRECT
        p.connect(p.GUI)
        p.setTimeStep(1./30.)
        p.setRealTimeSimulation(0)
        # Add search path for default PyBullet assets
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        print(pybullet_data.getDataPath())

        # Load a flat plane
        self.plane_id = p.loadURDF("plane.urdf")

        p.changeVisualShape(
            objectUniqueId=self.plane_id,
            linkIndex=-1,
            rgbaColor=[0.5, 0.5, 0.5, 1.0],  # z.B. hellgrau
>>>>>>> Stashed changes
        )
        wall_ids.append(wall_id)

<<<<<<< Updated upstream
    return plane_id, wall_ids
=======
        # Initialize agent as a green sphere
        self.agent_id = p.createMultiBody(
            baseMass=3,
            baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_SPHERE, radius=0.2),
            baseVisualShapeIndex=p.createVisualShape(p.GEOM_SPHERE, radius=0.2, rgbaColor=[0, 1, 0, 1]),
            basePosition=self.agent_start_pos  # <-- Agent spawn
        )
        
        # Initialize a red cube as the target
        self.cube_id = p.createMultiBody(
            baseMass=0.1,  # Set small mass to allow pushing
            baseCollisionShapeIndex=p.createCollisionShape(
                p.GEOM_BOX, halfExtents=[0.4, 0.4, 0.4]
            ),
            baseVisualShapeIndex=p.createVisualShape(
                p.GEOM_BOX, halfExtents=[0.4, 0.4, 0.4], rgbaColor=[1, 0, 0, 1]
            ),
            basePosition=self.cube_start_pos  # <-- Cube spawn
        )
>>>>>>> Stashed changes

def randomize_positions():
    """
    Generates random initial positions for the ball and cube within the room.
    Ensures they do not spawn too far from each other.
    Returns:
        ball_start_pos (list): Position of the ball.
        cube_start_pos (list): Position of the cube.
    """
    ball_start_pos = [random.uniform(-4.5, 4.5), random.uniform(-4.5, 4.5), 0.5]
    cube_start_pos = [random.uniform(-4.5, 4.5), random.uniform(-4.5, 4.5), 0.5]

    while np.linalg.norm(np.array(ball_start_pos[:2]) - np.array(cube_start_pos[:2])) > 2.0:
        cube_start_pos = [random.uniform(-4.5, 4.5), random.uniform(-4.5, 4.5), 0.5]

    return ball_start_pos, cube_start_pos

def setup_objects(ball_start_pos, cube_start_pos):
    """
    Creates the ball and cube objects in the PyBullet environment.
    Args:
        ball_start_pos (list): Initial position of the ball.
        cube_start_pos (list): Initial position of the cube.
    Returns:
        sphere_id (int): ID of the ball.
        cube_id (int): ID of the cube.
    """
    # Ball setup
    sphere_collision = p.createCollisionShape(p.GEOM_SPHERE, radius=0.25)
    sphere_visual = p.createVisualShape(p.GEOM_SPHERE, radius=0.25)
    sphere_id = p.createMultiBody(baseMass=1.0, baseCollisionShapeIndex=sphere_collision, baseVisualShapeIndex=sphere_visual, basePosition=ball_start_pos)
    p.changeDynamics(sphere_id, -1, mass=1.0, lateralFriction=0.8, restitution=0.0, linearDamping=0.1, angularDamping=0.1)

    # Cube setup
    cube_id = p.loadURDF("cube.urdf", cube_start_pos)
    p.changeDynamics(cube_id, -1, mass=1.0, lateralFriction=0.8, restitution=0.0)

    return sphere_id, cube_id
