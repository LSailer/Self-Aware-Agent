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

    wall_ids = []
    for wall in walls:
        wall_id = p.createMultiBody(
            baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=wall["size"]),
            baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=wall["size"]),
            basePosition=wall["pos"],
            baseMass=0  # Static walls
        )
        wall_ids.append(wall_id)

    return plane_id, wall_ids

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
