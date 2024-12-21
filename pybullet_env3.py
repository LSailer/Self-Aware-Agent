import pybullet as p
import pybullet_data
import time

# Initialize PyBullet simulation
p.connect(p.GUI)  # Ensure GUI mode is active
p.setAdditionalSearchPath(pybullet_data.getDataPath())  # Load plane and other assets

# Set up the environment
p.loadURDF("plane.urdf")  # Add ground plane
p.setGravity(0, 0, -9.8)

# Create walls to form a 10x10 room
wall_thickness = 0.1  # Thickness of the walls
wall_height = 1.0  # Height of the walls
wall_length = 10  # Length of the walls

# Add walls (4 walls for the room)
# Wall 1: Front
p.createCollisionShape(p.GEOM_BOX, halfExtents=[wall_length / 2, wall_thickness / 2, wall_height / 2])
p.createVisualShape(p.GEOM_BOX, halfExtents=[wall_length / 2, wall_thickness / 2, wall_height / 2])
p.createMultiBody(baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[wall_length / 2, wall_thickness / 2, wall_height / 2]),
                  baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[wall_length / 2, wall_thickness / 2, wall_height / 2]),
                  basePosition=[0, 5, wall_height / 2],
                  baseMass=0)  # Static wall

# Wall 2: Back
p.createMultiBody(baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[wall_length / 2, wall_thickness / 2, wall_height / 2]),
                  baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[wall_length / 2, wall_thickness / 2, wall_height / 2]),
                  basePosition=[0, -5, wall_height / 2],
                  baseMass=0)

# Wall 3: Left
p.createMultiBody(baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[wall_thickness / 2, wall_length / 2, wall_height / 2]),
                  baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[wall_thickness / 2, wall_length / 2, wall_height / 2]),
                  basePosition=[-5, 0, wall_height / 2],
                  baseMass=0)

# Wall 4: Right
p.createMultiBody(baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[wall_thickness / 2, wall_length / 2, wall_height / 2]),
                  baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[wall_thickness / 2, wall_length / 2, wall_height / 2]),
                  basePosition=[5, 0, wall_height / 2],
                  baseMass=0)

# Add a cube (object) in the center of the room
cube_start_pos = [0, 0, 0.5]
cube_start_orientation = p.getQuaternionFromEuler([0, 0, 0])
cube_id = p.loadURDF("cube.urdf", cube_start_pos, cube_start_orientation)

# Add a sphere (agent) at a starting position
sphere_start_pos = [3, 3, 0.5]  # Placed within the 10x10 room
sphere_start_orientation = p.getQuaternionFromEuler([0, 0, 0])
sphere_id = p.loadURDF("sphere2.urdf", sphere_start_pos, sphere_start_orientation)

# Simulate and move the sphere
for step in range(240):  # Simulate for 240 steps (4 seconds at 60 Hz)
    if step == 100:  # Move sphere at step 100
        p.resetBasePositionAndOrientation(sphere_id, [1, 1, 0.5], sphere_start_orientation)
    p.stepSimulation()
    time.sleep(1/240)  # Real-time simulation

# Keep the GUI open for observation
print("Simulation complete. Press Ctrl+C to exit.")
while True:
    time.sleep(1)
