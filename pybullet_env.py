import pybullet as p
import pybullet_data
import time

# Initialize PyBullet simulation
p.connect(p.GUI)  # Ensure GUI mode is active
p.setAdditionalSearchPath(pybullet_data.getDataPath())  # Load plane and other assets

# Set up the environment
p.loadURDF("plane.urdf")  # Add ground plane
p.setGravity(0, 0, -9.8)

# Add a cube (object)
cube_start_pos = [0, 0, 0.5]
cube_start_orientation = p.getQuaternionFromEuler([0, 0, 0])
cube_id = p.loadURDF("cube.urdf", cube_start_pos, cube_start_orientation)

# Add a sphere (agent)
sphere_start_pos = [1, 0, 0.5]
sphere_start_orientation = p.getQuaternionFromEuler([0, 0, 0])
sphere_id = p.loadURDF("sphere2.urdf", sphere_start_pos, sphere_start_orientation)

# Simulate and move the sphere
for step in range(240):  # Simulate for 240 steps (4 seconds at 60 Hz)
    if step == 100:  # Move sphere at step 100
        p.resetBasePositionAndOrientation(sphere_id, [0, 0, 0.5], sphere_start_orientation)
    p.stepSimulation()
    time.sleep(1/240)  # Real-time simulation

# Keep the GUI open for observation
print("Simulation complete. Press Ctrl+C to exit.")
while True:
    time.sleep(1)
