import pybullet as p
import pybullet_data
import time
import torch
import torch.nn as nn
import torch.optim as optim

# Define the World Model
class WorldModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(WorldModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# Initialize PyBullet
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.loadURDF("plane.urdf")
p.setGravity(0, 0, -9.8)

# Create walls for the 10x10 room
wall_thickness = 0.1
wall_height = 1.0
wall_length = 10

# Wall positions
walls = [
    {"pos": [0, 5, wall_height / 2], "orientation": [0, 0, 0, 1], "size": [wall_length / 2, wall_thickness / 2, wall_height / 2]},
    {"pos": [0, -5, wall_height / 2], "orientation": [0, 0, 0, 1], "size": [wall_length / 2, wall_thickness / 2, wall_height / 2]},
    {"pos": [-5, 0, wall_height / 2], "orientation": [0, 0, 0, 1], "size": [wall_thickness / 2, wall_length / 2, wall_height / 2]},
    {"pos": [5, 0, wall_height / 2], "orientation": [0, 0, 0, 1], "size": [wall_thickness / 2, wall_length / 2, wall_height / 2]},
]

for wall in walls:
    p.createMultiBody(
        baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=wall["size"]),
        baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=wall["size"]),
        basePosition=wall["pos"],
        baseOrientation=wall["orientation"],
        baseMass=0  # Static wall
    )

# Add objects
cube_id = p.loadURDF("cube.urdf", [0, 0, 0.5])
sphere_id = p.loadURDF("sphere2.urdf", [1, 1, 0.5])

# Initialize World Model
input_size = 6  # Example: [x, y, z, vx, vy, vz]
hidden_size = 64
output_size = 6
world_model = WorldModel(input_size, hidden_size, output_size)
optimizer = optim.Adam(world_model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Simulation loop
for step in range(240):
    # Get current state
    cube_pos, cube_orn = p.getBasePositionAndOrientation(cube_id)
    cube_vel, cube_ang_vel = p.getBaseVelocity(cube_id)
    state = torch.tensor(cube_pos + cube_vel, dtype=torch.float32).unsqueeze(0)  # [x, y, z, vx, vy, vz]

    # Predict next state
    predicted_next_state = world_model(state)

    # Move the sphere for demonstration
    if step == 100:
        p.resetBasePositionAndOrientation(sphere_id, [0.5, 0.5, 0.5], [0, 0, 0, 1])

    # Step simulation
    p.stepSimulation()
    time.sleep(1/240)

    # Get actual next state from PyBullet
    next_cube_pos, _ = p.getBasePositionAndOrientation(cube_id)
    next_cube_vel, _ = p.getBaseVelocity(cube_id)
    actual_next_state = torch.tensor(next_cube_pos + next_cube_vel, dtype=torch.float32).unsqueeze(0)

    # Compute loss and train
    loss = criterion(predicted_next_state, actual_next_state)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Step {step}, Loss: {loss.item()}")

# Close PyBullet
p.disconnect()
