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


# Define the Self-Model
class SelfModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SelfModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.softmax(self.fc3(x), dim=-1)  # Predict probability distribution over loss bins


# Initialize PyBullet
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.loadURDF("plane.urdf")
p.setGravity(0, 0, -9.8)

# Create walls for the 10x10 room
wall_thickness = 0.1
wall_height = 1.0
wall_length = 10
walls = [
    {"pos": [0, 5, wall_height / 2], "size": [wall_length / 2, wall_thickness / 2, wall_height / 2]},
    {"pos": [0, -5, wall_height / 2], "size": [wall_length / 2, wall_thickness / 2, wall_height / 2]},
    {"pos": [-5, 0, wall_height / 2], "size": [wall_thickness / 2, wall_length / 2, wall_height / 2]},
    {"pos": [5, 0, wall_height / 2], "size": [wall_thickness / 2, wall_length / 2, wall_height / 2]},
]
for wall in walls:
    p.createMultiBody(
        baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=wall["size"]),
        baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=wall["size"]),
        basePosition=wall["pos"],
        baseMass=0
    )

# Add objects
cube_id = p.loadURDF("cube.urdf", [0, 0, 0.5])
sphere_id = p.loadURDF("sphere2.urdf", [1, 1, 0.5])

# Initialize Models
input_size = 6  # Example: [x, y, z, vx, vy, vz]
hidden_size = 64
output_size = 6
world_model = WorldModel(input_size, hidden_size, output_size)
self_model = SelfModel(input_size + 1, hidden_size, 10)  # Predict loss bins for future timesteps

world_optimizer = optim.Adam(world_model.parameters(), lr=0.001)
self_optimizer = optim.Adam(self_model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Simulation loop
for step in range(240):
    # Get current state
    cube_pos, cube_orn = p.getBasePositionAndOrientation(cube_id)
    cube_vel, cube_ang_vel = p.getBaseVelocity(cube_id)
    state = torch.tensor(cube_pos + cube_vel, dtype=torch.float32).unsqueeze(0)  # [x, y, z, vx, vy, vz]

    # Predict next state with world model
    predicted_next_state = world_model(state)

    # Calculate actual next state
    p.stepSimulation()
    next_cube_pos, _ = p.getBasePositionAndOrientation(cube_id)
    next_cube_vel, _ = p.getBaseVelocity(cube_id)
    actual_next_state = torch.tensor(next_cube_pos + next_cube_vel, dtype=torch.float32).unsqueeze(0)

    # Compute world-model loss
    loss = criterion(predicted_next_state, actual_next_state)
    world_optimizer.zero_grad()
    loss.backward()
    world_optimizer.step()

    # Use self-model to predict future loss
    proposed_action = torch.tensor([1.0], dtype=torch.float32).unsqueeze(0)  # Example action (adjust as needed)
    self_input = torch.cat([state, proposed_action], dim=-1)
    predicted_loss_bins = self_model(self_input)

    # Choose action based on predicted loss
    best_action_index = torch.argmax(predicted_loss_bins, dim=-1).item()
    chosen_action = best_action_index * 0.1  # Example mapping from bins to actions

    # Apply chosen action
    p.resetBasePositionAndOrientation(sphere_id, [chosen_action, chosen_action, 0.5], [0, 0, 0, 1])

    
    # Train self-model
    num_bins = 10
    scaled_loss = int(loss.item() * num_bins)  # Scale loss to bin range
    clamped_loss = max(0, min(num_bins - 1, scaled_loss))  # Clamp value to [0, num_bins-1]
    loss_bin = torch.tensor([clamped_loss], dtype=torch.long)  # Convert to tensor

    # Compute self-model loss
    self_loss = nn.CrossEntropyLoss()(predicted_loss_bins, loss_bin)
    self_optimizer.zero_grad()
    self_loss.backward()
    self_optimizer.step()


    print(f"Step {step}, World Loss: {loss.item()}, Self Loss: {self_loss.item()}")

# Close PyBullet
p.disconnect()
