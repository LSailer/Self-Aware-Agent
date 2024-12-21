import pybullet as p
import pybullet_data
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

# Define a force to push the cube
force_direction = [0.5, 0, 0]  # Push along +x direction
p.applyExternalForce(cube_id, -1, force_direction, [0, 0, 0], p.WORLD_FRAME)


# Initialize Models
input_size = 6
hidden_size = 64
output_size = 6

# Buffers for historical states and actions
buffer_size = 3
state = torch.zeros(1, 6, dtype=torch.float32)
proposed_action = torch.zeros(1, 1, dtype=torch.float32)
state_buffer = [state.clone() for _ in range(buffer_size)]
action_buffer = [proposed_action.clone() for _ in range(buffer_size)]
self_input_size = (state.size(-1) + proposed_action.size(-1)) * (buffer_size + 1)

# Models and optimizers
world_model = WorldModel(input_size, hidden_size, output_size)
self_model = SelfModel(self_input_size, hidden_size, 10)
world_optimizer = optim.Adam(world_model.parameters(), lr=0.001)
self_optimizer = optim.Adam(self_model.parameters(), lr=0.001)
criterion = nn.MSELoss()

visited_states = set()

# Initialize loss tracking
world_losses = []
self_losses = []

# Simulation loop
for step in range(240):
    # Get current state
    cube_pos, cube_orn = p.getBasePositionAndOrientation(cube_id)
    cube_vel, cube_ang_vel = p.getBaseVelocity(cube_id)
    state = torch.tensor(cube_pos + cube_vel, dtype=torch.float32).unsqueeze(0)

    # Predict next state
    predicted_next_state = world_model(state)
    p.stepSimulation()
    next_cube_pos, _ = p.getBasePositionAndOrientation(cube_id)
    next_cube_vel, _ = p.getBaseVelocity(cube_id)
    actual_next_state = torch.tensor(next_cube_pos + next_cube_vel, dtype=torch.float32).unsqueeze(0)

    # Train world model
    loss = criterion(predicted_next_state, actual_next_state)
    world_optimizer.zero_grad()
    loss.backward()
    world_optimizer.step()

    # Update buffers
    state_buffer.append(state)
    action_buffer.append(torch.tensor([[1.0]], dtype=torch.float32))
    if len(state_buffer) > buffer_size:
        state_buffer.pop(0)
        action_buffer.pop(0)

    # Combine historical data
    historical_states = torch.cat(state_buffer, dim=-1)
    historical_actions = torch.cat(action_buffer, dim=-1)
    self_input = torch.cat([state, proposed_action, historical_states, historical_actions], dim=-1)

    # Train self-model
    predicted_loss_bins = self_model(self_input)
    novelty_reward = 1.0 if tuple(state.squeeze(0).tolist()) not in visited_states else 0.0
    visited_states.add(tuple(state.squeeze(0).tolist()))
    num_bins = 10
    scaled_loss = int(loss.item() * num_bins)
    clamped_loss = max(0, min(num_bins - 1, scaled_loss))
    loss_bin = torch.tensor([clamped_loss], dtype=torch.long)
    alpha = 0.1
    self_loss = nn.CrossEntropyLoss()(predicted_loss_bins, loss_bin) - alpha * novelty_reward
    self_optimizer.zero_grad()
    self_loss.backward()
    self_optimizer.step()

    # Select and apply action
    best_action_index = torch.argmax(predicted_loss_bins, dim=-1).item()
    chosen_action = best_action_index * 0.1
    p.resetBasePositionAndOrientation(sphere_id, [chosen_action, chosen_action, 0.5], [0, 0, 0, 1])

    print(f"Step {step}, World Loss: {loss.item()}, Self Loss: {self_loss.item()}")

    # Append losses to track them
    world_losses.append(loss.item())
    self_losses.append(self_loss.item())

# Plot the losses
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(world_losses, label="World Loss")
plt.plot(self_losses, label="Self Loss")
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.title("World Loss and Self Loss Over Time")
plt.legend()
plt.show()

# Close PyBullet
p.disconnect()
