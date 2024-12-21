import pybullet as p
import pybullet_data
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import random

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
        return self.fc3(x)  # Predict future world-model loss


# Initialize PyBullet
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.loadURDF("plane.urdf")
p.setGravity(0, 0, -9.8)  # Updated gravity as per original paper's repro

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

# Create the walls in the simulation
for wall in walls:
    p.createMultiBody(
        baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=wall["size"]),
        baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=wall["size"]),
        basePosition=wall["pos"],
        baseMass=0  # Static walls
    )

# Generate random positions for the ball and cube
ball_start_pos = [random.uniform(-4.5, 4.5), random.uniform(-4.5, 4.5), 0.5]
cube_start_pos = [random.uniform(-4.5, 4.5), random.uniform(-4.5, 4.5), 0.5]

# Ensure the ball and cube spawn close to each other
while np.linalg.norm(np.array(ball_start_pos[:2]) - np.array(cube_start_pos[:2])) > 2.0:
    cube_start_pos = [random.uniform(-4.5, 4.5), random.uniform(-4.5, 4.5), 0.5]

# Ball setup
sphere_collision = p.createCollisionShape(p.GEOM_SPHERE, radius=0.25)
sphere_visual = p.createVisualShape(p.GEOM_SPHERE, radius=0.25)
sphere_id = p.createMultiBody(baseMass=1.0, baseCollisionShapeIndex=sphere_collision, baseVisualShapeIndex=sphere_visual, basePosition=ball_start_pos)

# Adjust ball dynamics to obey Newtonian physics
p.changeDynamics(sphere_id, -1, mass=1.0, lateralFriction=0.8, restitution=0.0, linearDamping=0.1, angularDamping=0.1)

# Cube setup
cube_id = p.loadURDF("cube.urdf", cube_start_pos)
p.changeDynamics(cube_id, -1, mass=1.0, lateralFriction=0.8, restitution=0.0)

# Initialize models and optimizers
state_dim = 6
hidden_dim = 64
action_dim = 3
world_model = WorldModel(state_dim + action_dim, hidden_dim, state_dim)
self_model = SelfModel(state_dim + action_dim, hidden_dim, 1)
world_optimizer = optim.Adam(world_model.parameters(), lr=0.001)
self_optimizer = optim.Adam(self_model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Simulation loop
world_losses = []
self_losses = []

# Define actions based on the agent's view
actions = {
    "forward": [0.25, 0.0, 0.0],
    "backward": [-0.25, 0.0, 0.0],
    "left": [0.0, -0.25, 0.0],
    "right": [0.0, 0.25, 0.0],
    "rotate_left": [0.0, 0.0, -0.1],  # Rotational movement to the left
    "rotate_right": [0.0, 0.0, 0.1]   # Rotational movement to the right
}

for step in range(240):
    # Get current state
    cube_pos, _ = p.getBasePositionAndOrientation(cube_id)
    sphere_pos, sphere_orientation = p.getBasePositionAndOrientation(sphere_id)
    state = torch.tensor(list(cube_pos) + list(sphere_pos), dtype=torch.float32).unsqueeze(0)

    # Generate candidate actions
    candidate_action_keys = list(actions.keys())
    future_losses = []

    for action_key in candidate_action_keys:
        action = torch.tensor(actions[action_key], dtype=torch.float32)
        input_to_self_model = torch.cat([state.squeeze(0), action], dim=-1).unsqueeze(0)
        predicted_loss = self_model(input_to_self_model)
        future_losses.append(predicted_loss.item())

    # Select the action with the highest predicted loss
    best_action_key = candidate_action_keys[np.argmax(future_losses)]
    best_action = actions[best_action_key]

    # Apply action to move the agent or rotate
    if "rotate" in best_action_key:
        # Rotate the agent
        current_orientation = p.getBaseVelocity(sphere_id)[1]  # Get angular velocity
        p.resetBaseVelocity(sphere_id, angularVelocity=[0, 0, best_action[2]])
    else:
        # Move the agent
        p.applyExternalForce(sphere_id, -1, best_action, [0, 0, 0], p.WORLD_FRAME)

    # Apply force to move the cube
    cube_force = [-best_action[0], -best_action[1], 0.0]
    p.applyExternalForce(cube_id, -1, cube_force, [0, 0, 0], p.WORLD_FRAME)

    # Step simulation
    p.stepSimulation()

    # Compute losses
    next_cube_pos, _ = p.getBasePositionAndOrientation(cube_id)
    next_sphere_pos, _ = p.getBasePositionAndOrientation(sphere_id)
    next_state = torch.tensor(list(next_cube_pos) + list(next_sphere_pos), dtype=torch.float32).unsqueeze(0)
    predicted_next_state = world_model(torch.cat([state.squeeze(0), torch.tensor(best_action, dtype=torch.float32)], dim=-1).unsqueeze(0))
    world_loss = criterion(predicted_next_state, next_state)

    world_optimizer.zero_grad()
    world_loss.backward()
    world_optimizer.step()

    predicted_loss = self_model(torch.cat([state.squeeze(0), torch.tensor(best_action, dtype=torch.float32)], dim=-1).unsqueeze(0))
    self_loss = criterion(predicted_loss, world_loss.detach())
    self_optimizer.zero_grad()
    self_loss.backward()
    self_optimizer.step()

    world_losses.append(world_loss.item())
    self_losses.append(self_loss.item())

    print(f"Step {step}, World Loss: {world_loss.item()}, Self Loss: {self_loss.item()}, Best Action: {best_action_key}")

# Plot losses
plt.figure(figsize=(10, 5))
plt.plot(world_losses, label="World Loss")
plt.plot(self_losses, label="Self Loss")
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.legend()
plt.show()

p.disconnect()
