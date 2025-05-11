from typing import Dict, List
import pybullet as p
import pybullet_data
import numpy as np
import math


class Environment:
    def __init__(self, use_gui=False):
        """Initialize the PyBullet environment for two agents."""
        # -- Configure start positions --
        # Agent 1 (Green)
        self.agent1_start_pos = [-1.0, 0.0, 0.2]
        self.agent1_start_ori = p.getQuaternionFromEuler([0, 0, 0])  # Yaw = 0 degrees

        # Agent 2 (Purple) - Different start position and orientation
        self.agent2_start_pos = [1.0, 0.0, 0.2]
        self.agent2_start_ori = p.getQuaternionFromEuler([0, 0, math.pi])  # Yaw = 180 degrees

        # Objects
        self.cube_start_pos = [0.0, 1.5, 0.4]
        self.cube_start_ori = p.getQuaternionFromEuler([0, 0, 0])
        self.cylinder_start_pos = [0.0, -1.5, 0.5]
        self.cylinder_start_ori = p.getQuaternionFromEuler([0, 0, 0])

        # Action Map (identical for both agents)
        # Format: [vx, vy, dummy, torque_z] (relative to agent)
        self.action_map = {
            "forward":      [50.0,   0,   0,  0],
            "backward":     [-50.0,  0,   0,  0],
            "left":         [0, -50.0,   0,  0],
            "right":        [0,  50.0,   0,  0],
            "rotate_left":  [0,   0,   0,  5.0],
            "rotate_right": [0,   0,   0, -5.0],
            "stop":         [0,    0,   0,   0],  
        }

        self.use_gui = use_gui

        # Initialize PyBullet
        self._initialize_pybullet()

        # Create environment objects
        self._create_environment_objects()

        # Set up dynamics
        self._setup_dynamics()
        
        

    def _initialize_pybullet(self) -> None:
        """Initialize PyBullet connection and basic settings."""
        try:
            if self.use_gui:
                self.physics_client = p.connect(p.GUI)
                print("Successfully connected to PyBullet GUI.")
            else:
                self.physics_client = p.connect(p.DIRECT)
                print("Successfully connected to PyBullet DIRECT (no GUI).")
        except p.error as e:
            print(f"Could not connect to PyBullet GUI (might already be connected?): {e}")
            try:
                self.physics_client = p.connect(p.DIRECT)
                print("Connected to PyBullet DIRECT instead.")
            except p.error:
                print("PyBullet connection already exists or failed.")
                self.physics_client = 0

        p.setTimeStep(1./60.)
        p.setRealTimeSimulation(0)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        print(f"Using PyBullet data path: {pybullet_data.getDataPath()}")

    def _create_environment_objects(self) -> None:
        """Create all environment objects (plane, agents, objects, walls)."""
        # Load plane
        self.plane_id = p.loadURDF("plane.urdf")
        p.changeVisualShape(
            objectUniqueId=self.plane_id,
            linkIndex=-1,
            rgbaColor=[0.8, 0.8, 0.8, 1.0]  # Light gray
        )

        # Create agents
        self.agent_id_1 = self._create_agent(
            position=self.agent1_start_pos,
            orientation=self.agent1_start_ori,
            color=[0, 1, 0, 1]  # Green
        )
        self.agent_id_2 = self._create_agent(
            position=self.agent2_start_pos,
            orientation=self.agent2_start_ori,
            color=[0.5, 0, 1, 1]  # Purple
        )

        # Create objects
        self.cube_id = self._create_cube()
        self.cylinder_id = self._create_cylinder()

        # Create room
        self._create_room()

    def _create_agent(self, position: List[float], orientation: List[float], color: List[float]) -> int:
        """Create an agent with specified position, orientation, and color."""
        return p.createMultiBody(
            baseMass=3,
            baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_SPHERE, radius=0.2),
            baseVisualShapeIndex=p.createVisualShape(p.GEOM_SPHERE, radius=0.2, rgbaColor=color),
            basePosition=position,
            baseOrientation=orientation
        )

    def _create_cube(self) -> int:
        """Create the cube object."""
        return p.createMultiBody(
            baseMass=6,
            baseCollisionShapeIndex=p.createCollisionShape(
                p.GEOM_BOX, halfExtents=[0.4, 0.4, 0.4]
            ),
            baseVisualShapeIndex=p.createVisualShape(
                p.GEOM_BOX, halfExtents=[0.4, 0.4, 0.4], rgbaColor=[1, 0, 0, 1]  # Red
            ),
            basePosition=self.cube_start_pos,
            baseOrientation=self.cube_start_ori
        )

    def _create_cylinder(self) -> int:
        """Create the cylinder object."""
        return p.createMultiBody(
            baseMass=1,
            baseCollisionShapeIndex=p.createCollisionShape(
                p.GEOM_CYLINDER, radius=0.2, height=1.0
            ),
            baseVisualShapeIndex=p.createVisualShape(
                p.GEOM_CYLINDER, radius=0.2, length=1.0, rgbaColor=[0, 0, 1, 1]  # Blue
            ),
            basePosition=self.cylinder_start_pos,
            baseOrientation=self.cylinder_start_ori
        )

    def _setup_dynamics(self) -> None:
        """Set up physics dynamics for all objects."""
        # Set gravity
        p.setGravity(0, 0, -9.8)

        # Plane dynamics
        p.changeDynamics(
            self.plane_id, -1,
            lateralFriction=0.1,
            angularDamping=0.5,
            linearDamping=0.5
        )

        # Agent dynamics (same for both agents)
        for agent_id in [self.agent_id_1, self.agent_id_2]:
            p.changeDynamics(
                agent_id, -1,
                lateralFriction=1.0,
                rollingFriction=0.005,
                spinningFriction=0.005,
                restitution=0.0,
                angularDamping=0.5,
                linearDamping=0.5
            )

        # Cube dynamics
        p.changeDynamics(
            self.cube_id, -1,
            lateralFriction=0.5,
            restitution=0.1
        )

        # Cylinder dynamics
        p.changeDynamics(
            self.cylinder_id, -1,
            lateralFriction=0.5,
            restitution=0.1,
            angularDamping=0.5,
            linearDamping=0.5
        )

    def _create_room(self) -> None:
        """Create a 10x10 room with walls."""
        wall_thickness = 0.2
        wall_height = 2.0
        room_half_size = 5.0  # Half size of the room #TODO: wall_length

        # Define positions and sizes of walls relative to origin (0,0)
        walls = [
            # Wall at +y
            {"pos": [0, room_half_size, wall_height / 2],
             "size": [room_half_size, wall_thickness / 2, wall_height / 2]},
            # Wall at -y
            {"pos": [0, -room_half_size, wall_height / 2],
             "size": [room_half_size, wall_thickness / 2, wall_height / 2]},
            # Wall at -x
            {"pos": [-room_half_size, 0, wall_height / 2],
             "size": [wall_thickness / 2, room_half_size, wall_height / 2]},
            # Wall at +x
            {"pos": [room_half_size, 0, wall_height / 2],
             "size": [wall_thickness / 2, room_half_size, wall_height / 2]},
        ]

        self.wall_ids = []
        for wall in walls:
            wall_visual_shape = p.createVisualShape(
                shapeType=p.GEOM_BOX,
                halfExtents=wall["size"],
                rgbaColor=[255, 255,255, 1]  # black
            )
            wall_collision_shape = p.createCollisionShape(
                shapeType=p.GEOM_BOX,
                halfExtents=wall["size"]
            )
            wall_id = p.createMultiBody(
                baseCollisionShapeIndex=wall_collision_shape,
                baseVisualShapeIndex=wall_visual_shape,
                basePosition=wall["pos"],
                baseMass=0  # Static walls
            )
            self.wall_ids.append(wall_id)

    def reset(self) -> None:
        """Reset the environment to its initial state for both agents and objects."""
        # Reset agents
        for agent_id, start_pos, start_ori in [
            (self.agent_id_1, self.agent1_start_pos, self.agent1_start_ori),
            (self.agent_id_2, self.agent2_start_pos, self.agent2_start_ori)
        ]:
            p.resetBasePositionAndOrientation(agent_id, start_pos, start_ori)
            p.resetBaseVelocity(agent_id, linearVelocity=[0, 0, 0], angularVelocity=[0, 0, 0])

        # Reset objects
        for obj_id, start_pos, start_ori in [
            (self.cube_id, self.cube_start_pos, self.cube_start_ori),
            (self.cylinder_id, self.cylinder_start_pos, self.cylinder_start_ori)
        ]:
            p.resetBasePositionAndOrientation(obj_id, start_pos, start_ori)
            p.resetBaseVelocity(obj_id, linearVelocity=[0, 0, 0], angularVelocity=[0, 0, 0])

        print("Environment reset.")

    def get_camera_image(self, agent_id: int) -> np.ndarray:
        """
        Capture the current camera view from the perspective of the specified agent.

        Args:
            agent_id (int): The unique ID of the agent (self.agent_id_1 or self.agent_id_2).

        Returns:
            np.ndarray: RGB image array (height, width, 3).
        """
        if agent_id not in [self.agent_id_1, self.agent_id_2]:
            raise ValueError(f"Invalid agent_id: {agent_id}")

        # Camera offset: Position relative to agent
        camera_offset = [0.0, 0.0, 0.3]  # Camera slightly above agent center

        # Get agent position and orientation
        agent_pos, agent_ori = p.getBasePositionAndOrientation(agent_id)

        # Calculate forward direction based on current yaw rotation
        euler = p.getEulerFromQuaternion(agent_ori)
        yaw = euler[2]  # Yaw angle (rotation around Z-axis)

        # Define forward direction in world coordinates
        forward_dir = np.array([math.cos(yaw), math.sin(yaw), 0])

        # Camera position relative to agent
        camera_eye = np.array(agent_pos) + np.array(camera_offset)

        # Target point for camera to look at
        camera_target = camera_eye + forward_dir * 2.0  # Look 2 units ahead

        # Calculate view matrix
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=camera_eye.tolist(),
            cameraTargetPosition=camera_target.tolist(),
            cameraUpVector=[0, 0, 1]  # "Up" is Z-axis
        )

        # Define projection matrix
        projection_matrix = p.computeProjectionMatrixFOV(
            fov=90, aspect=1.0, nearVal=0.1, farVal=10.0
        )

        # Capture camera image
        width, height, rgb_img, _, _ = p.getCameraImage(
            width=640, height=480,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )

        # Convert to RGB array
        rgb_array = np.array(rgb_img, dtype=np.uint8).reshape(height, width, 4)
        rgb_image = rgb_array[:, :, :3]  # Remove alpha channel

        return rgb_image

    def get_state(self) -> Dict:
        """
        Retrieve the current state of the environment, including both agents and objects.

        Returns:
            dict: Dictionary containing the state information for agents and objects.
        """
        state = {}

        # Get states for both agents
        for agent_id, agent_name in [(self.agent_id_1, "agent_1"), (self.agent_id_2, "agent_2")]:
            pos, ori = p.getBasePositionAndOrientation(agent_id)
            vel, ang_vel = p.getBaseVelocity(agent_id)
            state[agent_name] = {
                "position": pos,
                "orientation": ori,
                "velocity": vel,
                "angular_velocity": ang_vel,
            }

        # Get states for objects
        for obj_id, obj_name in [(self.cube_id, "cube"), (self.cylinder_id, "cylinder")]:
            pos, ori = p.getBasePositionAndOrientation(obj_id)
            vel, ang_vel = p.getBaseVelocity(obj_id)
            state[obj_name] = {
                "position": pos,
                "orientation": ori,
                "velocity": vel,
                "angular_velocity": ang_vel,
            }

        return state

    def apply_action(self, agent_id: int, action_key: str) -> None:
        """
        Apply an action to the specified agent by applying forces/torques.

        Args:
            agent_id (int): The unique ID of the agent (self.agent_id_1 or self.agent_id_2).
            action_key (str): The key representing the action from self.action_map.
        """
        if agent_id not in [self.agent_id_1, self.agent_id_2]:
            raise ValueError(f"Invalid agent_id: {agent_id}")

        if action_key not in self.action_map:
            print(f"Warning: Unknown action '{action_key}' for agent {agent_id}")
            return

        force_params = self.action_map[action_key]  # [vx, vy, dummy, torque_z]
        vx, vy, _, torque_z = force_params

        agent_pos, agent_ori = p.getBasePositionAndOrientation(agent_id)

        if action_key == "stop":
            # Stop agent by resetting velocities
            p.resetBaseVelocity(
                agent_id,
                linearVelocity=[0, 0, 0],
                angularVelocity=[0, 0, 0]
            )
        else:
            # --- Movement (Translation) ---
            if vx != 0 or vy != 0:
                # Transform local movement (vx, vy) to world coordinates
                rotation_matrix = p.getMatrixFromQuaternion(agent_ori)
                forward_vec = np.array([rotation_matrix[0], rotation_matrix[3], rotation_matrix[6]])
                left_vec = np.array([rotation_matrix[1], rotation_matrix[4], rotation_matrix[7]])

                # Combined movement force in world coordinates
                move_force_world = forward_vec * vx + left_vec * vy

                # Apply force at center of mass
                p.applyExternalForce(
                    objectUniqueId=agent_id,
                    linkIndex=-1,
                    forceObj=move_force_world.tolist(),
                    posObj=agent_pos,
                    flags=p.WORLD_FRAME
                )

            # --- Rotation (around Z-axis) ---
            if torque_z != 0:
                # Apply torque around Z-axis in world coordinates
                p.applyExternalTorque(
                    objectUniqueId=agent_id,
                    linkIndex=-1,
                    torqueObj=[0, 0, torque_z],
                    flags=p.WORLD_FRAME
                )

    def clamp_velocity(self, agent_id: int, max_linear_velocity: float = 2.0, max_angular_velocity: float = 5.0) -> None:
        """
        Limit the linear and angular velocity of the specified agent.

        Args:
            agent_id (int): The unique ID of the agent.
            max_linear_velocity (float): Maximum allowed linear velocity.
            max_angular_velocity (float): Maximum allowed angular velocity.
        """
        if agent_id not in [self.agent_id_1, self.agent_id_2]:
            return

        lin_vel, ang_vel = p.getBaseVelocity(agent_id)
        lin_vel_np = np.array(lin_vel)
        ang_vel_np = np.array(ang_vel)

        lin_speed = np.linalg.norm(lin_vel_np)
        ang_speed = np.linalg.norm(ang_vel_np)

        new_lin_vel = lin_vel_np
        new_ang_vel = ang_vel_np

        # Limit linear velocity
        if lin_speed > max_linear_velocity:
            new_lin_vel = (lin_vel_np / lin_speed) * max_linear_velocity

        # Limit angular velocity (especially around Z)
        if abs(ang_vel_np[2]) > max_angular_velocity:
            new_ang_vel[2] = math.copysign(max_angular_velocity, ang_vel_np[2])

        # Apply limited velocities if changes were needed
        if lin_speed > max_linear_velocity or abs(ang_vel_np[2]) > max_angular_velocity:
            p.resetBaseVelocity(agent_id, linearVelocity=new_lin_vel.tolist(), angularVelocity=new_ang_vel.tolist())

    def step_simulation(self) -> None:
        """Step the simulation forward and apply velocity clamping."""
        p.stepSimulation()
        # Apply velocity clamping to both agents
        self.clamp_velocity(self.agent_id_1)
        self.clamp_velocity(self.agent_id_2)

    def close(self) -> None:
        """Disconnect the simulation."""
        try:
            p.disconnect(physicsClientId=self.physics_client)
            print("Successfully disconnected from PyBullet.")
        except p.error as e:
            print(f"Error disconnecting from PyBullet: {e}")

# Example usage (for testing)
if __name__ == "__main__":
    env = Environment()
    print("Environment created.")

    # Test reset
    env.reset()

    # Test get_state
    initial_state = env.get_state()
    print("\nInitial State:")
    print(f"  Agent 1 Pos: {initial_state['agent_1']['position']}")
    print(f"  Agent 2 Pos: {initial_state['agent_2']['position']}")

    # Test get_camera_image
    try:
        img1 = env.get_camera_image(env.agent_id_1)
        img2 = env.get_camera_image(env.agent_id_2)
        print(f"\nGot camera image for Agent 1, shape: {img1.shape}")
        print(f"Got camera image for Agent 2, shape: {img2.shape}")
    except Exception as e:
        print(f"Error getting camera image: {e}")

    # Test apply_action and step_simulation
    print("\nApplying actions...")
    for i in range(100):
        env.apply_action(env.agent_id_1, "forward")
        env.apply_action(env.agent_id_2, "rotate_left")
        env.step_simulation()
        if i % 20 == 0:
            state = env.get_state()
            print(f"  Step {i}: Agent 1 Pos: {state['agent_1']['position']}, Agent 2 Pos: {state['agent_2']['position']}")
            import time
            time.sleep(0.05)

    final_state = env.get_state()
    print("\nFinal State:")
    print(f"  Agent 1 Pos: {final_state['agent_1']['position']}")
    print(f"  Agent 2 Pos: {final_state['agent_2']['position']}")

    # Test close
    env.close()
    print("\nEnvironment closed.")
