import os
import torch
import pybullet as p
import pybullet_data
import numpy as np
import math


class Environment:
    def __init__(self, use_gui=False, sky_color=(10, 130, 200), wall_color=(0, 0, 0, 0)):
        """Initialize the PyBullet environment for two agents."""
        self.use_gui = use_gui
        
        # -- Configure start positions --
        # Agent 1 (Green)
        self.agent1_start_pos = [-1.0, 0.0, 0.2]
        self.agent1_start_ori = p.getQuaternionFromEuler([0, 0, 0])  # Yaw = 0 degrees

        # Agent 2 (Purple) - Different start position and orientation
        self.agent2_start_pos = [1.0, 0.0, 0.2]
        self.agent2_start_ori = p.getQuaternionFromEuler([0, 0, math.pi])  # Yaw = 180 degrees

        # Objects
        self.cylinder_start_pos = [2.4, 0.2, 0.5]
        self.cylinder_start_ori = [0, 0, 0, 1]
        self.disk_start_pos = [0.0, 2.0, 0]
        self.disk_start_ori = [0, 0, 0, 1]
        self.pyramid_start_pos = [-2.0, -0.5, 0.0]
        self.pyramid_start_ori = [0, 0, 0, 1]
        self.sphere_start_pos = [0.5, -2.0, 0.3]
        self.sphere_start_ori = [0, 0, 0, 1]

        # Action Map (identical for both agents)
        self.action_map = {
            'forward':      [50.0,  0,    0, 0],
            'backward':     [-50.0, 0,    0, 0],
            'left':         [0,    -50.0, 0, 0],
            'right':        [0,     50.0, 0, 0],
            'rotate_left':  [0,     0,    0, 5.0],
            'rotate_right': [0,     0,    0, -5.0],
            'stop':         [0,     0,    0, 0],
        }

        # Initialize PyBullet
        if self.use_gui:
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)
            
        # Configure debug visualizer
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
        
        # Set physics parameters
        p.setTimeStep(1.0/60.0)
        p.setRealTimeSimulation(0)
        p.setPhysicsEngineParameter(enableConeFriction=0)
        p.setPhysicsEngineParameter(numSolverIterations=10)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Set sky and wall colors
        self.sky_color = np.array(sky_color[:3], dtype=np.uint8)
        self.sky_color_norm = (self.sky_color.astype(np.float32) / 255.0).tolist()
        if wall_color is None:
            wall_color = (*sky_color[:3], 255)
        wc_rgb, wc_a = wall_color[:3], wall_color[3]
        self.wall_color = np.array(wc_rgb, dtype=np.uint8)
        self.wall_alpha = wc_a / 255.0
        self.wall_color_norm = (self.wall_color.astype(np.float32) / 255.0).tolist()

        # Create environment objects
        self._create_environment_objects()
        
        # Set up dynamics
        self._setup_dynamics()

    def _create_environment_objects(self):
        """Create all environment objects (plane, agents, objects, walls)."""
        # Load plane
        self.plane_id = p.loadURDF("plane.urdf")
        p.changeVisualShape(
            objectUniqueId=self.plane_id,
            linkIndex=-1,
            rgbaColor=[0.5, 0.5, 0.5, 1.0],
            specularColor=[0, 0, 0]
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
        self.cylinder_id = self._create_cylinder()
        self.disk_id = self._create_disk()
        self.pyramid_id = self._create_pyramid()
        self.sphere_id = self._create_sphere()

        # Create room
        self._create_room()

    def _create_agent(self, position, orientation, color):
        """Create an agent with specified position, orientation, and color."""
        half = 0.2
        return p.createMultiBody(
            baseMass=6,
            baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[half]*3),
            baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[half]*3, rgbaColor=color),
            basePosition=position,
            baseOrientation=orientation
        )

    def _create_cylinder(self):
        """Create the cylinder object."""
        return p.createMultiBody(
            baseMass=0.01,
            baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_CYLINDER, radius=0.7, height=0.1),
            baseVisualShapeIndex=p.createVisualShape(p.GEOM_CYLINDER, radius=0.7, length=0.1,
                                                     rgbaColor=[1, 0, 0, 1]),
            basePosition=self.cylinder_start_pos,
            baseOrientation=self.cylinder_start_ori
        )

    def _create_disk(self):
        """Create the disk object."""
        return p.createMultiBody(
            baseMass=0.01,
            baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_CYLINDER, radius=0.2, height=1.0),
            baseVisualShapeIndex=p.createVisualShape(p.GEOM_CYLINDER, radius=0.2, length=1.0,
                                                     rgbaColor=[0, 0, 1, 1]),
            basePosition=self.disk_start_pos,
            baseOrientation=self.disk_start_ori
        )

    def _create_pyramid(self):
        """Create the pyramid object."""
        base_dir = os.path.dirname(os.path.realpath(__file__))
        urdf_path = os.path.join(base_dir, 'pyramid.urdf')
        return p.loadURDF(urdf_path,
                         basePosition=self.pyramid_start_pos,
                         baseOrientation=self.pyramid_start_ori)

    def _create_sphere(self):
        """Create the sphere object."""
        sphere_radius = 0.3
        sphere_mass = 0.01
        sphere_collision = p.createCollisionShape(
            p.GEOM_SPHERE,
            radius=sphere_radius
        )
        sphere_visual = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=sphere_radius,
            rgbaColor=[1.0, 1.0, 0.0, 1.0]  # Yellow
        )
        return p.createMultiBody(
            baseMass=sphere_mass,
            baseCollisionShapeIndex=sphere_collision,
            baseVisualShapeIndex=sphere_visual,
            basePosition=self.sphere_start_pos,
            baseOrientation=self.sphere_start_ori
        )

    def _setup_dynamics(self):
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

        # Object dynamics
        for obj_id in [self.cylinder_id, self.disk_id, self.sphere_id]:
            p.changeDynamics(
                obj_id, -1,
                lateralFriction=0.5,
                restitution=0.1,
                angularDamping=0.5,
                linearDamping=0.5
            )

    def _create_room(self):
        """Create a room with walls."""
        wall_thickness = 0.2
        wall_height = 1.0
        wall_length = 7.5
        walls = [
            {'pos': [0,  wall_length/2, wall_height/2], 'size': [wall_length/2, wall_thickness/2, wall_height/2]},
            {'pos': [0, -wall_length/2, wall_height/2], 'size': [wall_length/2, wall_thickness/2, wall_height/2]},
            {'pos': [-wall_length/2, 0, wall_height/2],  'size': [wall_thickness/2, wall_length/2, wall_height/2]},
            {'pos': [ wall_length/2, 0, wall_height/2], 'size': [wall_thickness/2, wall_length/2, wall_height/2]},
        ]
        self.wall_ids = []
        base_dir = os.path.dirname(os.path.realpath(__file__))
        steinwand_path = os.path.join(base_dir, 'steinwand.jpg')
        texture_id = p.loadTexture(steinwand_path)
        
        for w in walls:
            cid = p.createCollisionShape(p.GEOM_BOX, halfExtents=w['size'])
            vid = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=w['size'],
                flags=p.VISUAL_SHAPE_DOUBLE_SIDED
            )
            wid = p.createMultiBody(
                baseCollisionShapeIndex=cid,
                baseVisualShapeIndex=vid,
                basePosition=w['pos'],
                baseMass=0
            )
            p.changeVisualShape(wid, -1, textureUniqueId=texture_id)
            self.wall_ids.append(wid)

    def reset(self):
        """Reset the environment to its initial state."""
        # Reset agents
        for agent_id, start_pos, start_ori in [
            (self.agent_id_1, self.agent1_start_pos, self.agent1_start_ori),
            (self.agent_id_2, self.agent2_start_pos, self.agent2_start_ori)
        ]:
            p.resetBasePositionAndOrientation(agent_id, start_pos, start_ori)
            p.resetBaseVelocity(agent_id, linearVelocity=[0, 0, 0], angularVelocity=[0, 0, 0])

        # Reset objects
        for obj_id, start_pos, start_ori in [
            (self.cylinder_id, self.cylinder_start_pos, self.cylinder_start_ori),
            (self.disk_id, self.disk_start_pos, self.disk_start_ori),
            (self.pyramid_id, self.pyramid_start_pos, self.pyramid_start_ori),
            (self.sphere_id, self.sphere_start_pos, self.sphere_start_ori)
        ]:
            p.resetBasePositionAndOrientation(obj_id, start_pos, start_ori)
            p.resetBaseVelocity(obj_id, linearVelocity=[0, 0, 0], angularVelocity=[0, 0, 0])

    def get_camera_image(self, agent_id):
        """Get camera image from agent's perspective."""
        if agent_id not in [self.agent_id_1, self.agent_id_2]:
            raise ValueError(f"Invalid agent_id: {agent_id}")

        agent_pos, agent_ori = p.getBasePositionAndOrientation(agent_id)
        euler = p.getEulerFromQuaternion(agent_ori)
        yaw = euler[2]
        forward = np.array([math.cos(yaw), math.sin(yaw), 0])
        eye = np.array(agent_pos) + np.array([0.0, 0.0, 0.2])
        target = eye + forward * 2.0

        view_matrix = p.computeViewMatrix(
            cameraEyePosition=eye.tolist(),
            cameraTargetPosition=target.tolist(),
            cameraUpVector=[0, 0, 1]
        )
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=90,
            aspect=640/480,
            nearVal=0.1,
            farVal=10.0
        )
        renderer = p.ER_BULLET_HARDWARE_OPENGL if self.use_gui else p.ER_TINY_RENDERER
        width, height, rgb_img, depth_buffer, seg_buffer = p.getCameraImage(
            width=640, height=480,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=renderer
        )

        rgba = np.array(rgb_img, dtype=np.uint8).reshape(height, width, 4)
        seg = np.array(seg_buffer, dtype=np.int32).reshape(height, width)
        rgb = rgba[:, :, :3]

        bg_mask = (seg < 0)
        rgb[bg_mask] = self.sky_color

        return rgb

    def get_state(self):
        """Get the current state of the environment."""
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
        for obj_id, obj_name in [
            (self.cylinder_id, "cylinder"),
            (self.disk_id, "disk"),
            (self.pyramid_id, "pyramid"),
            (self.sphere_id, "sphere")
        ]:
            pos, ori = p.getBasePositionAndOrientation(obj_id)
            state[obj_name] = {
                "position": pos,
                "orientation": ori
            }

        return state

    def apply_action(self, agent_id, action):
        """Apply an action to the specified agent."""
        if agent_id not in [self.agent_id_1, self.agent_id_2]:
            raise ValueError(f"Invalid agent_id: {agent_id}")

        if action not in self.action_map:
            return

        vx, vy, _, tz = self.action_map[action]
        pos, ori = p.getBasePositionAndOrientation(agent_id)
        mat = p.getMatrixFromQuaternion(ori)
        fwd = np.array(mat[:3])
        right = np.array(mat[3:6])

        if vx or vy:
            p.applyExternalForce(agent_id, -1, (fwd*vx + right*vy).tolist(), pos, p.WORLD_FRAME)
        if tz:
            p.resetBaseVelocity(agent_id, angularVelocity=[0, 0, tz])

    def clamp_velocity(self, agent_id, max_linear_velocity=2.0, max_angular_velocity=5.0):
        """Limit the linear and angular velocity of the specified agent."""
        if agent_id not in [self.agent_id_1, self.agent_id_2]:
            return

        lin_vel, ang_vel = p.getBaseVelocity(agent_id)
        lin_vel_np = np.array(lin_vel)
        ang_vel_np = np.array(ang_vel)

        lin_speed = np.linalg.norm(lin_vel_np)
        ang_speed = np.linalg.norm(ang_vel_np)

        new_lin_vel = lin_vel_np
        new_ang_vel = ang_vel_np

        if lin_speed > max_linear_velocity:
            new_lin_vel = (lin_vel_np / lin_speed) * max_linear_velocity

        if abs(ang_vel_np[2]) > max_angular_velocity:
            new_ang_vel[2] = math.copysign(max_angular_velocity, ang_vel_np[2])

        if lin_speed > max_linear_velocity or abs(ang_vel_np[2]) > max_angular_velocity:
            p.resetBaseVelocity(agent_id, linearVelocity=new_lin_vel.tolist(), angularVelocity=new_ang_vel.tolist())

    def step_simulation(self):
        """Step the simulation forward and apply velocity clamping."""
        p.stepSimulation()
        self.clamp_velocity(self.agent_id_1)
        self.clamp_velocity(self.agent_id_2)

    def close(self):
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
