import os
import pybullet as p
import pybullet_data
import numpy as np
import math

class Environment:
    def __init__(self, use_gui=False, num_agents=2, sky_color=(10, 130, 200), wall_color=(0, 0, 0, 0)):
        """
        Initialize the PyBullet environment for a variable number of agents.

        Args:
            use_gui (bool): If True, run PyBullet with a GUI.
            num_agents (int): The number of agents to create in the environment.
            sky_color (tuple): RGB color for the sky.
            wall_color (tuple): RGBA color for the walls.
        """
        self.use_gui = use_gui
        self.num_agents = num_agents
        self.agent_ids = []
        self.wall_ids = []
        # -- Define agent start configurations (extensible for more agents) --
        if self.num_agents == 1:
            self.agent_start_configs = [
                {'pos': [0, 0.0, 0.2], 'ori': p.getQuaternionFromEuler([0, 0, 0]), 'color': [0, 1, 0, 1]},       # Agent 0 (Green)
            ]
        else:
            self.agent_start_configs = [
                {'pos': [-0.5, 0.0, 0.2], 'ori': p.getQuaternionFromEuler([0, 0, 0]), 'color': [0, 1, 0, 1]},       # Agent 0 (Green)
                {'pos': [0.5, 0.0, 0.2], 'ori': p.getQuaternionFromEuler([0, 0, math.pi]), 'color': [0.5, 0, 1, 1]}, # Agent 1 (Purple)
                # Add more agent configs here if needed
            ]
        if self.num_agents > len(self.agent_start_configs):
            raise ValueError(f"Requested {self.num_agents} agents, but only {len(self.agent_start_configs)} start configurations are defined.")

        # -- Define object start positions --
        self.cylinder_start_pos = [2.4, 0.2, 0.5]
        self.cylinder_start_ori = [0, 0, 0, 1]
        self.disk_start_pos = [0.0, 2.0, 0]
        self.disk_start_ori = [0, 0, 0, 1]
        self.pyramid_start_pos = [-2.0, -0.5, 0.0]
        self.pyramid_start_ori = [0, 0, 0, 1]
        self.sphere_start_pos = [0.5, -2.0, 0.3]
        self.sphere_start_ori = [0, 0, 0, 1]

        # -- Action Map (identical for all agents) --
        self.action_map = {
            'forward':      [50.0,  0,    0, 0],
            'backward':     [-50.0, 0,    0, 0],
            'left':         [0,    -50.0, 0, 0],
            'right':        [0,     50.0, 0, 0],
            'rotate_left':  [0,     0,    0, 5.0],
            'rotate_right': [0,     0,    0, -5.0],
            'stop':         [0,     0,    0, 0],
        }
        # -- Initialize PyBullet --
        if self.use_gui:
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)

        # -- Configure simulation settings --
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
        p.setTimeStep(1.0/60.0)

        p.setRealTimeSimulation(0)
        p.setPhysicsEngineParameter(enableConeFriction=0)
        p.setPhysicsEngineParameter(numSolverIterations=10)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # -- Set up environment colors --
        self.sky_color = np.array(sky_color[:3], dtype=np.uint8)

        # -- Create all objects and agents --
        self._create_environment_objects()
        #self._setup_dynamics()

    def _create_environment_objects(self):
        """Create all environment objects (plane, agents, objects, walls)."""
        self.plane_id = p.loadURDF("plane.urdf")
        p.changeVisualShape(objectUniqueId=self.plane_id, linkIndex=-1, rgbaColor=[0.5, 0.5, 0.5, 1.0])

        # Create agents in a loop based on num_agents
        for i in range(self.num_agents):
            config = self.agent_start_configs[i]
            agent_id = self._create_agent(
                position=config['pos'],
                orientation=config['ori'],
                color=config['color']
            )
            self.agent_ids.append(agent_id)

        # Create objects
        self.cylinder_id = self._create_cylinder()
        self.disk_id = self._create_disk()
        self.pyramid_id = self._create_pyramid()
        self.sphere_id = self._create_sphere()
        self._create_room()

    def _create_agent(self, position, orientation, color):
        """Helper function to create a single agent."""
        half_extents = [0.2] * 3
        return p.createMultiBody(
            baseMass=6,
            baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents),
            baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=color),
            basePosition=position,
            baseOrientation=orientation
        )

    def _create_cylinder(self):
        return p.createMultiBody(
            baseMass=0.01,
            baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_CYLINDER, radius=0.2, height=1.0),
            baseVisualShapeIndex=p.createVisualShape(p.GEOM_CYLINDER, radius=0.2, length=1.0,
                                                     rgbaColor=[1, 0, 0, 1]),
            basePosition=self.cylinder_start_pos
        )
    def _create_disk(self):
        return p.createMultiBody(
            baseMass=0.01,
            baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_CYLINDER, radius=0.7, height=0.1),
            baseVisualShapeIndex=p.createVisualShape(p.GEOM_CYLINDER, radius=0.7, length=0.1,
                                                    rgbaColor=[0, 0, 1, 1]),
            basePosition=self.disk_start_pos
        )

    def _create_pyramid(self):
        urdf_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'pyramid.urdf')
        return p.loadURDF(urdf_path, basePosition=self.pyramid_start_pos, baseOrientation=self.pyramid_start_ori)

    def _create_sphere(self):
        sphere_radius = 0.3
        sphere_mass   = 0.01
        sphere_collision = p.createCollisionShape(
            p.GEOM_SPHERE,
            radius=sphere_radius
        )
        sphere_visual = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=sphere_radius,
            rgbaColor=[1.0, 1.0, 0.0, 1.0]  # Gelb (R=1,G=1,B=0,A=1)
        )
        return p.createMultiBody(
            baseMass=sphere_mass,
            baseCollisionShapeIndex=sphere_collision,
            baseVisualShapeIndex=sphere_visual,
            basePosition=self.sphere_start_pos,
            baseOrientation=self.sphere_start_ori
        )
    def _create_room(self):
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

            # rgbaColor brauchst du nicht, wenn du eine Textur nutzt
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

            # Wechsle die Textur nach dem Erzeugen
            p.changeVisualShape(wid, -1, textureUniqueId=texture_id)

            self.wall_ids.append(wid)

    def _setup_dynamics(self):
        """Set up physics dynamics for all objects."""
        p.setGravity(0, 0, -9.8)
        # for agent_id in self.agent_ids: Without the physics the agent are moving
        #         p.changeDynamics(
        #             agent_id, -1,
        #             lateralFriction=0.9,
        #             restitution=0.4,
        #             angularDamping=0.5,
        #             linearDamping=0.0  
        #         )

        # Movable object dynamics: Make them bouncy and have some friction
        movable_objects = [self.cylinder_id, self.disk_id, self.pyramid_id, self.sphere_id]
        for obj_id in movable_objects:
            p.changeDynamics(
                obj_id, -1,
                lateralFriction=0.3,      # Standard friction
                restitution=0.6           # Make objects quite bouncy
            )

        # Wall dynamics: Give the walls some bounciness too
        for wall_id in self.wall_ids:
            p.changeDynamics(
                wall_id, -1,
                lateralFriction=1.0,      # High friction for walls
                restitution=0.5           # Walls have bounce
            )

    def reset(self):
        """Reset the environment to its initial state for all agents and objects."""
        # Reset agents
        for i, agent_id in enumerate(self.agent_ids):
            config = self.agent_start_configs[i]
            p.resetBasePositionAndOrientation(agent_id, config['pos'], config['ori'])
            p.resetBaseVelocity(agent_id, linearVelocity=[0, 0, 0], angularVelocity=[0, 0, 0])

        # Reset objects
        for obj_id, start_pos, start_ori in [(self.cylinder_id, self.cylinder_start_pos, self.cylinder_start_ori), (self.disk_id, self.disk_start_pos, self.disk_start_ori), (self.pyramid_id, self.pyramid_start_pos, self.pyramid_start_ori), (self.sphere_id, self.sphere_start_pos, self.sphere_start_ori)]:
            p.resetBasePositionAndOrientation(obj_id, start_pos, start_ori)
            p.resetBaseVelocity(obj_id, linearVelocity=[0, 0, 0], angularVelocity=[0, 0, 0])

    def get_camera_image(self, agent_id):
        agent_pos, agent_ori = p.getBasePositionAndOrientation(agent_id)
        euler = p.getEulerFromQuaternion(agent_ori)
        yaw = euler[2]
        forward = np.array([math.cos(yaw), math.sin(yaw), 0])
        eye    = np.array(agent_pos) + np.array([0.0, 0.0, 0.2])
        target = eye + forward * 2.0

        view_matrix = p.computeViewMatrix(cameraEyePosition=eye.tolist(),
                                          cameraTargetPosition=target.tolist(),
                                          cameraUpVector=[0, 0, 1])
        proj_matrix = p.computeProjectionMatrixFOV(fov=90,
                                                   aspect=640/480,
                                                   nearVal=0.1,
                                                   farVal=10.0)
        renderer = p.ER_BULLET_HARDWARE_OPENGL if self.use_gui else p.ER_TINY_RENDERER
        width, height, rgb_img, depth_buffer, seg_buffer = p.getCameraImage(
            width=640, height=480,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=renderer
        )

        rgba = np.array(rgb_img,   dtype=np.uint8).reshape(height, width, 4)
        seg  = np.array(seg_buffer, dtype=np.int32).reshape(height, width)
        rgb  = rgba[:, :, :3]

        bg_mask = (seg < 0)
        rgb[bg_mask] = self.sky_color

        return rgb

    # def get_camera_image(self, agent_id):
    #     """Get camera image from a specific agent's perspective."""
    #     if agent_id not in self.agent_ids:
    #         raise ValueError(f"Invalid agent_id: {agent_id}. Valid IDs are {self.agent_ids}")

    #     pos, ori = p.getBasePositionAndOrientation(agent_id)
    #     yaw = p.getEulerFromQuaternion(ori)[2]
    #     forward = np.array([math.cos(yaw), math.sin(yaw), 0])
    #     eye = np.array(pos) + np.array([0.0, 0.0, 0.2])
    #     target = eye + forward * 2.0

    #     view_matrix = p.computeViewMatrix(cameraEyePosition=eye, cameraTargetPosition=target, cameraUpVector=[0, 0, 1])
    #     proj_matrix = p.computeProjectionMatrixFOV(fov=90, aspect=640/480, nearVal=0.1, farVal=10.0)
    #     renderer = p.ER_BULLET_HARDWARE_OPENGL if self.use_gui else p.ER_TINY_RENDERER

    #     _, _, rgb_img, _, seg_buffer = p.getCameraImage(width=640, height=480, viewMatrix=view_matrix, projectionMatrix=proj_matrix, renderer=renderer)
    #     rgb = np.array(rgb_img, dtype=np.uint8).reshape(480, 640, 4)[:, :, :3]
    #     seg = np.array(seg_buffer, dtype=np.int32).reshape(480, 640)
    #     rgb[seg < 0] = self.sky_color # Apply background color
    #     return rgb

    def get_state(self):
        """Get the current state of all agents and objects in the environment."""
        state = {'objects': {}}
        # Get states for all agents
        for i, agent_id in enumerate(self.agent_ids):
            pos, ori = p.getBasePositionAndOrientation(agent_id)
            vel, ang_vel = p.getBaseVelocity(agent_id)
            state[f'agent_{i}'] = {"position": pos, "orientation": ori, "velocity": vel, "angular_velocity": ang_vel}
        # Get states for objects
        for obj_id, obj_name in [(self.cylinder_id, "cylinder"), (self.disk_id, "disk"), (self.pyramid_id, "pyramid"), (self.sphere_id, "sphere")]:
            pos, ori = p.getBasePositionAndOrientation(obj_id)
            state['objects'][obj_name] = {"position": pos, "orientation": ori}
        return state

    def apply_action(self, agent_id, action):
        """Apply an action from the action_map to the specified agent using velocity control."""
        if action not in self.action_map:
            return
        vx, vy, _, tz = self.action_map[action]
        pos, ori = p.getBasePositionAndOrientation(agent_id)
        mat = p.getMatrixFromQuaternion(ori)
        fwd = np.array(mat[:3]); right = np.array(mat[3:6])
        if vx or vy:
            p.applyExternalForce(agent_id, -1, (fwd*vx + right*vy).tolist(), pos, p.WORLD_FRAME)
        if tz:
            p.resetBaseVelocity(agent_id, angularVelocity=[0, 0, tz])
    
    def clamp_velocity(self, agent_id, max_linear_velocity=2.0, max_angular_velocity=5.0):
        """Limit the linear and angular velocity of the specified agent."""
        if agent_id not in self.agent_ids: return
        lin_vel, ang_vel = p.getBaseVelocity(agent_id)
        lin_vel_norm = np.linalg.norm(lin_vel)
        if lin_vel_norm > max_linear_velocity:
            lin_vel = (np.array(lin_vel) / lin_vel_norm) * max_linear_velocity
            p.resetBaseVelocity(agent_id, linearVelocity=lin_vel.tolist(), angularVelocity=ang_vel)

    def step_simulation(self):
        """Step the simulation forward and apply velocity clamping to all agents."""
        p.stepSimulation()
        for agent_id in self.agent_ids:
            self.clamp_velocity(agent_id)

    def close(self):
        """Disconnect the simulation."""
        try:
            p.disconnect(physicsClientId=self.physics_client)
        except p.error as e:
            print(f"Warning: Harmless error during PyBullet disconnect: {e}")
