import os
import torch
import pybullet as p
import pybullet_data
import numpy as np
import math

class Environment:
    def __init__(self,use_gui):
        """Initialize the PyBullet environment."""
        # Start positions and orientations
        self.agent_start_pos    = [0.0, 0.0, 0.2]
        self.agent_start_ori    = [0, 0, 0, 1]
        self.cylinder_start_pos = [1.4, 0.2, 0.0]
        self.cylinder_start_ori = [0, 0, 0, 1]
        self.disk_start_pos     = [2.0, 2.0, 0.4]
        self.disk_start_ori     = [0, 0, 0, 1]
        self.pyramid_start_pos  = [-1.8, -1.0, 0.0]
        self.pyramid_start_ori  = [0, 0, 0, 1]

        # Action map
        self.action_map = {
            'forward':      [50.0,  0,    0, 0],
            'backward':     [-50.0, 0,    0, 0],
            'left':         [0,    -50.0, 0, 0],
            'right':        [0,     50.0, 0, 0],
            'rotate_left':  [0,     0,    0, 5.0],
            'rotate_right': [0,     0,    0, -5.0],
            'stop':         [0,     0,    0, 0],
        }

        # Connect and configure PyBullet
        if use_gui:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
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

        # Load a flat plane
        self.plane_id = p.loadURDF('plane.urdf')
        p.changeVisualShape(
            objectUniqueId=self.plane_id,
            linkIndex=-1,
            rgbaColor=[0.5, 0.5, 0.5, 1.0],  # z.B. hellgrau
            specularColor=[0, 0, 0],
            #textureUniqueId=-1       # Hier kann man das Schabrettmuster umändern
        )

        # Initialize agent as a green cube
        half = 0.2
        self.agent_id = p.createMultiBody(
            baseMass=6,
            baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[half]*3),
            baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[half]*3, rgbaColor=[0,1,0,1]),
            basePosition=self.agent_start_pos
        )

        # Initialize a red cylinder
        self.cylinder_id = p.createMultiBody(
            baseMass=1,
            baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_CYLINDER, radius=0.7, height=0.1),
            baseVisualShapeIndex=p.createVisualShape(p.GEOM_CYLINDER, radius=0.7, length=0.1, rgbaColor=[1,0,0,1]),
            basePosition=self.cylinder_start_pos
        )

        # Initialize a blue disk
        self.disk_id = p.createMultiBody(
            baseMass=1,
            baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_CYLINDER, radius=0.2, height=1.0),
            baseVisualShapeIndex=p.createVisualShape(p.GEOM_CYLINDER, radius=0.2, length=1.0, rgbaColor=[0,0,1,1]),
            basePosition=self.disk_start_pos
        )

        # Load a pyramid URDF
        base_dir = os.path.dirname(os.path.realpath(__file__))
        urdf_path = os.path.join(base_dir, 'pyramid.urdf')
        self.pyramid_id = p.loadURDF(urdf_path, basePosition=self.pyramid_start_pos, baseOrientation=self.pyramid_start_ori)

        # Set gravity and create room
        p.setGravity(0,0,-9.8)
        self._create_room()

    def _create_room(self):
        wall_thickness = 0.2
        wall_height = 1.0
        wall_length = 5
        walls = [
            {'pos': [0,  wall_length/2, wall_height/2], 'size':[wall_length/2, wall_thickness/2, wall_height/2]},
            {'pos': [0, -wall_length/2, wall_height/2], 'size':[wall_length/2, wall_thickness/2, wall_height/2]},
            {'pos': [-wall_length/2, 0, wall_height/2],'size':[wall_thickness/2, wall_length/2, wall_height/2]},
            {'pos': [ wall_length/2, 0, wall_height/2],'size':[wall_thickness/2, wall_length/2, wall_height/2]},
        ]
        self.wall_ids=[]
        for w in walls:
            wid = p.createMultiBody(
                baseCollisionShapeIndex=p.createCollisionShape(
                    p.GEOM_BOX,
                    halfExtents=w['size']
                ),
                baseVisualShapeIndex=p.createVisualShape(
                    p.GEOM_BOX,
                    halfExtents=w['size'],
                    rgbaColor=[0, 0, 0, 1],              # Alpha=0 → komplett transparent
                    flags=p.VISUAL_SHAPE_DOUBLE_SIDED    # Damit beide Seiten sichtbar bleiben
                ),
                basePosition=w['pos'],
                baseMass=0
            )
            self.wall_ids.append(wid)

    def reset(self):
        p.resetBasePositionAndOrientation(self.agent_id, self.agent_start_pos, self.agent_start_ori)
        p.resetBasePositionAndOrientation(self.cylinder_id, self.cylinder_start_pos, self.cylinder_start_ori)
        p.resetBasePositionAndOrientation(self.disk_id, self.disk_start_pos, self.disk_start_ori)
        p.resetBasePositionAndOrientation(self.pyramid_id, self.pyramid_start_pos, self.pyramid_start_ori)
        for oid in [self.agent_id, self.cylinder_id, self.disk_id, self.pyramid_id]:
            p.resetBaseVelocity(oid, [0,0,0], [0,0,0])

    def get_camera_image(self):
        """
        Capture the current camera view from the agent's perspective,
        replace the background (“sky”) with a custom color.
        Returns:
            np.ndarray: RGB image of shape (height, width, 3)
        """
        # 1) Get agent pose and forward direction
        agent_pos, agent_ori = p.getBasePositionAndOrientation(self.agent_id)
        euler = p.getEulerFromQuaternion(agent_ori)            # (roll, pitch, yaw)
        yaw = euler[2]
        forward = np.array([math.cos(yaw), math.sin(yaw), 0])

        # 2) Compute eye & target for ego-perspective
        camera_offset = [0.0, 0.0, 0.2]                       # 30 cm über Agent
        eye    = np.array(agent_pos) + np.array(camera_offset)
        target = eye + forward * 2.0                          # 2 Einheiten vorwärts

        # 3) Build view and projection matrices
        view_matrix = p.computeViewMatrix(
            cameraEyePosition   = eye.tolist(),
            cameraTargetPosition= target.tolist(),
            cameraUpVector      = [0, 0, 1]
        )
        proj_matrix = p.computeProjectionMatrixFOV(
            fov    = 90,
            aspect = 640/480,
            nearVal= 0.1,
            farVal = 10.0
        )

        # 4) Render RGB + depth
        width, height, rgb_img, depth_buffer, _ = p.getCameraImage(
            width           = 640,
            height          = 480,
            viewMatrix      = view_matrix,
            projectionMatrix= proj_matrix,
            renderer        = p.ER_BULLET_HARDWARE_OPENGL
        )

        # 5) Convert to NumPy arrays
        rgba   = np.array(rgb_img,   dtype=np.uint8).reshape(height, width, 4)
        depth  = np.array(depth_buffer, dtype=np.float32).reshape(height, width)
        rgb    = rgba[:, :, :3]       # drop alpha channel

        # 6) Create background mask (depth == far clip → no geometry)
        bg_mask = (depth >= 1.0)

        # 7) Define sky color (in RGB 0–255)
        sky_color = np.array([0, 100, 200], dtype=np.uint8)  # helles Himmelblau

        # 8) Apply sky color to background pixels
        rgb[bg_mask] = sky_color

        return rgb


    def get_state(self):
        agent_pos, agent_ori = p.getBasePositionAndOrientation(self.agent_id)
        agent_vel, agent_ang = p.getBaseVelocity(self.agent_id)
        cpos,_ = p.getBasePositionAndOrientation(self.cylinder_id)
        dpos,_ = p.getBasePositionAndOrientation(self.disk_id)
        ppos,_ = p.getBasePositionAndOrientation(self.pyramid_id)
        return {
            'agent':{'position':agent_pos,'orientation':agent_ori,'velocity':agent_vel,'angular_velocity':agent_ang},
            'cylinder':{'position':cpos},
            'disk':{'position':dpos},
            'pyramid':{'position':ppos}
        }

    def apply_action(self, action):
        if action not in self.action_map:
            return
        vx, vy, _, tz = self.action_map[action]
        pos, ori = p.getBasePositionAndOrientation(self.agent_id)
        mat = p.getMatrixFromQuaternion(ori)
        fwd = np.array(mat[:3]); right = np.array(mat[3:6])
        if vx or vy:
            p.applyExternalForce(self.agent_id, -1, (fwd*vx + right*vy).tolist(), pos, p.WORLD_FRAME)
        if tz:
            p.resetBaseVelocity(self.agent_id, angularVelocity=[0,0,tz])

    def clamp_angular_velocity(self, maxv=5.0):
        _,ang = p.getBaseVelocity(self.agent_id)
        speed = np.linalg.norm(ang)
        if speed>maxv:
            p.resetBaseVelocity(self.agent_id, angularVelocity=(np.array(ang)/speed*maxv).tolist())

    def step_simulation(self):
        p.stepSimulation()
        self.clamp_angular_velocity()

    def close(self):
        p.disconnect()
