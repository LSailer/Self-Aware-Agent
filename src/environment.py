import os
import torch
import pybullet as p
import pybullet_data
import numpy as np
import math

class Environment:
    def __init__(self, use_gui, sky_color=(10, 130, 200), wall_color=(0, 0, 0, 0)):
        self.use_gui = use_gui
        self.agent_start_pos    = [0.0, 0.0, 0.2]
        self.agent_start_ori    = [0, 0, 0, 1]
        self.cylinder_start_pos = [2.4, 0.2, 0.5]
        self.cylinder_start_ori = [0, 0, 0, 1]
        self.disk_start_pos     = [0.0, 2.0, 0]
        self.disk_start_ori     = [0, 0, 0, 1]
        self.pyramid_start_pos  = [-2.0, -0.5, 0.0]
        self.pyramid_start_ori  = [0, 0, 0, 1]
        self.sphere_start_pos = [0.5, -2.0, 0.3]   # X, Y, Z
        self.sphere_start_ori = [0, 0, 0, 1]

        self.action_map = {
            'forward':      [50.0,  0,    0, 0],
            'backward':     [-50.0, 0,    0, 0],
            'left':         [0,    -50.0, 0, 0],
            'right':        [0,     50.0, 0, 0],
            'rotate_left':  [0,     0,    0, 5.0],
            'rotate_right': [0,     0,    0, -5.0],
            'stop':         [0,     0,    0, 0],
        }

        if self.use_gui:
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

        self.sky_color      = np.array(sky_color[:3], dtype=np.uint8)
        self.sky_color_norm = (self.sky_color.astype(np.float32) / 255.0).tolist()
        if wall_color is None:
            wall_color = (*sky_color[:3], 255)
        wc_rgb, wc_a = wall_color[:3], wall_color[3]
        self.wall_color      = np.array(wc_rgb, dtype=np.uint8)
        self.wall_alpha      = wc_a / 255.0
        self.wall_color_norm = (self.wall_color.astype(np.float32) / 255.0).tolist()

        self.plane_id = p.loadURDF('plane.urdf')
        p.changeVisualShape(self.plane_id, -1,
                            rgbaColor=[0.5, 0.5, 0.5, 1.0],
                            specularColor=[0, 0, 0])

        half = 0.2
        self.agent_id = p.createMultiBody(
            baseMass=6,
            baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[half]*3),
            baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[half]*3,
                                                     rgbaColor=[0, 1, 0, 1]),
            basePosition=self.agent_start_pos
        )

        self.disk_id = p.createMultiBody(
            baseMass=0.01,
            baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_CYLINDER, radius=0.7, height=0.1),
            baseVisualShapeIndex=p.createVisualShape(p.GEOM_CYLINDER, radius=0.7, length=0.1,
                                                     rgbaColor=[0, 0, 1, 1]),
            basePosition=self.cylinder_start_pos
        )

        self.cylinder_id = p.createMultiBody(
            baseMass=0.01,
            baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_CYLINDER, radius=0.2, height=1.0),
            baseVisualShapeIndex=p.createVisualShape(p.GEOM_CYLINDER, radius=0.2, length=1.0,
                                                     rgbaColor=[1, 0, 0, 1]),
            basePosition=self.disk_start_pos
        )

        base_dir = os.path.dirname(os.path.realpath(__file__))
        urdf_path = os.path.join(base_dir, 'pyramid.urdf')
        self.pyramid_id = p.loadURDF(urdf_path,
                                     basePosition=self.pyramid_start_pos,
                                     baseOrientation=self.pyramid_start_ori)
        
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
        self.sphere_id = p.createMultiBody(
            baseMass=sphere_mass,
            baseCollisionShapeIndex=sphere_collision,
            baseVisualShapeIndex=sphere_visual,
            basePosition=self.sphere_start_pos,
            baseOrientation=self.sphere_start_ori
        )


        p.setGravity(0, 0, -9.8)
        self._create_room()

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

    def reset(self):
        p.resetBasePositionAndOrientation(self.agent_id, self.agent_start_pos, self.agent_start_ori)
        p.resetBasePositionAndOrientation(self.cylinder_id, self.cylinder_start_pos, self.cylinder_start_ori)
        p.resetBasePositionAndOrientation(self.disk_id, self.disk_start_pos, self.disk_start_ori)
        p.resetBasePositionAndOrientation(self.pyramid_id, self.pyramid_start_pos, self.pyramid_start_ori)
        for oid in [self.agent_id, self.cylinder_id, self.disk_id, self.pyramid_id]:
            p.resetBaseVelocity(oid, [0, 0, 0], [0, 0, 0])

    def get_camera_image(self):
        agent_pos, agent_ori = p.getBasePositionAndOrientation(self.agent_id)
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

    def get_state(self):
        agent_pos, agent_ori = p.getBasePositionAndOrientation(self.agent_id)
        agent_vel, agent_ang = p.getBaseVelocity(self.agent_id)
        cpos, _ = p.getBasePositionAndOrientation(self.cylinder_id)
        dpos, _ = p.getBasePositionAndOrientation(self.disk_id)
        ppos, _ = p.getBasePositionAndOrientation(self.pyramid_id)
        return {
            'agent':   {'position': agent_pos, 'orientation': agent_ori, 'velocity': agent_vel, 'angular_velocity': agent_ang},
            'cylinder':{'position': cpos},
            'disk':    {'position': dpos},
            'pyramid': {'position': ppos}
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
            p.resetBaseVelocity(self.agent_id, angularVelocity=[0, 0, tz])

    def clamp_angular_velocity(self, maxv=5.0):
        _, ang = p.getBaseVelocity(self.agent_id)
        speed = np.linalg.norm(ang)
        if speed > maxv:
            p.resetBaseVelocity(self.agent_id,
                                angularVelocity=(np.array(ang)/speed*maxv).tolist())

    def step_simulation(self):
        p.stepSimulation()
        self.clamp_angular_velocity()

    def close(self):
        p.disconnect()
