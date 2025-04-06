import pybullet as p
import pybullet_data
import numpy as np
import cv2
import math

from video_recorder import VideoRecorder

class Environment:
    def __init__(self):
        """Initialize the PyBullet environment."""

        # -- Konfiguriere hier deine Startpositionen --
        self.agent_start_pos = [0.0, 0.0, 0.2]
        self.agent_start_ori = [0, 0, 0, 1]  # Quaternion
        self.cube_start_pos  = [0.8, 0, 0.4]
        self.cube_start_ori  = [0, 0, 0, 1]
        self.cylinder_start_pos = [2.0, 2.0, 0.5]  # Beispielposition des Zylinders

        # Action Map
        self.action_map = {
            "forward":      [50.0,   0,   0,  0],
            "backward":     [-50.0,  0,   0,  0],
            "left":         [0, -50.0,   0,  0],
            "right":        [0,  50.0,   0,  0],
            "rotate_left":  [0,   0,   0,  5.0],
            "rotate_right": [0,   0,   0, -5.0],
            "stop":         [0,    0,   0,   0],  
        }
    
        # Connect to the PyBullet simulator in GUI mode: p.GUI else: p.DIRECT
        p.connect(p.GUI)
        p.setTimeStep(1./60.)
        p.setRealTimeSimulation(0)
        # Add search path for default PyBullet assets
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        print(pybullet_data.getDataPath())

        # Load a flat plane
        self.plane_id = p.loadURDF("plane.urdf")

        p.changeVisualShape(
            objectUniqueId=self.plane_id,
            linkIndex=-1,
            rgbaColor=[0.5, 0.5, 0.5, 1.0],  # z.B. hellgrau
        )

        # Initialize agent as a green sphere
        self.agent_id = p.createMultiBody(
            baseMass=3,
            baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_SPHERE, radius=0.2),
            baseVisualShapeIndex=p.createVisualShape(p.GEOM_SPHERE, radius=0.2, rgbaColor=[0, 1, 0, 1]),
            basePosition=self.agent_start_pos  # <-- Agent spawn
        )
        
        # Initialize a red cube as the target
        self.cube_id = p.createMultiBody(
            baseMass=6,  # Set small mass to allow pushing
            baseCollisionShapeIndex=p.createCollisionShape(
                p.GEOM_BOX, halfExtents=[0.4, 0.4, 0.4]
            ),
            baseVisualShapeIndex=p.createVisualShape(
                p.GEOM_BOX, halfExtents=[0.4, 0.4, 0.4], rgbaColor=[1, 0, 0, 1]
            ),
            basePosition=self.cube_start_pos  # <-- Cube spawn
        )

        # Initialize a blue cylinder
        self.cylinder_id = p.createMultiBody(
            baseMass=1,  # Masse des Zylinders (kann angepasst werden)
            baseCollisionShapeIndex=p.createCollisionShape(
                p.GEOM_CYLINDER, radius=0.2, height=1.0
            ),
            baseVisualShapeIndex=p.createVisualShape(
                p.GEOM_CYLINDER, radius=0.2, length=1.0, rgbaColor=[0, 0, 1, 1]  # Blau
            ),
            basePosition=self.cylinder_start_pos  # Position des Zylinders
        )

        # Set gravity in the environment
        p.setGravity(0, 0, -9.8)

        # Create a 10x10 room with walls
        self._create_room()

        # Reduce friction for smoother motion
        p.changeDynamics(
            self.plane_id, -1,
            lateralFriction=0.1,
            angularDamping=0.5,  # Hinzugefügte Drehmomentdämpfung
            linearDamping=0.5    # Hinzugefügte lineare Dämpfung
        )
        # Reduce mass for the agent
        p.changeDynamics(
            self.agent_id, -1,
            lateralFriction=1.0,    # Höherer Wert -> weniger Rutschen
            rollingFriction=0.005,  # Sorgt dafür, dass Ball realistisch rollt
            spinningFriction=0.005,
            restitution=0.0,         # Verhindert Hüpfen/Bouncen
            angularDamping=0.5,      # Hinzugefügte Drehmomentdämpfung
            linearDamping=0.5        # Hinzugefügte lineare Dämpfung
        )
        # Optional: Set dynamics for the cylinder, if necessary
        p.changeDynamics(
            self.cylinder_id, -1,
            lateralFriction=0.5,
            restitution=0.1,
            angularDamping=0.5,      # Hinzugefügte Drehmomentdämpfung für den Zylinder
            linearDamping=0.5        # Hinzugefügte lineare Dämpfung für den Zylinder
        )

    def _create_room(self):
        """Create a 10x10 room with walls."""
        wall_thickness = 0.2
        wall_height = 2.0
        wall_length = 5

        # Define positions and sizes of walls
        walls = [
            {"pos": [0,  wall_length / 2, wall_height / 2],
             "size": [wall_length / 2, wall_thickness / 2, wall_height / 2]},
            {"pos": [0, -wall_length / 2, wall_height / 2],
             "size": [wall_length / 2, wall_thickness / 2, wall_height / 2]},
            {"pos": [-wall_length / 2, 0, wall_height / 2],
             "size": [wall_thickness / 2, wall_length / 2, wall_height / 2]},
            {"pos": [ wall_length / 2, 0, wall_height / 2],
             "size": [wall_thickness / 2, wall_length / 2, wall_height / 2]},
        ]
        self.wall_ids = []
        for wall in walls:
            wall_id = p.createMultiBody(
                baseCollisionShapeIndex=p.createCollisionShape(
                    p.GEOM_BOX, halfExtents=wall["size"]
                ),
                baseVisualShapeIndex=p.createVisualShape(
                    p.GEOM_BOX, halfExtents=wall["size"]
                ),
                basePosition=wall["pos"],
                baseMass=0
            )
            self.wall_ids.append(wall_id)

    def reset(self):
        """Reset the environment to its initial state."""
        # Reset agent
        p.resetBasePositionAndOrientation(
            self.agent_id,
            self.agent_start_pos,
            self.agent_start_ori
        )
        # Reset cube
        p.resetBasePositionAndOrientation(
            self.cube_id,
            self.cube_start_pos,
            self.cube_start_ori
        )
        # Reset cylinder
        p.resetBasePositionAndOrientation(
            self.cylinder_id,
            self.cylinder_start_pos,
            [0, 0, 0, 1]  # Keine Rotation
        )
        # Reset velocities to ensure complete Stillstand
        p.resetBaseVelocity(
            self.agent_id,
            linearVelocity=[0, 0, 0],
            angularVelocity=[0, 0, 0]
        )
        p.resetBaseVelocity(
            self.cylinder_id,
            linearVelocity=[0, 0, 0],
            angularVelocity=[0, 0, 0]
        )

    def get_camera_image(self):
        """Capture the current camera view from the agent's perspective."""
        # Kamera-Offset: Position der Kamera relativ zum Ball
        camera_offset = [0.0, 0.0, 0.3]  # Kamera leicht über dem Zentrum des Agenten

        # Abrufen der Position und Orientierung des Balls
        agent_pos, agent_ori = p.getBasePositionAndOrientation(self.agent_id)

        # Berechnung der Vorwärtsrichtung basierend auf der aktuellen Yaw-Rotation des Balls
        euler = p.getEulerFromQuaternion(agent_ori)  # (Roll, Pitch, Yaw) in Radian
        yaw = euler[2]  # Yaw-Winkel (Rotation um die Z-Achse)

        # Definiere die Vorwärtsrichtung des Balls
        forward_dir = np.array([math.cos(yaw), math.sin(yaw), 0])

        # Position der Kamera relativ zum Ball
        camera_eye = np.array(agent_pos) + np.array(camera_offset)

        # Zielpunkt, auf den die Kamera schaut (in Vorwärtsrichtung des Balls)
        camera_target = camera_eye + forward_dir * 2  # 2 Einheiten in die Vorwärtsrichtung

        # Berechnung der View-Matrix
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=camera_eye.tolist(),
            cameraTargetPosition=camera_target.tolist(),
            cameraUpVector=[0, 0, 1]  # "Oben" ist die Z-Achse
        )

        # Definiere die Projektionsmatrix
        projection_matrix = p.computeProjectionMatrixFOV(
            fov=90, aspect=1.0, nearVal=0.1, farVal=10.0
        )


        # Erfasse das Kamerabild
        width, height, rgb_img, _, _ = p.getCameraImage(
            width=640, height=480,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix
        )
        
        # Konvertiere das Bild in ein RGB-Array
        rgb_array = np.array(rgb_img, dtype=np.uint8).reshape(height, width, 4)
        rgb_image = rgb_array[:, :, :3]  # Entferne den Alphakanal

        # Zusätzliche Debugging-Ausgabe für Rotationsgeschwindigkeit
        agent_vel, agent_ang_vel = p.getBaseVelocity(self.agent_id)

        return rgb_image

    def get_state(self):
        """Retrieve the current state of the environment."""
        # agent
        agent_pos, agent_ori = p.getBasePositionAndOrientation(self.agent_id)
        agent_vel, agent_ang_vel = p.getBaseVelocity(self.agent_id)
        # cube
        cube_pos, cube_ori = p.getBasePositionAndOrientation(self.cube_id)
        # cylinder
        cylinder_pos, cylinder_ori = p.getBasePositionAndOrientation(self.cylinder_id)
        return {
            "agent": {
                "position": agent_pos,
                "orientation": agent_ori,
                "velocity": agent_vel,
                "angular_velocity": agent_ang_vel,
            },
            "cube": {
                "position": cube_pos,
                "orientation": cube_ori,
            },
            "cylinder": {
                "position": cylinder_pos,
                "orientation": cylinder_ori,
            }
        }

    def apply_action(self, action):
        """Apply an action to the agent by applying forces/torques."""
        if action in self.action_map:
            force = self.action_map[action]  # [vx, vy, dummy, torque_z]
            vx, vy, _, torque_z = force

            if action == "stop":
                # Stoppe den Agenten durch Zurücksetzen der Geschwindigkeiten
                p.resetBaseVelocity(
                    self.agent_id,
                    linearVelocity=[0, 0, 0],
                    angularVelocity=[0, 0, 0]
                )
            else:
                agent_pos, agent_ori = p.getBasePositionAndOrientation(self.agent_id)
                rotation_matrix = p.getMatrixFromQuaternion(agent_ori)
                forward_vec = np.array(rotation_matrix[0:3])  # x-Achse
                right_vec   = np.array(rotation_matrix[3:6])  # y-Achse

                # Combine forward/backward + left/right
                move_vec = forward_vec * vx + right_vec * vy

                # If we have movement
                if vx != 0 or vy != 0:
                    p.applyExternalForce(
                        objectUniqueId=self.agent_id,
                        linkIndex=-1,
                        forceObj=move_vec.tolist(),
                        posObj=agent_pos,
                        flags=p.WORLD_FRAME
                    )

                # If we have rotation, set angular velocity directly statt kontinuierlicher Drehmoment
                if torque_z != 0:
                    # Hier setzen wir die Rotationsgeschwindigkeit direkt
                    current_angular_velocity = p.getBaseVelocity(self.agent_id)[1]
                    new_angular_velocity = [0, 0, torque_z]
                    p.resetBaseVelocity(self.agent_id, angularVelocity=new_angular_velocity)
        else:
            print(f"Unknown action: {action}")

    def clamp_angular_velocity(self, max_angular_velocity=5.0):
        """Begrenze die Rotationsgeschwindigkeit des Agenten."""
        agent_vel, agent_ang_vel = p.getBaseVelocity(self.agent_id)
        angular_velocity = np.array(agent_ang_vel)
        speed = np.linalg.norm(angular_velocity)
        
        if speed > max_angular_velocity:
            angular_velocity = angular_velocity / speed * max_angular_velocity
            p.resetBaseVelocity(self.agent_id, angularVelocity=angular_velocity.tolist())

    def step_simulation(self):
        """Step the simulation forward."""
        p.stepSimulation()
        self.clamp_angular_velocity()

    def close(self):
        """Disconnect the simulation."""
        p.disconnect()
