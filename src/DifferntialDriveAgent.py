import pybullet as p
import pybullet_data
import numpy as np

class DifferentialDriveAgent:
    def __init__(self, start_pos, start_ori):
        self.start_pos = start_pos
        self.start_ori = start_ori  # Quaternion

        self.robot_id = None
        # Index 0 -> linkes Rad, Index 1 -> rechtes Rad
        self.left_joint_index = 0
        self.right_joint_index = 1

        # Mögliche Aktionstasten:
        self.action_map = {
            "forward":      [10.0,  10.0],   # Beide Räder vorwärts
            "backward":     [-10.0, -10.0],  # Beide Räder rückwärts
            "rotate_left":  [-5.0,   5.0],   # Linksdrehung
            "rotate_right": [ 5.0,  -5.0],   # Rechtsdrehung
            "stop":         [ 0.0,   0.0],   # Keine Bewegung
        }

    def create(self):
        """
        Erstellt einen Differential-Drive-Roboter in EINEM MultiBody
        mit zwei Revolute-Joints (linkes + rechtes Rad).
        """

        # -- Basiskörper (Zylinder) --
        base_radius = 0.3
        base_height = 0.2

        base_collision = p.createCollisionShape(
            shapeType=p.GEOM_CYLINDER,
            radius=base_radius,
            height=base_height
        )
        base_visual = p.createVisualShape(
            shapeType=p.GEOM_CYLINDER,
            radius=base_radius,
            length=base_height,
            rgbaColor=[0, 1, 0, 1]  # grüne Basis
        )

        # -- Räder (Collision + Visual) --
        wheel_radius = 0.1
        wheel_length = 0.05

        # Wir erstellen 2 CollisionShapes und 2 VisualShapes (je ein Rad)
        wheel_collision_left = p.createCollisionShape(
            p.GEOM_CYLINDER, radius=wheel_radius, height=wheel_length
        )
        wheel_collision_right = p.createCollisionShape(
            p.GEOM_CYLINDER, radius=wheel_radius, height=wheel_length
        )
        wheel_visual_left = p.createVisualShape(
            p.GEOM_CYLINDER, radius=wheel_radius, length=wheel_length, rgbaColor=[0.1, 0.1, 0.1, 1]
        )
        wheel_visual_right = p.createVisualShape(
            p.GEOM_CYLINDER, radius=wheel_radius, length=wheel_length, rgbaColor=[0.1, 0.1, 0.1, 1]
        )

        # Alle Felder brauchen dieselbe Länge (2 Links -> 2 Einträge)
        link_masses = [1.0, 1.0]
        linkParentIndices = [0, 0]
        link_collision_indices = [wheel_collision_left, wheel_collision_right]
        link_visual_indices    = [wheel_visual_left,    wheel_visual_right]

        # Relative Positionen der Räder zur Basis
        link_positions = [
            [-0.3, 0.0, 0.0],  # linkes Rad
            [ 0.3, 0.0, 0.0]   # rechtes Rad
        ]
        link_orientations = [
            p.getQuaternionFromEuler([0, 0, 0]),
            p.getQuaternionFromEuler([0, 0, 0])
        ]

        link_inertial_frames = [
            [0, 0, 0],
            [0, 0, 0]
        ]
        link_inertial_orientations = [
            [0, 0, 0, 1],
            [0, 0, 0, 1]
        ]
        

        # Beide Gelenke: JOINT_REVOLUTE, Rotationsachse y -> [0,1,0]
        link_joint_types = [p.JOINT_REVOLUTE, p.JOINT_REVOLUTE]
        link_joint_axes  = [[0,1,0],         [0,1,0]]
        print("linkMasses =", link_masses)
        print("linkCollisionShapeIndices =", link_collision_indices)
        print("linkVisualShapeIndices =", link_visual_indices)
        print("linkPositions =", link_positions)
        print("linkOrientations =", link_orientations)
        print("linkInertialFramePositions =", link_inertial_frames)
        print("linkInertialFrameOrientations =", link_inertial_orientations)
        print("linkJointTypes =", link_joint_types)
        print("linkJointAxis =", link_joint_axes)

        print("Lengths:",
            len(link_masses),
            len(link_collision_indices),
            len(link_visual_indices),
            len(link_positions),
            len(link_orientations),
            len(link_inertial_frames),
            len(link_inertial_orientations),
            len(link_joint_types),
            len(link_joint_axes))
     
        
        # createMultiBody
        self.robot_id = p.createMultiBody(
            baseMass=5.0,
            baseCollisionShapeIndex=base_collision,
            baseVisualShapeIndex=base_visual,
            basePosition=self.start_pos,
            baseOrientation=self.start_ori,

            linkMasses=link_masses,
            linkCollisionShapeIndices=link_collision_indices,
            linkVisualShapeIndices=link_visual_indices,
            linkPositions=link_positions,
            linkOrientations=link_orientations,
            linkInertialFramePositions=link_inertial_frames,
            linkInertialFrameOrientations=link_inertial_orientations,
            linkParentIndices=linkParentIndices,            # <--- WICHTIG!
            linkJointTypes=link_joint_types,
            linkJointAxis=link_joint_axes
        )
        print("Number of joints in the robot:", p.getNumJoints(self.robot_id))

        


        # Motor-Kontrolle auf 0
        p.setJointMotorControl2(
            bodyUniqueId=self.robot_id,
            jointIndex=self.left_joint_index,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocity=0,
            force=100
        )
        p.setJointMotorControl2(
            bodyUniqueId=self.robot_id,
            jointIndex=self.right_joint_index,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocity=0,
            force=100
        )

    def apply_action(self, action):
        """
        Stellt die Zielgeschwindigkeit der beiden Rad-Gelenke ein.
        z.B. "forward", "rotate_left", ...
        """
        if action not in self.action_map:
            raise ValueError(f"Ungültige Aktion: {action}")

        left_speed, right_speed = self.action_map[action]

        p.setJointMotorControl2(
            bodyUniqueId=self.robot_id,
            jointIndex=self.left_joint_index,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocity=left_speed,
            force=100
        )
        p.setJointMotorControl2(
            bodyUniqueId=self.robot_id,
            jointIndex=self.right_joint_index,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocity=right_speed,
            force=100
        )

    def reset(self):
        """
        Bringt den Roboter in seine Start-Position/Ausrichtung zurück;
        Gelenkwinkel auf 0. 
        Danach stoppt der Roboter (targetVelocity=0).
        """
        p.resetBasePositionAndOrientation(
            self.robot_id, 
            self.start_pos, 
            self.start_ori
        )
        # Beide JointStates = 0
        p.resetJointState(self.robot_id, self.left_joint_index, 0)
        p.resetJointState(self.robot_id, self.right_joint_index, 0)

        # Motoren auf 0
        p.setJointMotorControl2(
            bodyUniqueId=self.robot_id,
            jointIndex=self.left_joint_index,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocity=0,
            force=100
        )
        p.setJointMotorControl2(
            bodyUniqueId=self.robot_id,
            jointIndex=self.right_joint_index,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocity=0,
            force=100
        )
