from io import BytesIO
import numpy as np
import random
import json
import os
import signal
import time
from PIL import Image

# TDW imports
from tdw.controller import Controller
from tdw.add_ons.third_person_camera import ThirdPersonCamera
from tdw.add_ons.image_capture import ImageCapture
from tdw.add_ons.object_manager import ObjectManager
from tdw.output_data import OutputData, Images, Transforms, Rigidbodies, Bounds
from tdw.backend.paths import EXAMPLE_CONTROLLER_OUTPUT_PATH

class Environment(object):
    def __init__(self):
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()

    def step(self, action):
        '''Returns observation after taking action.
        And a boolean representing whether .reset() should
        be called.
        '''
        raise NotImplementedError()
    
class TDWClientEnvironment(Environment):
    def __init__(self,
                 unity_seed,
                 random_seed,
                 screen_dims=(128, 170),
                 room_dims=(20., 20.),
                 image_dir="",
                 **kwargs):
        self.unity_seed = unity_seed
        self.random_seed = random_seed
        self.screen_height, self.screen_width = screen_dims
        self.room_width, self.room_length = room_dims
        self.image_dir = image_dir
        self.rng = np.random.RandomState(random_seed)
        self.controller = Controller()
        # Add-ons
        self.camera = ThirdPersonCamera(position={"x": 0, "y": 1.6, "z": -2},
                                        look_at={"x": 0, "y": 0, "z": 0},
                                        avatar_id="a")
        self.image_capture = ImageCapture(
            avatar_ids=["a"],
            pass_masks=["_img", "_id"],  # Use strings for pass masks
            path=image_dir if image_dir else EXAMPLE_CONTROLLER_OUTPUT_PATH.joinpath("tdw_env_images"),
            png=True
        )
        self.object_manager = ObjectManager(transforms=True, rigidbodies=True)
        self.controller.add_ons.extend([self.camera, self.image_capture, self.object_manager])
        self.num_steps = 0
        self.nn_frames = 0
        self.obs_raw = None
        self.round_info = None
        self._avatar_position = None
        self._avatar_forward = None
        self.environment_pid = None

    def reset(self, *round_info):
        self.nn_frames = 0
        self.round_info = round_info
        # Load the scene
        commands = [
            {"$type": "load_scene", "scene_name": "ProcGenScene"},
            {"$type": "create_empty_room", "width": self.room_width, "length": self.room_length},
            {"$type": "set_screen_size", "width": self.screen_width, "height": self.screen_height},
            {"$type": "send_images", "frequency": "always"},
            {"$type": "send_transforms", "frequency": "always"},
            {"$type": "send_rigidbodies", "frequency": "always"},
            {"$type": "send_bounds", "frequency": "always"}
        ]
        self.controller.communicate(commands)
        # Update avatar's position and forward vector
        self._observe_world()
        return self._observe_world()
    
    def _observe_world(self):
        # Capture images and other data from the environment
        resp = self.controller.communicate([])
        observation = self.process_response(resp)
        self.obs_raw = observation
        return observation

    def process_response(self, resp):
        observation = {}
        for r in resp:
            r_id = OutputData.get_data_type_id(r)
            if r_id == "IMAG":
                images = Images(r)
                for i in range(images.get_num_passes()):
                    pass_mask_name = images.get_pass_mask_name(i)
                    if pass_mask_name == "img":
                        img_data = images.get_image(i)
                        observation['images1'] = np.array(Image.open(BytesIO(img_data)))
                    elif pass_mask_name == "id":
                        img_data = images.get_image(i)
                        observation['objects1'] = np.array(Image.open(BytesIO(img_data)))
            elif r_id == "TRAN":
                transforms = Transforms(r)
                for i in range(transforms.get_num()):
                    if transforms.get_id(i) == self.camera.avatar_id:
                        self._avatar_position = transforms.get_position(i)
                        self._avatar_forward = transforms.get_forward(i)
            elif r_id == "RBDS":
                pass  # Process rigidbodies if needed
            elif r_id == "BOUN":
                pass  # Process bounds if needed
        # Add object data from ObjectManager
        observation['objects'] = self.object_manager.objects_static
        observation['transforms'] = self.object_manager.transforms
        observation['rigidbodies'] = self.object_manager.rigidbodies
        return observation

    def step(self, action):
        # Update avatar's position and forward vector
        self._observe_world()

        # Build commands
        commands = []
        masked_action = action.copy()
        action = self.limits * action
        wall_feedback = self._ego_helper(action, commands)
        obj_there_mask = self._object_interaction_helper(action, commands)
        resp = self.controller.communicate(commands)
        self.observation = self.process_response(resp)
        self.observation['wall_feedback'] = wall_feedback
        masked_action[0] = masked_action[0] * wall_feedback
        for (slot_num, obj_there_indicator) in enumerate(obj_there_mask):
            start_idx = 2 + slot_num * self.single_action_len
            end_idx = start_idx + self.single_action_len
            masked_action[start_idx:end_idx] *= float(obj_there_indicator)
        self.observation['action_post'] = masked_action
        term_signal = self._termination_condition()
        return self.observation, term_signal

    def _termination_condition(self):
        # Implement your termination logic here
        return False

class TelekineticMagicianEnvironment(TDWClientEnvironment):
    def __init__(self,
                    limits,
                    max_interaction_distance,
                    wall_safety,
                    do_torque,
                    **kwargs):
        self.limits = np.array(limits)
        self.max_interaction_distance = max_interaction_distance
        self.wall_safety = wall_safety
        self.do_torque = do_torque
        self.single_action_len = 6 if self.do_torque else 3
        super(TelekineticMagicianEnvironment, self).__init__(**kwargs)

    def _termination_condition(self):
        # Implement your termination logic here
        return False

    def _ego_helper(self, action, commands):
        agent_vel = action[0]
        wall_feedback = 1.0
        # Compute proposed next position
        proposed_next_position = np.array(self._avatar_position) + agent_vel * np.array(self._avatar_forward)
        if (proposed_next_position[0] < self.wall_safety or
            proposed_next_position[0] > self.room_width - 0.5 - self.wall_safety or
            proposed_next_position[2] < self.wall_safety or
            proposed_next_position[2] > self.room_length - 0.5 - self.wall_safety):
            agent_vel = 0.0
            wall_feedback = 0.0
        commands.extend([
            {
                "$type": "move_avatar_forward_by",
                "magnitude": agent_vel,
                "avatar_id": self.camera.avatar_id
            },
            {
                "$type": "rotate_avatar_by",
                "angle": action[1],
                "avatar_id": self.camera.avatar_id
            }
        ])
        return wall_feedback

    def _object_interaction_helper(self, action, commands):
        # Implement object interaction logic
        obj_there_mask = []
        num_interacted = 0
        # Access object positions from object_manager
        object_positions = {}
        for obj_id in self.object_manager.objects_static:
            if obj_id != self.camera.avatar_id:
                object_positions[obj_id] = self.object_manager.transforms[obj_id].position
        # Compute distances and apply forces
        for obj_id, position in object_positions.items():
            distance = np.linalg.norm(np.array(position) - np.array(self._avatar_position))
            if distance < self.max_interaction_distance:
                # Object is within interaction distance
                force_start_idx = 2 + num_interacted * self.single_action_len
                force = action[force_start_idx:force_start_idx+3]
                if self.do_torque:
                    torque = action[force_start_idx+3:force_start_idx+6]
                else:
                    torque = [0.0, 0.0, 0.0]
                commands.append({
                    "$type": "apply_force_to_object",
                    "id": obj_id,
                    "force": {"x": force[0], "y": force[1], "z": force[2]}
                })
                if self.do_torque:
                    commands.append({
                        "$type": "apply_torque_to_object",
                        "id": obj_id,
                        "torque": {"x": torque[0], "y": torque[1], "z": torque[2]}
                    })
                obj_there_mask.append(1.0)
                num_interacted += 1
            else:
                obj_there_mask.append(0.0)
        return obj_there_mask

if __name__ == "__main__":
    env = TelekineticMagicianEnvironment(
        limits=[1.0, 1.0],
        max_interaction_distance=5.0,
        wall_safety=0.5,
        do_torque=False,
        unity_seed=0,
        random_seed=0,
        image_dir=""  # Set your image directory if needed
    )
    env.reset()
    for _ in range(10):
        action = np.array([0.1, 5.0])  # Use a NumPy array
        observation, done = env.step(action)
        if done:
            break
    env.controller.communicate({"$type": "terminate"})
