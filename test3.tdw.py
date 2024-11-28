# tdw_environment.py

import os
import time
import numpy as np
import random
from collections import namedtuple
from pdb import set_trace

from tdw.controller import Controller
from tdw.tdw_utils import TDWUtils
from tdw.output_data import OutputData, Images, Transforms, Rigidbodies, SegmentationColors
from tdw.librarian import ModelLibrarian
from tdw.add_ons.third_person_camera import ThirdPersonCamera
from tdw.add_ons.image_capture import ImageCapture

# Constants and default parameters
SHADERS = ['_img']
HDF5_NAMES = ['images']


class Environment(object):
    def __init__(self):
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()

    def step(self, action):
        raise NotImplementedError()


class TDWClientEnvironment(Environment):
    def __init__(self,
                 unity_seed,
                 random_seed,
                 screen_dims=(128, 170),
                 room_dims=(20., 20.),
                 image_dir="",
                 max_steps=100,
                 build_path=None  # Add build_path parameter
                 ):
        self.unity_seed = unity_seed
        self.random_seed = random_seed
        self.screen_dims = screen_dims
        self.room_dims = room_dims
        self.image_dir = image_dir
        self.num_steps = 0
        self.max_steps = max_steps
        self.rng = np.random.RandomState(random_seed)
        self.model_librarian = ModelLibrarian()
        self.object_ids = []
        self.camera = ThirdPersonCamera(position={"x": 0, "y": 10, "z": -10},
                                        look_at={"x": 0, "y": 0, "z": 0})
        if self.image_dir:
            self.capture = ImageCapture(path=self.image_dir, avatar_ids=["a"])
            self.controller = Controller(
                launch_build=True,
                #build_path=build_path,  # Specify the build path here
                #launch_build_check_frequency=5  # Optional: Adjust the check frequency
            )
            self.controller.add_ons.extend([self.camera, self.capture])
        else:
            self.controller = Controller(
                launch_build=True,
                #build_path=build_path
            )
            self.controller.add_ons.append(self.camera)

    def reset(self, *round_info):
        self.num_steps = 0
        self.round_info = round_info

        # Load the scene
        self.controller.communicate(TDWUtils.create_empty_room(10, 10))

        # Set random seed
        self.controller.communicate({"$type": "set_random_seed", "seed": self.unity_seed})

        # Add objects based on round_info
        self.object_ids = []

        for info in round_info:
            num_items = info['num_items']
            model_names = self.get_model_names(num_items)
            positions = self.get_random_positions(num_items)

            for model_name, position in zip(model_names, positions):
                object_id = self.controller.get_unique_id()
                self.controller.communicate(self.get_add_object_command(model_name, object_id, position))
                self.object_ids.append(object_id)

        # Initial observation
        return self._observe_world()

    def get_add_object_command(self, model_name, object_id, position):
        return {
            "$type": "add_object",
            "name": model_name,
            "url": self.model_librarian.get_model_url(model_name),
            "scale_factor": 1.0,
            "id": object_id,
            "position": position,
            "rotation": {"x": 0, "y": 0, "z": 0}
        }

    def get_model_names(self, num_items):
        # For simplicity, use 'cube' model for all items
        return ['cube' for _ in range(num_items)]

    def get_random_positions(self, num_items):
        positions = []
        for _ in range(num_items):
            x = random.uniform(-self.room_dims[0]/2, self.room_dims[0]/2)
            z = random.uniform(-self.room_dims[1]/2, self.room_dims[1]/2)
            positions.append({"x": x, "y": 0, "z": z})
        return positions

    def _observe_world(self):
        # Send commands to capture images and object data
        resp = self.controller.communicate([
            {"$type": "send_images", "frequency": "once"},
            {"$type": "send_transforms", "frequency": "once"},
            {"$type": "send_rigidbodies", "frequency": "once"},
            {"$type": "send_segmentation_colors", "frequency": "once"}
        ])

        # Parse the response to extract observations
        images = None
        transforms = {}
        rigidbodies = {}
        segmentation_colors = {}

        for r in resp:
            if OutputData.get_data_type_id(r) == "imag":
                images_data = Images(r)
                images = images_data.get_image(0)
            elif OutputData.get_data_type_id(r) == "tran":
                transforms_data = Transforms(r)
                for i in range(transforms_data.get_num()):
                    object_id = transforms_data.get_id(i)
                    position = transforms_data.get_position(i)
                    transforms[object_id] = {"x": position[0], "y": position[1], "z": position[2]}
            elif OutputData.get_data_type_id(r) == "rigi":
                rigidbodies_data = Rigidbodies(r)
                for i in range(rigidbodies_data.get_num()):
                    object_id = rigidbodies_data.get_id(i)
                    velocity = rigidbodies_data.get_velocity(i)
                    rigidbodies[object_id] = {"x": velocity[0], "y": velocity[1], "z": velocity[2]}
            elif OutputData.get_data_type_id(r) == "segm":
                segmentation_data = SegmentationColors(r)
                for i in range(segmentation_data.get_num()):
                    object_id = segmentation_data.get_object_id(i)
                    color = segmentation_data.get_object_color(i)
                    segmentation_colors[object_id] = {"r": color[0], "g": color[1], "b": color[2]}

        observation = {
            'images': images,
            'transforms': transforms,
            'rigidbodies': rigidbodies,
            'segmentation_colors': segmentation_colors
        }

        self.num_steps += 1
        return observation

    def step(self, action):
        # Create commands based on the action
        commands = []

        # For example, apply forces to objects
        for object_id, force in zip(self.object_ids, action):
            commands.append({
                "$type": "apply_force_to_object",
                "id": object_id,
                "force": force
            })

        # Send the commands and get the response
        resp = self.controller.communicate(commands)

        # Observe the new state
        observation = self._observe_world()

        # Check termination condition
        done = self._termination_condition()

        return observation, done

    def _termination_condition(self):
        # Define your termination condition
        if self.num_steps >= self.max_steps:
            self.controller.communicate({"$type": "terminate"})
            return True
        else:
            return False


# Example usage
if __name__ == "__main__":
    # Set up parameters
    unity_seed = 1
    random_seed = 1
    screen_dims = (128, 170)
    room_dims = (20., 20.)
    image_dir = "images"

    # Specify the path to the Unity build
    build_path = "/TDW/TDW.app"  # Update this path to your TDW.app

    # Create the environment
    env = TDWClientEnvironment(
        unity_seed=unity_seed,
        random_seed=random_seed,
        screen_dims=screen_dims,
        room_dims=room_dims,
        image_dir=image_dir,
        max_steps=10,
        build_path=build_path  # Pass the build_path here
    )

    # Reset the environment
    env.reset({'num_items': 5})

    done = False
    while not done:
        # Generate random actions for demonstration
        action = [{"x": random.uniform(-5, 5), "y": random.uniform(-5, 5), "z": random.uniform(-5, 5)} for _ in env.object_ids]

        # Take a step in the environment
        observation, done = env.step(action)

        # You can process the observation as needed
        print(f"Step {env.num_steps}:")
        print(f"Transforms: {observation['transforms']}")
        print(f"Rigidbodies: {observation['rigidbodies']}")

    print("Simulation completed.")
