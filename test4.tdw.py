from tdw.controller import Controller
from tdw.tdw_utils import TDWUtils
from tdw.add_ons.third_person_camera import ThirdPersonCamera
from tdw.add_ons.object_manager import ObjectManager

c = Controller()

c.communicate({"$type": "load_scene", "scene_name": "archviz_house_2019"})

avatar_id = "avatar"
c.add_ons.append(ThirdPersonCamera(avatar_id=avatar_id))
c.communicate({"$type": "create_avatar",
               "type": "A_Img_Caps",
               "id": "avatar",
               "position": {"x": 0, "y": 1.5, "z": 0}})


object_manager = ObjectManager()
c.add_ons.append(object_manager)

# Add a chair
chair_id = c.get_unique_id()
chair_position = {"x": 1, "y": 0, "z": 1}
c.add_object(
    model_name="chair_billiani_doll",
    position=chair_position,
    object_id=chair_id
)

# Add a table
table_id = c.get_unique_id()
table_position = {"x": -1, "y": 0, "z": -1}
c.add_object(
    model_name="table_round_metal_white",
    position=table_position,
    object_id=table_id
)

for i in range(1000):
    c.communicate([])
    
c.communicate({"$type": "terminate"})