from tdw.controller import Controller
from tdw.tdw_utils import TDWUtils

class TDWEnvironment(Controller):
    def __init__(self):
        super().__init__()
        self.objects = []

    def create_environment(self):
        """Create a simple TDW environment with an agent and an object."""
        # Add a room
        self.communicate(TDWUtils.create_empty_room(10,10))

        # Add an object
        object_id = self.get_unique_id()
        self.communicate({"$type": "add_object",
                        "name": "cube",
                        "id": object_id,
                        "position": {"x": 0, "y": 0, "z": 0},
                        "rotation": {"x": 0, "y": 0, "z": 0}})
        self.objects.append(object_id)

        # Add an agent (e.g., a sphere to simulate the agent's body)
        agent_id = self.get_unique_id()
        self.communicate({"$type": "add_object",
                        "name": "sphere",
                        "id": agent_id,
                        "position": {"x": 1, "y": 0, "z": 1},
                        "rotation": {"x": 0, "y": 0, "z": 0}})
        self.objects.append(agent_id)
        print("Environment created with agent and object.")

    def move_agent(self, agent_id, direction):
        """Move the agent in a specified direction.

        Args:
            agent_id: The ID of the agent object.
            direction: A dictionary with 'x', 'y', 'z' values for movement.
        """
        self.communicate({"$type": "teleport_object",
                          "id": agent_id,
                          "position": direction})

    def close_environment(self):
        """Terminate the TDW simulation."""
        self.communicate({"$type": "terminate"})

if __name__ == "__main__":
    # Initialize the environment
    env = TDWEnvironment()
    
    # Create the environment
    env.create_environment()

    # Move the agent
    agent_id = env.objects[1]
    env.move_agent(agent_id, direction={"x": 2, "y": 0, "z": 2})

    # Close the environment
    env.close_environment()
