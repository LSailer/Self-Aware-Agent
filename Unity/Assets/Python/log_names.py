import UnityEngine as ue

"""

"""
#Print Objects
"""
objects = ue.Object.FindObjectsOfType(ue.GameObject)
# for go in objects:
#     ue.Debug.Log(go.name)

"""
#Move Agent-Object
"""
for go in objects:
    if go.name == "Agent":
        ue.Debug.Log(f"Found Agent: {go.name}")

        # Take Controll Component
        agent_transform = go.transform

        # Move Agent Object
        new_position = ue.Vector3(agent_transform.position.x + 1, 
                                  agent_transform.position.y, 
                                  agent_transform.position.z)
        agent_transform.position = new_position

        ue.Debug.Log(f"Agent moved to: {agent_transform.position}")

"""

# Finde das "Agent"-Objekt
agent = None
objects = ue.Object.FindObjectsOfType(ue.GameObject)
for go in objects:
    if go.name == "Agent":
        agent = go
        break

if agent:
    # Original Position
    original_position = agent.transform.position

    # Move Agent in Play Mode
    def move_agent():
        
        if ue.Application.isPlaying:
            # Bewege den Agenten z. B. nach rechts
            agent.transform.position += ue.Vector3(-0.5, 0.0, 0.2)
            ue.Debug.Log(f"Agent position: {agent.transform.position}")
        else:
            ue.Debug.Log("Not in Play Mode!")

    # Reset Position after the Game
    def reset_agent_position():
        agent.transform.position = original_position
        ue.Debug.Log("Agent position reset!")

    move_agent()
else:
    ue.Debug.Log("Agent not found!")
