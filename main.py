import curiosity.models.my_model as my_model
import curiosity.data.environment as environment
import curiosity.interaction.agents as agents


def main():
    # Initialize environment
    print("Initializing environment...")
    env = environment.Environment()
    
    # Load or initialize the model
    print("Loading model...")
    model = my_model.MyModel()
    
    # Create agent
    print("Creating agent...")
    agent = agents.Agent(model, env)
    
    # Train or run interactions
    print("Starting interaction...")
    agent.run()


if __name__ == "__main__":
    main()
