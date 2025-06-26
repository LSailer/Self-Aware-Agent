
#   A Multi-Agent World Model with Self-Driven Curiosity Coordinated Exploration

This project explores the fascinating world of intrinsically motivated agents within a dynamic, physics-based environment. It extends the foundational work of "Learning to Play with Intrinsically-Motivated, Self-Aware Agents" (https://github.com/neuroailab/learning_to_play.git) into a multi-agent setting, allowing for complex interactions and emergent behaviors

##   Overview
Our simulation, built upon the robust PyBullet engine, models a vibrant environment where agents learn and adapt based on their own curiosity. Unlike traditional reinforcement learning, our agents are not driven by external rewards. Instead, they possess a "Self-Model" that encourages them to explore, experiment, and learn from their interactions with the world.

## Getting Started

This project uses `uv` for package and environment management. The dependencies are defined in `pyproject.toml`.

### 1. Install uv
First, install `uv` on your system. You can find the instructions at the [official uv installation guide](https://github.com/astral-sh/uv#installation).

### 2. Install Dependencies
Once `uv` is installed, you can install the project dependencies.

```
uv sync
```
## Usage
To run the simulation, use the following commands from the root of the project directory:
Single-Agent Simulation: 

```
python src/main.py --config configs.single_agent_config
```

Multi-Agent Simulation:
```
python src/main.py --config configs.multi_agent_config
```

## Project Structure

The project is organized into the following key files:

* `main.py`: The main entry point for running the simulation.
* `environment.py`: Defines the PyBullet environment, including the agents, objects, and their interactions.
* `single_agent_controller.py`: The controller for the single-agent simulation.
* `multi_agent_controller.py`: The controller for the multi-agent simulation.
* `networks.py`: Contains the implementations of the VAE, RNN, and Self-Model.
* `replay_buffer.py`: A simple replay buffer for storing and sampling experiences.
* `metric_logger.py`: Logs simulation metrics, such as rewards and losses, to a CSV file.
* `video_recorder.py`: Records the simulation as a video.

## Architecture

The agent's architecture is a combination of a predictive world model and a self-model that encourages exploratory behavior. The key components are:

* **Vision Model (VAE)**: A Variational Autoencoder that compresses high-dimensional observations into a low-dimensional latent vector.
* **Memory Model (RNN)**: A Recurrent Neural Network that learns the temporal dynamics of the environment.
* **Self-Model**: Predicts the expected error of the world model, which in turn drives the agent's curiosity.
