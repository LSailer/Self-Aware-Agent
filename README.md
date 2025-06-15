
#   Intrinsically Motivated Multi-Agent Simulation

This project explores intrinsically motivated agents in a physics-based simulation, extending the work of "Learning to Play with Intrinsically-Motivated, Self-Aware Agents" (https://github.com/neuroailab/learning_to_play.git) to a multi-agent setting.

##   Overview

The simulation uses PyBullet to model a simple environment with a spherical agent, a cube, and a cylinder. The agent learns to interact with the environment based on a curiosity-driven reward mechanism.  The agent's curiosity is driven by a Self Model that predicts the World Model's prediction error.


* `environment.py`:  Sets up the PyBullet environment, manages the agent and objects, and provides sensor input (camera images).
* `curiosity_driven_agent.py`:  Defines the agent, which uses a VAE to encode images, an RNN to model temporal dynamics, and a Self Model to predict curiosity.
* `models.py`: Contains the VAE, RNN, and Self Model neural network implementations.
* `simulation.py`:  Orchestrates the main simulation loop, controlling agent-environment interaction and model training.
* `metric_logger.py`:  Handles logging of simulation metrics (rewards, losses, etc.).
* `video_recorder.py`:  Records the simulation as a video.
* `utility.py`:  Provides utility functions, such as visualization.

##   Functions

###   Environment (`environment.py`)

* `__init__()`:  Initializes the PyBullet environment, loads the agent and objects, and sets up the room.
* `reset()`:  Resets the environment to its initial state.
* `get_camera_image()`:  Captures the camera view from the agent's perspective.
* `get_state()`:  Retrieves the current state of the environment (agent and object positions/orientations).
* `apply_action(action)`: Applies a given action to the agent.
* `step_simulation()`:  Advances the simulation by one step.
* `close()`:  Disconnects from the PyBullet simulator.

###   CuriosityDrivenAgent (`curiosity_driven_agent.py`)

* `__init__(actions, latent_dim, rnn_hidden_dim, buffer_size, batch_size, device)`:  Initializes the agent with its models, optimizers, and replay buffer.
* `_preprocess_image(raw_image)`:  Preprocesses a raw image for VAE input.
* `encode_image(raw_image)`:  Encodes a raw image into a latent vector using the VAE.
* `reset_hidden_state()`:  Resets the RNN's hidden state.
* `choose_action(epsilon)`:  Selects an action based on exploration/exploitation.
* `store_experience(raw_image, action_key, action_array, reward, next_raw_image, done)`:  Stores a single timestep of experience in the replay buffer.
* `calculate_curiosity_reward(predicted_next_z, actual_next_z)`: Calculates the reward based on the difference between predicted and actual next latent states.
* `update_models(visualize, step)`:  Performs a training step for the VAE, RNN, and Self Model.

###   Models (`models.py`)

* `VAE`:  A Convolutional Variational Autoencoder for image encoding and reconstruction.
    * `encode(x)`:  Encodes image `x` into latent space.
    * `decode(z)`:  Decodes latent vector `z` into an image.
    * `forward(x)`:  Full VAE forward pass.
    * `loss_function(recon_x, x, mu, logvar, kld_weight)`:  Calculates the VAE loss.
* `RNNModel`:  An LSTM-based Recurrent Neural Network to model the environment's dynamics.
    * `forward(z, action, hidden_state)`:  Predicts the next latent state.
    * `init_hidden(batch_size, device)`:  Initializes the RNN's hidden state.
* `SelfModel`:  Predicts the curiosity reward.
    * `forward(z, h, action)`:  Predicts the reward.

###   MetricLogger (`metric_logger.py`)

* `__init__(log_dir, csv_filename, plot_filename_base)`:  Initializes the logger to save data.
* `log_metrics(...)`:  Logs simulation metrics to a CSV file.
* `plot_metrics(rolling_window)`:  Generates plots of the logged metrics.
* `close()`:  Closes any open resources.

###   VideoRecorder (`video_recorder.py`)

* `__init__(filename, resolution, fps)`:  Initializes the video writer.
* `write_frame(frame)`:  Writes a frame to the output video file.
* `annotate_frame(frame, step, curiosity_reward, self_loss)`:  Adds annotations (step number, reward, loss) to a frame.
* `close()`:  Releases video writer resources.

###   Utility (`utility.py`)

* `visualize_vae_reconstruction(originals, reconstructions, step, save_dir, num_examples)`:  Visualizes and saves VAE reconstructions.

###   DifferentialDriveAgent.py

* This file defines a different type of agent (a differential drive robot) and is not part of the primary simulation loop.

###   save* .py files

* These files appear to be older versions or duplicates and are not part of the current implementation. They should be removed.