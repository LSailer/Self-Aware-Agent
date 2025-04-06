##   Step 1: Single-Agent Setup

###   Environment & Core Functionality

   * [X] Set up the PyBullet environment with a single agent.
        * [X] Generate a 10x10 room with border walls.
        * [X] Add a sphere agent with a camera.
        * [X] Implement basic agent movement (forward/backward/left/right and rotation).
        * [X] Add a red cube and a blue cylinder as interactable objects.
   * [X] Implement the Schmidhuber VAE/RNN-based world model in Python.
        * [X] Code the VAE (Vision Model).
        * [X] Code the MDN-RNN (Memory RNN).
        * [X] Code the Self Model.
        * [X] Integrate the VAE, RNN, and Self Model.
   * [X] Implement the agent with action selection based on Self Model predictions.
   * [X] Implement curiosity-driven reward calculation.
   * [X] Implement logging of relevant metrics (position, orientation, losses, interaction frequency).
   * [X] Implement visualization of VAE reconstruction.
   * [X] Implement video recording with metric annotations.
   * [X] Test the world model by training the agent and observing its behavior.
   
###   Policy Search Methods Comparison

   * [ ] Implement alternative action selection/policy search methods.
        * [ ]  (e.g., Softmax, UCB, or others)
   * [ ] Compare the performance of different action selection methods.
        * [ ]  (Evaluate metrics like exploration efficiency and task completion.)

###   Testing & Deployment

   * [ ] Generate unit tests for key modules (VAE, RNN, SelfModel).
        * [X] Test encoding and decoding of VAE.
        * [ ] Test RNN state transitions.
        * [ ] Test SelfModel reward prediction.
   * [ ] Create a Dockerfile for containerizing the application.
   * [ ] Define steps to deploy the application on the UniCluster.
        * [ ] (Specify resource requests, job submission commands, etc.)

###   Hyperparameter Optimization (wandb sweep)

   * [ ] Integrate PyTorch Lightning for cleaner training.
   * [ ] Define a `wandb` sweep configuration file (YAML).
        * [ ] Specify the search space for hyperparameters (learning rates, network sizes, etc.).
        * [ ] Define the metric to optimize (e.g., average reward).
        * [ ] Choose a search strategy (`grid`, `random`, `bayes`).
   * [ ] Modify the training script to use `wandb.init()` and `wandb.log()` for experiment tracking.
   * [ ] Implement the `wandb sweep` command to launch hyperparameter search.
   * [ ] Log hyperparameter configurations and corresponding evaluation metrics using `wandb`.

###   Documentation & Maintenance

   * [X] Document the code and setup.
   * [X] Update README.md with setup and usage instructions.
   * [X] Update PLANNING.md and TASK.md to reflect progress and new tasks.

###   (Next) Multi-Agent Extension

   * [ ] Add a second agent.
   * [ ] Define interaction rules (cooperation/competition).
   * [ ] Adapt reward structure for multi-agent.
   * [ ] Analyze emergent behaviors.