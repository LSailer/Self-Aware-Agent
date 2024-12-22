# Self-Aware-Agent

## Install Conda
conda create -n tdw_env python=3.8


## Acknowledgements
This project is based on [learning_to_play](https://github.com/neuroailab/learning_to_play.git) by neuroailab.

## Project Structure
- `models.py`: Contains the `WorldModel` and `SelfModel` neural network definitions.
- `environment.py`: Handles the PyBullet environment setup, including walls, objects, and physics adjustments.
- `simulation.py`: Runs the main simulation loop, applying forces and training the models.
- `env.sh`: Script for setting up the environment.

## Prerequisites
### Create and Activate the Environment
1. Create the Conda environment:
   ```bash
   conda create -n self_aware_agent python=3.10
   ```

2. Activate the environment:
   ```bash
   conda activate self_aware_agent
   ```

### Install Dependencies
Run the following commands to install the required packages:

1. Install PyBullet:
   ```bash
   conda install conda-forge::pybullet -y
   ```

2. Install Matplotlib:
   ```bash
   conda install conda-forge::matplotlib -y
   ```

3. Install PyTorch and related libraries:
   ```bash
   conda install pytorch::pytorch torchvision torchaudio -c pytorch -y
   ```

### Additional Notes
- Ensure the environment is activated before running any scripts.
  ```bash
  conda activate self_aware_agent
  ```


Kill TDW: pkill -f TDW
