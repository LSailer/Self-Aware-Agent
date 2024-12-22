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


Kill TDW: pkill -f TDW
