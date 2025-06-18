#!/bin/bash
#SBATCH --job-name=RunSimulation
#SBATCH --output=logs/Logs%j.out
#SBATCH --error=logs/Logs%j.err
#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=127G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_h100

echo "Job started ..." 


echo "Using Python: $(which python) — version: $(python --version)"
#python src/main.py --config configs.multi_agent_config
python src/main.py --config configs.multi_agent_config




echo "Job completed."
# Slurm Commands
# module load devel/python/3.10.5
# sbatch -> Run script
# squeue -> show the list
# scancel <job_id> -> cancel job
# sinfo_t_idle
# squeue --start


# conda activate Self_Aware_Agent


