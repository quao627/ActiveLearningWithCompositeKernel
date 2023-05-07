#!/bin/bash
SBATCH -o ../../log/constant_space_load_balancing_nz.log-%j
SBATCH -c 24 
SBATCH -g volta:1
SBATCH --time=24:00:00          # total run time limit (HH:MM:SS)

# Loading the required module
source /etc/profile
module unload anaconda
module load anaconda/2022a

# -u flag helps with unbuffered printing
python deep_al_experiments.py --simulator motorcycle --seed 1 --active_learning_steps 40 --metamodel_name "motorcycle_variance_1_40" --selection_criteria variance