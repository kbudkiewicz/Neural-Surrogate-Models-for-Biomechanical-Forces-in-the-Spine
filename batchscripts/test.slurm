#!/bin/bash

##SBATCH --qos=master
#SBATCH --partition=universe,asteroids
#SBATCH --nodelist=apollo   # DO NOT USE
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=spine_sim_test
#SBATCH --output=%x-%A.out
#SBATCH --error=%x-%A.err
##SBATCH --account=budk   # selene raises an error: Invalid account or account/partition combination specified
##SBATCH --mail-user=krystian.budkiewicz@tum.de
##SBATCH --mail-type=ALL

# Load python module
module load python/anaconda3
source activate /u/home/budk/.conda/envs/spine_simulation_312

# Run python
#python3 test.py
#python3 resample_test.py
python3 createCSV.py