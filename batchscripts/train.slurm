#!/bin/bash

#SBATCH --qos=master-queuesave
#SBATCH --partition=universe
##SBATCH --nodelist=apollo,leto,hercules # DONT USE. does not work on other nodes. Raises CUDA error
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1,gpumem:40G
#SBATCH --mem=64G
#SBATCH --job-name=spine_sim_train
#SBATCH --output=/u/home/budk/spine_simulation/batchreports/%x-%A.out
#SBATCH --error=/u/home/budk/spine_simulation/batchreports/%x-%A.err

# To learn more about what hides behind % check: https://slurm.schedmd.com/sbatch.html#SECTION_FILENAME-PATTERN

# Load python module
module load python/anaconda3
source activate /u/home/budk/.conda/envs/spine_simulation_312

# Run python
torchrun --rdzv_endpoint=localhost:29000 simple_test.py
