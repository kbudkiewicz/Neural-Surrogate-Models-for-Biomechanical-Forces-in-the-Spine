#!/bin/bash

##SBATCH --qos=master-queuesave
#SBATCH --partition=universe
##SBATCH --nodelist=apollo,leto # DONT USE. does not work on other nodes. Raises CUDA error
#SBATCH --time=0-01:00:00
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1,gpumem:50G
#SBATCH --mem=64G
#SBATCH --job-name=evaluate
#SBATCH --output=/u/home/budk/spine_simulation/batchreports/%x-%A.out
#SBATCH --error=/u/home/budk/spine_simulation/batchreports/%x-%A.err

# To learn more about what hides behind % check: https://slurm.schedmd.com/sbatch.html#SECTION_FILENAME-PATTERN

module load python/anaconda3
source activate spine_simulation_312

python3 evaluate_model.py --csv-file data/nako_from_slurm.csv -w data/eval_resnet3dregressor/nakoimages/3_layers/*.pth -s coord_0 coord_1 coord_2 coord_3 coord_4 coord_5