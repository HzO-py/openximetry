#!/bin/bash -l 
#SBATCH -J spo2_tran_test_86hz
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=15-00:00:00
#SBATCH --mail-user=zqliang@ucdavis.edu
#SBATCH --mail-type=END,FAIL
#SBATCH -o output/bench-%x-%j.output
#SBATCH -e output/bench-%x-%j.output
#SBATCH --partition=gpu-homayoun
#SBATCH --gres=gpu:1
hostname
# srun python -u ml.py
srun python -u test_spo2.py
# srun python -u compare.py