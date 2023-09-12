#!/bin/bash
#SBATCH --job-name=gaurika_CIFAR
#SBATCH --mail-type=All
#SBATCH --mail-user=g.diwan@uqconnect.edu.au
#SBATCH --partition=vgpu
#SBATCH --gres=gpu:1
#SBATCH -o test_ou.txt
#SBATCH -e test_er.txt

conda activate conda /home/Student/s4824098/miniconda3

python ~/test.py

conda deactivate
