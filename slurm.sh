#!/bin/bash
#SBATCH --job-name=test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH
--mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
