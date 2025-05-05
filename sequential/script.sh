#!/bin/bash
#SBATCH --job-name=haileyChen
#SBATCH --partition=GPU
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G

./nbody 100000 0.01 50 101 128