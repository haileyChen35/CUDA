#!/bin/bash
#SBATCH --job-name=haileyChen
#SBATCH --partition=GPU
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:TitanV:1
#SBATCH --mem=16G

OUTPUT="output.txt"



./nbody_par 1000 0.1 500 100 256 >> $OUTPUT
./nbody_par 10000 0.1 500 100 256 >> $OUTPUT
./nbody_par 100000 0.1 500 100 256 >> $OUTPUT