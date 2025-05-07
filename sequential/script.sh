#!/bin/bash
#SBATCH --job-name=haileyChen
#SBATCH --partition=GPU
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G

OUTPUT="output.txt"

echo "Test Case 0: planet dt=200 steps=5000 print_every=100 block_size=128" >> $OUTPUT
./nbody planet 200 5000 100 128 >> $OUTPUT
echo "" >> $OUTPUT