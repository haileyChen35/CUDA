#!/bin/bash
#SBATCH --job-name=haileyChen
#SBATCH --partition=GPU
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G

OUTPUT="output.txt"

echo "Nbody" >> $OUTPUT

echo "Test Case 1:" >> $OUTPUT
./nbody 1000 5 500 100 269 >> $OUTPUT
echo "" >> $OUTPUT

echo "Test Case 2:" >> $OUTPUT
./nbody 10000 10 500 100 269 >> $OUTPUT
echo "" >> $OUTPUT

echo "Test Case 3:" >> $OUTPUT
./nbody 100000 15 500 100 269 >> $OUTPUT
echo "" >> $OUTPUT

echo "Nbody_Par" >> $OUTPUT

echo "Test Case 1:" >> $OUTPUT
./nbody_par 1000 5 500 100 269 >> $OUTPUT
echo "" >> $OUTPUT

echo "Test Case 2:" >> $OUTPUT
./nbody_par 10000 10 500 100 269 >> $OUTPUT
echo "" >> $OUTPUT

echo "Test Case 3:" >> $OUTPUT
./nbody_par 100000 15 500 100 269 >> $OUTPUT
echo "" >> $OUTPUT