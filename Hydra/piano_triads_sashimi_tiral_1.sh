#!/bin/bash -l

# Send an email when the job begins and ends, and if the job is suspended or aborted
-m nafis

# Job name
#$ -N sashimi_train_piano_triads_1

# Memory allocation
#$ -pe omp 16

#Request 4 GPUs
#$ -l gpus=1

#Specify the minimum GPU compute capability

#GPU type
#$ -l gpu_type=V100

#Termination length
#$ -l h_rt=48:00:00

module load hydra/1.2.0
module load python3/3.8.10
module load pytorch/1.11.0

python train.py experiment=piano_triads model=sashimi  model.d_model=64 
