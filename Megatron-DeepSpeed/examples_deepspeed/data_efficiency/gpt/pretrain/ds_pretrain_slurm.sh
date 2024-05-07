#!/bin/bash
#SBATCH --partition=i64m1tga800u
#SBATCH --job-name=slurm_test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --qos=i64m1tga8+
#SBATCH --nodelist=gpu1-19

# source ~/.bashrc
# conda activate reft
# srun --pty bash ./ds_pretrain_gpt_1.3B_dense_base_script.sh
srun --pty --partition=i64m1tga800u --job-name=slurm_test --nodes=1 \
--ntasks-per-node=4 --cpus-per-task=8 --gres=gpu:4 --qos=i64m1tga8+ \
--nodelist=gpu1-30 bash ds_pretrain_slurm_run.sh
