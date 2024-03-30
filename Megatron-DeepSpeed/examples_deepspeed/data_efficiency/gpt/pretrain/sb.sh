#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH --partition=i64m1tga800u
#SBATCH -J deepspeed_pretrain_gpt
#SBATCH -N 1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --qos=high
#SBATCH --nodelist=gpu1-32

source ~/.bashrc
conda activate reft
bash /hpc2hdd/home/zli755/xueze/reft_ds/Megatron-DeepSpeed/examples_deepspeed/data_efficiency/gpt/pretrain/ds_pretrain_gpt_1.3B_dense_base_script.sh 