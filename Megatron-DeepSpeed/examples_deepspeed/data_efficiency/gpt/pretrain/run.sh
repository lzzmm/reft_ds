#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH --partition=i64m1tga800u
#SBATCH -J myFirstMPIJob
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --nodelist=gpu1-[5-6]

export MASTER_PORT=17717
source ~/.bashrc
conda activate reft
bash /hpc2hdd/home/zli755/xueze/reft_ds/Megatron-DeepSpeed/examples_deepspeed/data_efficiency/gpt/pretrain/ds_pretrain_gpt_1.3B_dense_run.sh
