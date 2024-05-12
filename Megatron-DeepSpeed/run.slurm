#!/bin/bash

#SBATCH -o job.%j.out
#SBATCH -J gpt_megatron
#SBATCH --partition=i64m1tga800u
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH -c 8
#SBATCH --gres=gpu:8
#SBATCH --nodelist=gpu1-43

source ~/.bashrc
conda activate reft

ranks_per_node=8
gpus_per_rank=$((8/$ranks_per_node))
ranks_total=$(($ranks_per_node*$SLURM_JOB_NUM_NODES))

echo $ranks_per_node $gpus_per_rank $ranks_total

export CHECKPOINT_PATH=./checkpoints
export VOCAB_FILE=/hpc2hdd/home/zli755/xueze/reft_ds/Megatron-DeepSpeed/examples_deepspeed/data_efficiency/gpt/pretrain/gpt2-vocab.json
export MERGE_FILE=/hpc2hdd/home/zli755/xueze/reft_ds/Megatron-DeepSpeed/examples_deepspeed/data_efficiency/gpt/pretrain/gpt2-merges.txt
export DATA_PATH=/hpc2hdd/home/zli755/xueze/reft_ds/md_preprocess/wikioutput_text_document

export GPT_ARGS="--tensor-model-parallel-size 2 \
          --pipeline-model-parallel-size 2 \
          --num-layers 24 \
          --hidden-size 2112 \
          --num-attention-heads 24 \
          --seq-length 2048 \
          --max-position-embeddings 2048 \
          --micro-batch-size 1 \
   --global-batch-size 40 \
          --lr 0.00015 \
          --train-iters 10000 \
          --lr-decay-iters 10 \
          --lr-decay-style cosine \
   --min-lr 6.0e-6 \
          --lr-decay-style cosine \
          --log-interval 1 \
          --eval-iters 3 \
          --eval-interval 3 \
          --save-interval 5 \
          --vocab-file $VOCAB_FILE \
          --merge-file $MERGE_FILE \
          --split 98,2,0 \
          --clip-grad 1.0 \
   --weight-decay 0.1 \
   --adam-beta1 0.9 \
   --adam-beta2 0.95 \
   --init-method-std 0.006 \
          --save $CHECKPOINT_PATH \
          --fp16"
          #--lr-warmup-fraction .01 \
 
#export CUDA_DEVICE_MAX_CONNECTIONS=1

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7


#time srun -u -n$ranks_total -c2 --ntasks-per-node=8 --gpus-per-node=8 --gpus-per-task=8 --gpu-bind=closest bash -c "
#time srun -u -n $ranks_total -c 7 --ntasks-per-node=8 --gpus-per-node=8 bash -c "lscpu "
time srun -u -n$ranks_total -c 8 --ntasks-per-node=8 --gpus-per-node=8 --gpus-per-task=1 bash -c "

export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID
export WORLD_SIZE=$SLURM_NTASKS
export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)
export MASTER_PORT=17717

python pretrain_zeyu.py \
       $GPT_ARGS \
       --data-path $DATA_PATH \
       --num-workers 0 \
       --deepspeed \
       --deepspeed_config ds_config.json \
       --no-gradient-accumulation-fusion
       "
#torchrun --standalone --nnodes=1 --nproc-per-node=8 pretrain_gpt.py  --data-path $DATA_PATH --deepspeed --deepspeed_config ds_config.json $GPT_ARGS
