#!/bin/bash

dir=`pwd`
###############################################################################
### Main configs
## GPT-3 models use 2K sequence length/context window
seq_len=2048

## The "GPT-3 XXX" below are configs from GPT-3 paper
## https://arxiv.org/abs/2005.14165, choose based on
## your desired model size or build your own configs

## init_std is standard deviation for weight initialization. Usually larger
## model needs lower std. We used a heuristic equation of sqrt(1/3/hidden_size)
## from the MT-NLG 530B work (https://arxiv.org/pdf/2201.11990.pdf)

## We changed min_lr to a lower number (1.0e-6), which we found is able to
## provide better zero-shot eval results. 

model_size_config=${1:-0}
model_size_config=${model_size_config#*=}
echo "model_size_config: $model_size_config"

if [ $model_size_config -eq 0 ]; then
# Small 125M
    model_size=0.125
    num_layers=12
    hidden_size=768
    num_attn_heads=12
    global_batch_size=16
    lr=6.0e-4
    min_lr=1.0e-6
    init_std=0.02
elif [ $model_size_config -eq 1 ]; then
# Medium 350M
    model_size=0.35
    num_layers=24
    hidden_size=1024
    num_attn_heads=16
    global_batch_size=32
    lr=3.0e-4
    min_lr=1.0e-6
    init_std=0.018
elif [ $model_size_config -eq 2 ]; then
# Large 760M
    model_size=0.76
    num_layers=24
    hidden_size=1536
    num_attn_heads=16
    global_batch_size=16
    lr=2.5e-4
    min_lr=1.0e-6
    init_std=0.015
elif [ $model_size_config -eq 3 ]; then
# XL 1.3B
    model_size=1.3
    num_layers=24
    hidden_size=2048
    num_attn_heads=16
    global_batch_size=32
    lr=2.0e-4
    min_lr=1.0e-6
    init_std=0.013
elif [ $model_size_config -eq 4 ]; then
# 2.7B
    model_size=2.7
    num_layers=32
    hidden_size=2560
    num_attn_heads=32
    global_batch_size=16
    lr=1.6e-4
    min_lr=1.0e-6
    init_std=0.011
elif [ $model_size_config -eq 5 ]; then
# 6.7B
    model_size=6.7
    num_layers=32
    hidden_size=4096
    num_attn_heads=32
    global_batch_size=8
    lr=1.2e-4
    min_lr=1.0e-6
    init_std=0.009
elif [ $model_size_config -eq 6 ]; then
# 13B
    model_size=13
    num_layers=40
    hidden_size=5120
    num_attn_heads=40
    global_batch_size=8
    lr=1.0e-4
    min_lr=1.0e-6
    init_std=0.008
elif [ $model_size_config -eq 7 ]; then
# 175B
    model_size=175
    num_layers=96
    hidden_size=12288
    num_attn_heads=96
    global_batch_size=8
    lr=0.6e-4
    min_lr=1.0e-6
    init_std=0.005
fi

# # GPT-3 Small 125M
# model_size=0.125
# num_layers=12
# hidden_size=768
# num_attn_heads=12
# global_batch_size=32
# lr=6.0e-4
# min_lr=1.0e-6
# init_std=0.02

## GPT-3 Medium 350M
# model_size=0.35
# num_layers=24
# hidden_size=1024
# num_attn_heads=16
# global_batch_size=16
# lr=3.0e-4
# min_lr=1.0e-6
# init_std=0.018

## GPT-3 Large 760M
# model_size=0.76
# num_layers=24
# hidden_size=1536
# num_attn_heads=16
# global_batch_size=16
# lr=2.5e-4
# min_lr=1.0e-6
# init_std=0.015

## GPT-3 XL 1.3B
# model_size=1.3
# num_layers=24
# hidden_size=2048
# num_attn_heads=16
# global_batch_size=32
# lr=2.0e-4
# min_lr=1.0e-6
# init_std=0.013

## GPT-3 2.7B
# model_size=2.7
# num_layers=32
# hidden_size=2560
# num_attn_heads=32
# global_batch_size=32
# lr=1.6e-4
# min_lr=1.0e-6
# init_std=0.011

## GPT-3 6.7B
# model_size=6.7
# num_layers=32
# hidden_size=4096
# num_attn_heads=32
# global_batch_size=32
# lr=1.2e-4
# min_lr=1.0e-6
# init_std=0.009

## GPT-3 13B
# model_size=13
# num_layers=40
# hidden_size=5120
# num_attn_heads=40
# global_batch_size=8
# lr=1.0e-4
# min_lr=1.0e-6
# init_std=0.008

## GPT-3 175B
# model_size=175
# num_layers=96
# hidden_size=12288
# num_attn_heads=96
# global_batch_size=1536
# lr=0.6e-4
# min_lr=1.0e-6
# init_std=0.005
###############################################################################
### Training duration configs
## The main termination condition, original GPT-3 paper trains for 300B tokens.
# train_tokens_in_billion=300
train_tokens_in_billion=150
train_tokens=$((${train_tokens_in_billion} * 1000000000))
# train_tokens=100000

## train_samples is another termination condition and also affect the number of 
## data samples to be indexed. Since we want to reach the train_tokens
## above, and data efficiency techniques may change num tokens in some samples,
## so we just set this config large enough to make sure we have enough
## processed data and don't terminate by train_samples.
# train_samples=$(( 300 * 1000000000 * 2 / ${seq_len} ))
train_iters=30

## Another wall-clock time termination condition in minutes. Set it large
## enough to avoid undesired early termination.
exit_duration=30000000
###############################################################################
### lr configs
## lr warmup and decay duration.
## Original GPT-3 paper uses 375M warmup tokens and 260B cosine decay tokens.
## Here we increase the warmup tokens to 3B since when batch size warmup is not
## used, there are more tokens per step. Thus we need to increase warmup tokens
## to make sure there are enough warmup steps, which is important for training
## stability.
lr_warmup_tokens_in_million=3000
lr_warmup_tokens=$((${lr_warmup_tokens_in_million} * 1000000))
## Here we changed the LR decay tokens to align with total train tokens, since
## related works (e.g., https://arxiv.org/abs/2203.15556) find that setting the
## learning rate schedule to match the number of training tokens results in the
## best final model quality 
lr_decay_tokens_in_billion=${train_tokens_in_billion}
lr_decay_tokens=$((${lr_decay_tokens_in_billion} * 1000000000))
lr_decay_style="cosine"
###############################################################################
### Parallelism configsf
## Model parallelism, 1 is no MP
mp_size=1

## Pipeline parallelism. To disable PP, set pp_size to 1 and no_pp to true.
## Note that currently both curriculum learning and random-LTD are NOT
## compatible with pipeline parallelism.
pp_size=8
# no_pp="true"
no_pp="false"

## ZeRO-based data parallelism, stage=0 will disable ZeRO
zero_stage=0

## Total number of GPUs. ds_ssh is from DeepSpeed library.
num_node=1
num_gpus=8
num_gpus_pernode=$(( ${num_gpus} / ${num_node} ))
## Data parallel size.
# dp_size=$(( ${num_gpus} / ${pp_size} / ${mp_size} ))
dp_size=1
gradient_accumulation_steps=8
## Micro batch size per GPU
## Make sure that batch_size <= global_batch_size*pp_size*mp_size/num_gpus
## Reduce it manually if GPU OOM
batch_size=$(( ${global_batch_size} / ${dp_size} / ${gradient_accumulation_steps} ))
echo "batch_size: $batch_size"
###############################################################################
### Random layerwise token dropping (random-LTD) configs
## random-LTD's main switch. "false" means disabled. "true" means enabled.
ltd_enabled='false'
## How much dropping ratio to start with. The value denotes the seqlen after
## dropping.
ltd_start=2048
## How many steps for random-LTD to gradually reduce dropping ratio to zero.
ltd_step=1

# ltd_enabled="true"
# ltd_start=128
# ltd_step=200000
###############################################################################
### Curriculum learning (CL) configs
## CL's main switch. "false" means disabled. "true" means enabled.
cl_enabled='false'
## Number of CL metrics to use.
cl_num_metric=1

## Name of difficulty metric
cl_1st_metric='dummy'
## Path to the data indexes for this difficulty metric. Samples on ith row of
## index_to_sample have the difficulty value equals to ith row of
## index_to_metric.
cl_1st_index_to_sample_path='dummy'
cl_1st_index_to_metric_path='dummy'
## During training, whether increase difficulty by value- or percentile-based.
cl_1st_difficulty_type='value'
## "single_cluster" means no clustering required and probably CL is achieved by
## data postprocessing. "schedule_based" means will cluster data based on the
## difficulty schedule (pacing function) below.
cl_1st_clustering_type='single_cluster'
## Start difficulty
cl_1st_min=2048
## End difficulty
cl_1st_max=2048
## Total step to reach end difficulty
cl_1st_total_step=1
## When changing difficulty, always make sure it's a multiple of the
## difficulty_step below.
cl_1st_difficulty_step=1
## Root degree of the schedule (pacing function).
cl_1st_root=1

cl_2nd_metric='dummy'
cl_2nd_index_to_sample_path='dummy'
cl_2nd_index_to_metric_path='dummy'
cl_2nd_difficulty_type='value'
cl_2nd_clustering_type='single_cluster'
cl_2nd_min=2048
cl_2nd_max=2048
cl_2nd_total_step=1
cl_2nd_difficulty_step=1
cl_2nd_root=1

# cl_enabled="true"
# cl_num_metric=2
# cl_1st_metric="voc"
# ## The *_index_to_sample_percentile_merged is a concatenated index for perf
# ## improvement, but it only works when you set difficulty_type="percentile" in
# ## ds_config. If you use difficulty_type="value", you need to change this to
# ## *_index_to_sample
# cl_1st_index_to_sample_path="/blob/users/conglli/data/analysis_pile_gpt_1epoch/vocab_rarity/vocab_rarity_index_to_sample_percentile_merged"
# # cl_1st_index_to_sample_path="/blob/users/conglli/data/analysis_pile_gpt_1epoch/vocab_rarity/vocab_rarity_index_to_sample"
# cl_1st_index_to_metric_path="/blob/users/conglli/data/analysis_pile_gpt_1epoch/vocab_rarity/vocab_rarity_index_to_metric"
# cl_1st_difficulty_type="percentile"
# cl_1st_clustering_type="schedule_based"
# cl_1st_min=1
# cl_1st_max=100
# cl_1st_total_step=110000
# cl_1st_difficulty_step=1
# cl_1st_root=2

# cl_2nd_metric="seqlen_truncate"
# cl_2nd_index_to_sample_path="dummy"
# cl_2nd_index_to_metric_path="dummy"
# cl_2nd_difficulty_type="value"
# cl_2nd_clustering_type="single_cluster"
# cl_2nd_min=80
# cl_2nd_max=2048
# cl_2nd_total_step=110000
# cl_2nd_difficulty_step=8
# cl_2nd_root=1
###############################################################################
### Misc configs
log_interval=1
# eval_iters=10
eval_iters=0
eval_interval=100
# num_save controls how frequent to save checkpoint. num_save=20 means that a
# checkpoint will be saved every 5% of training. For longer training you would
# want larger num_save to save more frequently, and vice versa.
num_save=100
estimated_train_iter=$((${train_tokens} / ${seq_len} / ${global_batch_size}))
# save_interval=$((${estimated_train_iter} / ${num_save}))
save_interval=1

## Activation checkpointing saves GPU memory, but reduces training speed
# activation_checkpoint="true"
activation_checkpoint="false"

## Whether or not log optimizer states (norms, max abs values) to tensorboard.
## This is not required for training and might save GPU memory when turned off.
log_optimizer_state="true"
###############################################################################
### Output and data configs
current_time=$(date "+%Y.%m.%d_%H.%M")
host="${HOSTNAME}"
seed=1234
num_workers=0

## Public the Pile dataset, can be downloaded at
## https://mystic.the-eye.eu/public/AI/pile_neox/ Change data_home to where you
## store the pile_text_document.bin and pile_text_document.idx.
# data_home="/vc_data_blob/users/conglli/the_pile_public_merged_nopreprocessing"
# if [[ "$host" == *"webxt"* ]]; then
#     data_home="/blob/data/the_pile_public_merged_nopreprocessing"
# fi
# data_path="${data_home}/pile_text_document"
data_home="${dir}/../../../../../md_preprocess"
if [[ "$host" == *"webxt"* ]]; then
    data_home="${dir}/../../../../../md_preprocess"
fi
data_path="${data_home}/wikioutput_text_document"
## *_idx_path force Megatron to use a specific data index file generated when
## we analyze data. This is needed because our index for curriculum learning
## difficulty metric is based on this data index.
doc_idx_path="${data_home}/pile_text_document_train_indexmap_exact1ep_2048sl_1234s_doc_idx.npy"
sample_idx_path="${data_home}/pile_text_document_train_indexmap_exact1ep_2048sl_1234s_sample_idx.npy"
shuffle_idx_path="${data_home}/pile_text_document_train_indexmap_exact1ep_2048sl_1234s_shuffle_idx.npy"

vocab_path="gpt2-vocab.json"
if [ ! -f "$vocab_path" ]; then
    wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json
fi
merge_path="gpt2-merges.txt"
if [ ! -f "$merge_path" ]; then
    wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt
fi

prescale_grad="true"
jobname="gpt_${model_size}B_tok${train_tokens_in_billion}B"
jobname="${jobname}_lr${lr}_min${min_lr}_w${lr_warmup_tokens_in_million}M_d${lr_decay_tokens_in_billion}B_${lr_decay_style}"
jobname="${jobname}_gbs${global_batch_size}_mbs${batch_size}_g${num_gpus}"
if [[ $zero_stage -gt 0 ]]; then
    jobname="${jobname}_z${zero_stage}"
    prescale_grad="false"
fi
if [[ $mp_size -gt 1 ]]; then
    jobname="${jobname}_mp${mp_size}"
fi
if [ "${no_pp}" = "false" ]; then
    jobname="${jobname}_pp${pp_size}"
fi
jobname="${jobname}_seed${seed}"
if [ "${ltd_enabled}" = "true" ]; then
    jobname="${jobname}_ltd_${ltd_start}_${ltd_step}"
fi
if [ "${cl_enabled}" = "true" ]; then
    jobname="${jobname}_cl_${cl_1st_metric}_${cl_1st_min}_${cl_1st_max}_${cl_1st_total_step}_${cl_1st_root}"
    if [[ $cl_num_metric -gt 1 ]]; then
        jobname="${jobname}_${cl_2nd_metric}_${cl_2nd_min}_${cl_2nd_max}_${cl_2nd_total_step}_${cl_2nd_root}"
    fi
fi

username=$(whoami)
checkpoint_new_thread=${2:-"true"}
checkpoint_new_thread=${checkpoint_new_thread#*=}
echo "checkpoint_new_thread: $checkpoint_new_thread"
checkpoint_new_stream=${3:-"true"}
checkpoint_new_stream=${checkpoint_new_stream#*=}
echo "checkpoint_new_stream: $checkpoint_new_stream"
enable_parity=${4:-"true"}
enable_parity=${enable_parity#*=}
echo "enable_parity: $enable_parity"
enable_pin_memory=${5:-"true"}
enable_pin_memory=${enable_pin_memory#*=}
echo "enable_pin_memory: $enable_pin_memory"
enable_sharding=${6:-"true"}
enable_sharding=${enable_sharding#*=}
echo "enable_sharding: $enable_sharding"
enable_profile=${7:-"true"}
enable_profile=${enable_profile#*=}
echo "enable_profile: $enable_profile"
enable_save=${8:-"false"}
enable_save=${enable_save#*=}
echo "enable_save: $enable_save"
save_location=${9:-"nfs"}
save_location=${save_location#*=}
echo "save_location: $save_location"
enable_snapshot=${10:-"true"}
enable_snapshot=${enable_snapshot#*=}
echo "enable_snapshot: $enable_snapshot"
prealloc=${11:-"true"}
prealloc=${prealloc#*=}
echo "prealloc: $prealloc"
pure_torch_save=${12:-"false"}
pure_torch_save=${pure_torch_save#*=}
echo "pure_torch_save: $pure_torch_save"
get_state_dict_shape=${13:-"false"}
get_state_dict_shape=${get_state_dict_shape#*=}
echo "get_state_dict_shape: $get_state_dict_shape"
save_checkpoint_in_bubble=${14:-"true"}
save_checkpoint_in_bubble=${save_checkpoint_in_bubble#*=}
echo "save_checkpoint_in_bubble: $save_checkpoint_in_bubble"
fail=${15:-"false"}
fail=${fail#*=}
echo "fail: $fail"
load=${16:-"false"}
load=${load#*=}
echo "load: $load"

failed_ranks=""
load_recovery=""
load_path=""
# output_home="/blob/users/${username}/project/data_efficient_gpt"
# output_home="/hpc2hdd/home/zli755/xueze/reft_ds/Megatron-DeepSpeed/examples_deepspeed/data_efficiency/gpt/output"
output_home="${dir}/../output"
log_path="${output_home}/log/${current_time}"
mkdir -p ${log_path}
# checkpoint_path="${output_home}/checkpoint/${jobname}"
# checkpoint_path="/hpc2hdd/home/zli755/xueze/reft_ds/Megatron-DeepSpeed/examples_deepspeed/data_efficiency/gpt/save"
if [ "${save_location}" == "tmpfs" ]; then
    checkpoint_path="/dev/shm/reft/save"
    recovery_path="/dev/shm/reft/recovery"
else
    checkpoint_path="${dir}/../save"
    recovery_path="${dir}/../recovery"
fi
# checkpoint_path="${dir}/../save"
## Microsoft internal constraint: because tensorboard is logged by last rank,
## it's better to put the path in NFS instead of Blob.
# tensorboard_dir="/hpc2hdd/home/zli755/xueze/reft_ds/Megatron-DeepSpeed/examples_deepspeed/data_efficiency/gpt/tensorboard"
tensorboard_dir="${dir}/../tensorboard"
tensorboard_path=""
# If log_path is not "", then mkdir
if [ "${log_path}" != "" ]; then
    mkdir -p ${log_path}
fi
if [ "${checkpoint_path}" != "" ]; then
    mkdir -p ${checkpoint_path}
fi
if [ "${recovery_path}" != "" ]; then
    mkdir -p ${recovery_path}
fi
if [ "${tensorboard_path}" != "" ]; then
    mkdir -p ${tensorboard_path}
fi
if [ "${cl_enabled}" = "true" ]; then
    data_cluster_path="${output_home}/data_cluster/${jobname}"
    mkdir -p ${data_cluster_path}
fi



###############################################################################
data_options=" \
    --vocab-file ${vocab_path} \
    --merge-file ${merge_path} \
    --data-path ${data_path} \
    --data-impl mmap"

## If CL is used, make sure to set "--split" the same as what you used during
## offline data analysis&indexing.
megatron_options=" \
    --override-opt_param-scheduler \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --tensor-model-parallel-size ${mp_size} \
    --init-method-std ${init_std} \
    --lr-decay-tokens ${lr_decay_tokens} \
    --lr-warmup-tokens ${lr_warmup_tokens} \
    --micro-batch-size ${batch_size} \
    --exit-duration-in-mins ${exit_duration} \
    --global-batch-size ${global_batch_size} \
    --num-layers ${num_layers} \
    --hidden-size ${hidden_size} \
    --num-attention-heads ${num_attn_heads} \
    --seq-length ${seq_len} \
    --max-position-embeddings ${seq_len} \
    --train-tokens ${train_tokens} \
    --train-iters ${train_iters} \
    --lr ${lr} \
    --min-lr ${min_lr} \
    --lr-decay-style ${lr_decay_style} \
    --split 949,50,1 \
    --log-interval ${log_interval} \
    --eval-interval ${eval_interval} \
    --eval-iters ${eval_iters} \
    --save-interval ${save_interval} \
    --weight-decay 0.1 \
    --clip-grad 1.0 \
    --hysteresis 2 \
    --num-workers ${num_workers} \
    --fp16 \
    --seed ${seed} \
    --tensorboard-queue-size 1 \
    --log-timers-to-tensorboard \
    --log-batch-size-to-tensorboard \
    --log-validation-ppl-to-tensorboard" 

if [ "${checkpoint_new_thread}" = "true" ]; then
    megatron_options="${megatron_options} \
        --checkpoint-new-thread"
fi

if [ "${checkpoint_new_stream}" = "true" ]; then
    megatron_options="${megatron_options} \
        --checkpoint-new-stream"
fi

if [ "${enable_parity}" = "true" ]; then
    megatron_options="${megatron_options} \
        --enable-parity"
fi

if [ "${enable_pin_memory}" = "true" ]; then
    megatron_options="${megatron_options} \
        --enable-pin-memory"
fi

if [ "${enable_sharding}" = "true" ]; then
    megatron_options="${megatron_options} \
        --enable-sharding"
fi

if [ "${enable_profile}" = "true" ]; then
    megatron_options="${megatron_options} \
        --enable-profile"
fi

if [ "${enable_save}" = "true" ]; then
    megatron_options="${megatron_options} \
        --enable-save"
    megatron_options="${megatron_options} \
        --save-location ${save_location}"
fi

if [ "${enable_snapshot}" = "true" ]; then
    megatron_options="${megatron_options} \
        --enable-snapshot"
fi

if [ "${prealloc}" = "true" ]; then
    megatron_options="${megatron_options} \
        --prealloc"
fi

if [ "${pure_torch_save}" = "true" ]; then
    megatron_options="${megatron_options} \
        --pure-torch-save"
fi

if [ "${get_state_dict_shape}" = "true" ]; then
    megatron_options="${megatron_options} \
        --get-state-dict-shape"
fi

if [ "${save_checkpoint_in_bubble}" = "true" ]; then
    megatron_options="${megatron_options} \
        --save-checkpoint-in-bubble"
fi

if [[ -n "${checkpoint_path}" ]]; then
    megatron_options+=" --save ${checkpoint_path}"
fi

if [ "${load}" = "true" ]; then
    megatron_options="${megatron_options} \
        --load ${load_path}"
    megatron_options="${megatron_options} \
        --load-tag global_step2"
    if [ "${fail}" = "true" ]; then
        megatron_options="${megatron_options} \
        --load-recovery ${load_recovery}"
        megatron_options="${megatron_options} \
        --failed-ranks ${failed_ranks}"
    fi
fi

if [[ -n "${recovery_path}" ]]; then
    megatron_options+=" --recovery-dir ${recovery_path}"
fi

if [[ -n "${tensorboard_path}" ]]; then
    megatron_options+=" --tensorboard-dir ${tensorboard_path}"
fi

if [ "${activation_checkpoint}" = "true" ]; then
megatron_options="${megatron_options} \
    --checkpoint-activations"
fi

if [ "${log_optimizer_state}" = "true" ]; then
megatron_options="${megatron_options} \
    --log-optimizer-states-to-tensorboard"
fi

if [ "${ltd_enabled}" = "true" ]; then
megatron_options="${megatron_options} \
    --random-ltd"
fi

if [ "${cl_enabled}" = "true" ]; then
megatron_options="${megatron_options} \
    --train-doc-idx-path ${doc_idx_path} \
    --train-sample-idx-path ${sample_idx_path} \
    --train-shuffle-idx-path ${shuffle_idx_path} \
    --data-efficiency-curriculum-learning"
fi

log_args="model_size_config ${mode_size_config}\nnum_node_${num_node}\nnum_gpus_${num_gpus}\nglobal_batch_size_${global_batch_size}\nbatch_size_${batch_size}\ncheckpoint_new_thread_${checkpoint_new_thread}\ncheckpoint_new_stream_${checkpoint_new_stream}\nenable_parity_${enable_parity}\nenable_pin_memory_${enable_pin_memory}\nenable_sharding_${enable_sharding}\nenable_profile_${enable_profile}\nenable_save_${enable_save}\nprealloc_${prealloc}\nenable_snapshot_${enable_snapshot}\nget_state_dict_shape_${get_state_dict_shape}\n"


if [ "${enable_save}" = "true" ]; then
    log_args="${log_args}\nsave_location_${save_location}"
fi

if [ "${pure_torch_save}" = "true" ]; then
    log_args="${log_args}\npure_torch_save_${pure_torch_save}"
fi

log_args="${log_args}save_checkpoint_in_bubble_${save_checkpoint_in_bubble}\nfail_${fail}\nfailed_ranks_${failed_ranks}\nload_${load}\nload_path_${load_path}\nrecovery_path_${recovery_path}\n"


echo -e ${log_args} > ${log_path}/log_args.txt

config_json="ds_config_gbs${global_batch_size}_mbs${batch_size}_log${log_interval}_zero${zero_stage}_seed${seed}"
if [ "${ltd_enabled}" = "true" ]; then
    config_json="${config_json}_ltd_${ltd_start}_${ltd_step}"
fi
if [ "${cl_enabled}" = "true" ]; then
    config_json="${config_json}_cl_${cl_1st_metric}_${cl_1st_min}_${cl_1st_max}_${cl_1st_total_step}_${cl_1st_root}"
    if [[ $cl_num_metric -gt 1 ]]; then
        config_json="${config_json}_${cl_2nd_metric}_${cl_2nd_min}_${cl_2nd_max}_${cl_2nd_total_step}_${cl_2nd_root}"
    fi
fi
config_json="${config_json}.json"
if [[ $cl_num_metric -gt 1 ]]; then
template_json="ds_config_gpt_2clmetrics_TEMPLATE.json"
sed "s/GBSIZE/${global_batch_size}/" ${template_json} \
    | sed "s/MBSIZE/${batch_size}/" \
    | sed "s/LOG_INTERVAL/${log_interval}/" \
    | sed "s/ZERO_STAGE/${zero_stage}/" \
    | sed "s/PRESCALE_GRAD/${prescale_grad}/" \
    | sed "s/DATA_EFFICIENCY_SEED/${seed}/" \
    | sed "s/LTD_ENABLED/${ltd_enabled}/" \
    | sed "s/LTD_MIN/${ltd_start}/" \
    | sed "s/LTD_MAX/${seq_len}/" \
    | sed "s/LTD_STEP/${ltd_step}/" \
    | sed "s/CL_ENABLED/${cl_enabled}/" \
    | sed "s/DATA_SAMPLING_NUM_WORKERS/${num_workers}/" \
    | sed "s#CL_CLUSTER_PATH#${data_cluster_path}#" \
    | sed "s#CL_1st_METRIC_NAME#${cl_1st_metric}#" \
    | sed "s#CL_1st_SAMPLE_PATH#${cl_1st_index_to_sample_path}#" \
    | sed "s#CL_1st_METRIC_PATH#${cl_1st_index_to_metric_path}#" \
    | sed "s#CL_1st_DIFF_TYPE#${cl_1st_difficulty_type}#" \
    | sed "s#CL_1st_CLUSTER_TYPE#${cl_1st_clustering_type}#" \
    | sed "s/CL_1st_MIN/${cl_1st_min}/" \
    | sed "s/CL_1st_MAX/${cl_1st_max}/" \
    | sed "s/CL_1st_TOTAL_STEP/${cl_1st_total_step}/" \
    | sed "s/CL_1st_DIFF_STEP/${cl_1st_difficulty_step}/" \
    | sed "s/CL_1st_ROOT/${cl_1st_root}/" \
    | sed "s#CL_2nd_METRIC_NAME#${cl_2nd_metric}#" \
    | sed "s#CL_2nd_SAMPLE_PATH#${cl_2nd_index_to_sample_path}#" \
    | sed "s#CL_2nd_METRIC_PATH#${cl_2nd_index_to_metric_path}#" \
    | sed "s#CL_2nd_DIFF_TYPE#${cl_2nd_difficulty_type}#" \
    | sed "s#CL_2nd_CLUSTER_TYPE#${cl_2nd_clustering_type}#" \
    | sed "s/CL_2nd_MIN/${cl_2nd_min}/" \
    | sed "s/CL_2nd_MAX/${cl_2nd_max}/" \
    | sed "s/CL_2nd_TOTAL_STEP/${cl_2nd_total_step}/" \
    | sed "s/CL_2nd_DIFF_STEP/${cl_2nd_difficulty_step}/" \
    | sed "s/CL_2nd_ROOT/${cl_2nd_root}/" \
      > ${config_json}
else
template_json="ds_config_gpt_1clmetric_TEMPLATE.json"
sed "s/GBSIZE/${global_batch_size}/" ${template_json} \
    | sed "s/MBSIZE/${batch_size}/" \
    | sed "s/LOG_INTERVAL/${log_interval}/" \
    | sed "s/ZERO_STAGE/${zero_stage}/" \
    | sed "s/PRESCALE_GRAD/${prescale_grad}/" \
    | sed "s/DATA_EFFICIENCY_SEED/${seed}/" \
    | sed "s/LTD_ENABLED/${ltd_enabled}/" \
    | sed "s/LTD_MIN/${ltd_start}/" \
    | sed "s/LTD_MAX/${seq_len}/" \
    | sed "s/LTD_STEP/${ltd_step}/" \
    | sed "s/CL_ENABLED/${cl_enabled}/" \
    | sed "s/DATA_SAMPLING_NUM_WORKERS/${num_workers}/" \
    | sed "s#CL_CLUSTER_PATH#${data_cluster_path}#" \
    | sed "s#CL_1st_METRIC_NAME#${cl_1st_metric}#" \
    | sed "s#CL_1st_SAMPLE_PATH#${cl_1st_index_to_sample_path}#" \
    | sed "s#CL_1st_METRIC_PATH#${cl_1st_index_to_metric_path}#" \
    | sed "s#CL_1st_DIFF_TYPE#${cl_1st_difficulty_type}#" \
    | sed "s#CL_1st_CLUSTER_TYPE#${cl_1st_clustering_type}#" \
    | sed "s/CL_1st_MIN/${cl_1st_min}/" \
    | sed "s/CL_1st_MAX/${cl_1st_max}/" \
    | sed "s/CL_1st_TOTAL_STEP/${cl_1st_total_step}/" \
    | sed "s/CL_1st_DIFF_STEP/${cl_1st_difficulty_step}/" \
    | sed "s/CL_1st_ROOT/${cl_1st_root}/" \
      > ${config_json}
fi
deepspeed_options=" \
    --deepspeed \
    --deepspeed_config ${config_json} \
    --zero-stage ${zero_stage} \
    --pipeline-model-parallel-size ${pp_size}" 

if [[ "${no_pp}" = "true" ]]; then
deepspeed_options="${deepspeed_options} \
    --no-pipeline-parallel"
fi

if [ "${activation_checkpoint}" = "true" ]; then
deepspeed_options="${deepspeed_options} \
    --deepspeed-activation-checkpointing"
fi

## When saving checkpoint to a storage with cache, their could be consistency
## issue of the pointer to latest checkpoint. Here we find the correct pointer
## and broadcast it to all nodes.
iteration_file="$checkpoint_path/latest_checkpointed_iteration.txt"
iteration_file_2="$checkpoint_path/latest"
iteration=0
for (( node = 0; node <= num_node-1; node++ ))
do
    if $(ssh -q worker-"$node" "test -f \"$iteration_file\""); then
        local_iteration=$(ssh -q worker-"$node" cat $iteration_file)
        iteration=$(( ${local_iteration} > ${iteration} ? ${local_iteration} :  ${iteration} ))
    fi
done
if [[ $iteration -gt 0 ]]; then
    iteration_2="global_step${iteration}"
    ds_ssh "echo $iteration > $iteration_file"
    ds_ssh "echo $iteration_2 > $iteration_file_2"
fi
JOB_ID=1010
# HOST_NODE_ADDR="hkbugpusrv04"
PORT=29537
echo "hostname: $(hostname)"
HOST_NODE_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
# echo "HOST_NODE_ADDR: $HOST_NODE_ADDR"
# if [[ "$(hostname)" == "hkbugpusrv04" ]]; then
#     export CUDA_VISIBLE_DEVICES=1
# elif [[ "$(hostname)" == "hkbugpusrv05" ]]; then
#     export CUDA_VISIBLE_DEVICES=1
# else
#     echo "Unknown node: $(hostname)"
#     exit 1
# fi

# export CUDA_VISIBLE_DEVICES=2
torchrun --nnodes=1 --rdzv-id=$JOB_ID --rdzv-backend=c10d --rdzv-endpoint=$HOST_NODE_ADDR:$PORT --nproc-per-node=${num_gpus_pernode} ${dir}/../../../../pretrain_gpt.py ${megatron_options} ${data_options} ${deepspeed_options}
