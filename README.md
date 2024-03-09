ds_report

```sh
--------------------------------------------------
DeepSpeed C++/CUDA extension op report
--------------------------------------------------
NOTE: Ops not installed will be just-in-time (JIT) compiled at
      runtime if needed. Op compatibility means that your system
      meet the required dependencies to JIT install the op.
--------------------------------------------------
JIT compiled ops requires ninja
ninja .................. [OKAY]
--------------------------------------------------
op name ................ installed .. compatible
--------------------------------------------------
 [WARNING]  async_io requires the dev libaio .so object and headers but these were not found.
 [WARNING]  async_io: please install the libaio-devel package with yum
 [WARNING]  If libaio is already installed (perhaps from source), try setting the CFLAGS and LDFLAGS environment variables to where it can be found.
async_io ............... [NO] ....... [NO]
fused_adam ............. [NO] ....... [OKAY]
cpu_adam ............... [NO] ....... [OKAY]
cpu_adagrad ............ [NO] ....... [OKAY]
cpu_lion ............... [NO] ....... [OKAY]
 [WARNING]  Please specify the CUTLASS repo directory as environment variable $CUTLASS_PATH
evoformer_attn ......... [NO] ....... [NO]
fused_lamb ............. [NO] ....... [OKAY]
fused_lion ............. [NO] ....... [OKAY]
inference_core_ops ..... [NO] ....... [OKAY]
cutlass_ops ............ [NO] ....... [OKAY]
transformer_inference .. [NO] ....... [OKAY]
quantizer .............. [NO] ....... [OKAY]
ragged_device_ops ...... [NO] ....... [OKAY]
ragged_ops ............. [NO] ....... [OKAY]
random_ltd ............. [NO] ....... [OKAY]
 [WARNING]  sparse_attn requires a torch version >= 1.5 and < 2.0 but detected 2.0
 [WARNING]  using untested triton version (2.0.0), only 1.0.0 is known to be compatible
sparse_attn ............ [NO] ....... [NO]
spatial_inference ...... [NO] ....... [OKAY]
transformer ............ [NO] ....... [OKAY]
stochastic_transformer . [NO] ....... [OKAY]
--------------------------------------------------
DeepSpeed general environment info:
torch install path ............... ['/home/datasets/yxwang/miniconda3/lib/python3.11/site-packages/torch']
torch version .................... 2.0.1+cu117
deepspeed install path ........... ['/home/datasets/yxwang/miniconda3/lib/python3.11/site-packages/deepspeed']
deepspeed info ................... 0.13.5+4578c249, 4578c249, HEAD
torch cuda version ............... 11.7
torch hip version ................ None
nvcc version ..................... 11.7
deepspeed wheel compiled w. ...... torch 2.0, cuda 11.7
shared memory (/dev/shm) size .... 251.53 GB
```

Install modified DeepSpeed

```sh
cd DeepSpeed
pip install .
```

Run Megatron-DeepSpeed

Firstly configure `Megatron-DeepSpeed/examples_deepspeed/bert_with_pile/ds_pretrain_bert_copy.sh`

```
line 9 global_batch_size
line 95 batch_size  Make sure that batch_size <= global_batch_size*pp_size*mp_size/num_gpus
line 85 / 86 GPU num
line 104 num_save
line 123 data_home
line 153 output_home
```

Then run

```sh
./ds_pretrain_bert_copy.sh
```

Thanks!
