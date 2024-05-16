srun --pty --partition=i64m1tga40u --job-name=slurm_test --nodes=1 \
--ntasks-per-node=8 --cpus-per-task=4 --gres=gpu:8 \
--nodelist=gpu3-11 -- bash -c '
  # Source environment
  source ~/.bashrc
  
  # Execute the deepspeed pre-training script
  bash ./ds_pretrain_gpt_slurm.sh model_size_config=6 checkpoint_new_thread="true" checkpoint_new_stream="true" \
                     enable_parity="false" enable_pin_memory="true" enable_sharding="false" \
                     enable_profile="true" enable_save="false" save_location="nfs" enable_snapshot="true" \
                     prealloc="true" pure_torch_save="false" get_state_dict_shape="false" \
                     save_checkpoint_in_bubble="true" fail="false" load="false"

'