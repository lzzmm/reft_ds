srun --pty --partition=i64m1tga40u --job-name=slurm_test --nodes=4 \
--ntasks-per-node=8 --cpus-per-task=4 --gres=gpu:32 \
--nodelist=gpu3-11,gpu3-12 -- bash -c '
  # Source environment
  source ~/.bashrc
  
  # Execute the deepspeed pre-training script
  bash ./ds_pretrain_gpt_slurm_test.sh true false
'
