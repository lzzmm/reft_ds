srun --pty --partition=i64m1tga800u --job-name=slurm_test --nodes=4 \
--ntasks-per-node=8 --cpus-per-task=8 --gres=gpu:32 \
bash ds_pretrain_slurm_run.sh

