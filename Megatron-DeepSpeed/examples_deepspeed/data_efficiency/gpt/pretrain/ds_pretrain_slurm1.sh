srun --pty --partition=i64m1tga40u --job-name=slurm_test --nodes=4 \
--ntasks-per-node=8 --cpus-per-task=4 --gres=gpu:32 \
--nodelist=gpu3-11,gpu3-12 bash ds_pretrain_slurm_run.sh

