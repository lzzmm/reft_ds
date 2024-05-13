srun --pty --partition=i64m1tga40u --job-name=slurm_test --nodes=2 \
--ntasks-per-node=1 --cpus-per-task=4 --gres=gpu:2 \
--nodelist=gpu3-11,gpu3-12 bash ds_pretrain_slurm_run.sh

