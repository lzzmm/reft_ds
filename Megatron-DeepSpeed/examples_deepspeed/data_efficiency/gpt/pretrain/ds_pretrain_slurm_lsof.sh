srun --pty --partition=i64m1tga40u --job-name=slurm_lsof --nodes=1 \
--ntasks-per-node=1 --cpus-per-task=1 --gres=gpu:0 \
--nodelist=gpu3-12 bash -c "lsof -i :29520"