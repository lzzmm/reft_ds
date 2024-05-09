srun --pty --partition=real --job-name=slurm_test --nodes=1 \
--ntasks-per-node=8 --cpus-per-task=4 --gres=gpu:8 \
--nodelist=hkbugpusrv04 bash ds_pretrain_slurm_run.sh

