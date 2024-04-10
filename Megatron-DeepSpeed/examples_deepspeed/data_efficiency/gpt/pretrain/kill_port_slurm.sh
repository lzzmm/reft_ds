srun --partition=i64m1tga800u --job-name=lsof --nodes=1 \
--ntasks-per-node=1 --cpus-per-task=1 --gres=gpu:0 --qos=i64m1tga8+ \
--nodelist=gpu1-19 bash -c 'lsof -ti :29600 | xargs -r kill'

