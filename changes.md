# 4.9 async_checkpoint_engine bug fix

- In `async_checkpoint_engine.py/_copy_tensors_to_cpu_buffers_prealloc`, the logic of checking if a tensor is to be snapshotted has been put under `if (isinstance(cpu_buffer, dict) and key not in cpu_buffer) or (isinstance(cpu_buffer, list) and key >= len(cpu_buffer)):`, 
    - The reason is that in `_init_cpu_buffer`, the tensors not to be snapshotted will not be included into `cpu_buffer`. So if you don't ignore the tensors when judging if the key is in cpu_buffer, it will trigger the key not found error


