# 4.14 Tensor shard load implemented
- The loading implementation corresponding to tensor shard has completed

# 4.13 Attach original shape to tensor in cpu buffer
- When loading checkpoint with sharding tensors, it needs to know the original shape of the tensor, so that it can remove the paddings. 
- It's not easy to get this shape from `model`. So we attach this shape to the tensors in the cpu buffer.

# 4.13 stack pop modified
- Modify `async_checkpoint_engine.py`, the `stack.pop` logic in both prealloc and copy is modified to `stack.pop(0)` to ensure the sequence of elements in the saved state dict is identical to the original state dict.

# 4.13 Shard with padding implemented.
- The sharding logic in `async_checkpoint_engine.py` has been modified to shard and separate every tensor to each rank, rather than the previous method to separate and designate layers to each rank.

# 4.10
- Move the layer-sharding-implementation of checkpoint engine to the `bak.py` file. 

# 4.10 add get_state_dict_shape
- Before adding the function of chunking snapshotted tensors, I'd like to get the information of all the shapes of state_dict's tensors first.
- I add a argument to specify if this running would get the state_dict shape or not. If yes, then in `async_checkpoint_engine.py/__update_cpu_buffer`, `current.shape` rather than `current`'s cpu tensor will added into `cpu_buffer`. So the `self.state_dict_cpu`
 we get in the end should be a dict with tensor shapes.
- Then all the ranks will write its `self.state_dict_cpu` to a file. The whole program will exit after execute the first `save`.

# 4.9 async_checkpoint_engine bug fix

- In `async_checkpoint_engine.py/_copy_tensors_to_cpu_buffers_prealloc`, the logic of checking if a tensor is to be snapshotted has been put under `if (isinstance(cpu_buffer, dict) and key not in cpu_buffer) or (isinstance(cpu_buffer, list) and key >= len(cpu_buffer)):`, 
    - The reason is that in `_init_cpu_buffer`, the tensors not to be snapshotted will not be included into `cpu_buffer`. So if you don't ignore the tensors when judging if the key is in cpu_buffer, it will trigger the key not found error


