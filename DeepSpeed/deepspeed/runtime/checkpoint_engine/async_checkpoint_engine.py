# lzzmm

import torch
from deepspeed.utils import logger, log_dist
from deepspeed.runtime.checkpoint_engine.checkpoint_engine import \
    CheckpointEngine
    
from concurrent.futures import ThreadPoolExecutor
import os
import asyncio
import aiofiles
import threading
import io
from datetime import datetime
import time
from torch.profiler import profile, record_function
import torch.distributed as dist

   
class AsyncCheckpointEngine(CheckpointEngine):
    
    def __init__(self, config_params=None):
        super().__init__(config_params)
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.thread_lock = threading.Lock()
        self.state_dict_cpu = {}

    def create(self, tag):
        log_dist(f"[AsyncCkpt] Checkpoint {tag} is about to be saved!", ranks=[0])

    def save(self, state_dict, path: str, device='cuda:0', snapshot_=True, use_copy_=True, snapshot_stream=torch.cuda.Stream(), parity_stream=torch.cuda.Stream(), shard_info_dict={}):
        # print(f"rank: {dist.get_rank()}, in engine save")
        info_dir = "/hpc2hdd/home/zli755/xueze/reft_ds/Megatron-DeepSpeed/examples_deepspeed/data_efficiency/gpt/info"
        time_stamp = datetime.now().strftime("%m%d-%H%M%S")
        info_path = os.path.join(info_dir, time_stamp + "_info.txt")
        with open(info_path, "w") as f:
            f.write(str(state_dict))
        # assert shard_info_dict != {}
        start_time = time.time()
        self.state_dict_cpu = {}
        if snapshot_:
            self.make_snapshot(state_dict, use_copy_, snapshot_stream, device, shard_info_dict)
            self.path = path
            # self.save_checkpoint(self.state_dict_cpu, path, snapshot_stream, start_time)
        else:
            self.save_checkpoint(state_dict, path, start_time)
        logger.info(f"[AsyncCkpt] Saved {path}.")
        # self.calculate_parity(state_dict, parity_stream, shard_info_dict)
        return None

    def load(self, path: str, map_location=None):
        logger.info(f"[AsyncCkpt] Loading checkpoint from {path}...")
        partition = torch.load(path, map_location=map_location)
        logger.info(f"[AsyncCkpt] Loaded checkpoint from {path}.")
        return partition

    def commit(self, tag):
        logger.info(f"[AsyncCkpt] Checkpoint {tag} is ready now!")
        return True
    def calculate_parity(self, state_dict, parity_stream, shard_info_dict):
        logger.info(f"[AsyncCkpt] Calculating parity...")
        
        parity_thread = threading.Thread(
            target=self._calculate_parity_thread,
            args=(state_dict, parity_stream, shard_info_dict)
        )
        parity_thread.start()
        return parity_thread
    def _calculate_parity_thread(self, state_dict, parity_stream, shard_info_dict):
        start_time = time.time()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._calculate_parity(state_dict, parity_stream, shard_info_dict))
        finally:
            loop.close()
        end_time = time.time()
        print(f"parity time: {end_time - start_time}\n")
    async def _calculate_parity(self, state_dict, parity_stream, shard_info_dict):
        def compute_xor(*gpu_tensors): 
            int_tensors = [tensor.view(torch.int32) for tensor in gpu_tensors]   
            parity = torch.zeros_like(int_tensors[0])
            for tensor in int_tensors:
                parity ^= tensor
                
            return parity
        
        # print rank and shard_info_dict
        logger.info(f"rank: {dist.get_rank()}, shard_info_dict: {shard_info_dict}")
        if shard_info_dict['pipeline_model_parallel_size'] > 1:
            start_num = 2
        else:
            start_num = 0
        model_params_dict = state_dict['module']['language_model']['encoder']
        # Obtain the layers current rank is responsible for calculating parity
        dp_rank = shard_info_dict['data_parallel_rank']
        assert shard_info_dict['num_layers'] % shard_info_dict['data_parallel_size'] == 0
        layers_per_rank = shard_info_dict['num_layers'] // shard_info_dict['data_parallel_size']
        parity_layer_num_list = []
        for i in range(dp_rank - 1): # For the nodes' dp_rank smaller than current dp_rank
            parity_layer_num_list.append(start_num + i * layers_per_rank + dp_rank - 1)
        for i in range(dp_rank + 1, shard_info_dict['num_layers']): # For the nodes' dp_rank larger than current dp_rank
            parity_layer_num_list.append(start_num + i * layers_per_rank + dp_rank)
        
        parity_tensor_dict_list = [{} for i in range(len(parity_layer_num_list))]
        # Traverse model_params_dict keys, and if parity_layer_num_list[0] is in the key, then add the key and tensor to the parity_tensor_dict, do this until the current key is not in parity_layer_num_list[0], delete parity_layer_num_list[0] and use the next number to compare with the key, continue until parity_layer_num_list is empty
        # print("model_params_dict", model_params_dict.keys())
        # logger.info(f"dp_rank: {dp_rank}, parity_layer_num_list: {parity_layer_num_list}")
        for key, tensor in model_params_dict.items():
            for i, layer_num in enumerate(parity_layer_num_list):
                if str(layer_num) in key:  # This checks for the layer number in the key more robustly
                    parity_tensor_dict_list[i][key] = tensor
                    break  # Move to the next key once a match is found, improving efficiency

        for i in range(len(parity_tensor_dict_list)):
            logger.info(f"dp_rank: {dp_rank}, parity_tensor_dict_list[{i}]: {parity_tensor_dict_list[i].keys()}")
        # Calculate parity
        # with torch.cuda.stream(parity_stream):
        #     parity_tensors = [tensor for tensor in parity_tensor_dict_list.values()]
        #     parity = compute_xor(*parity_tensors)
        #     return parity
            
            
        
    
    def make_snapshot(self, state_dict, use_copy_, snapshot_stream, device, shard_info_dict):
        logger.info(f"[AsyncCkpt] Snapshoting...")

        snapshot_thread = threading.Thread(
            target=self._snapshot_thread,
            args=(state_dict, use_copy_, snapshot_stream, device, shard_info_dict)
        )
        snapshot_thread.start()
        # self.make_snapshot_sync(state_dict, use_copy_, snapshot_stream, device, shard_info_dict)
        # return snapshot_thread
    
    def save_checkpoint(self, state_dict, path: str, snapshot_stream, start_time):
        checkpoint_thread = threading.Thread(
            target=self._checkpoint_thread,
            args=(state_dict, path, snapshot_stream, start_time)
        )
        checkpoint_thread.start()

    
    def snapshot_and_save():
        pass
        
    def _snapshot_thread(self, state_dict, use_copy_, snapshot_stream, device, shard_info_dict):
        # if use_timer and step_cnt > 10:
        start_time = time.perf_counter()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._make_snapshot(state_dict, use_copy_, snapshot_stream, device, shard_info_dict))
        finally:
            loop.close()
        # if use_timer and step_cnt > 10:
        end_time = time.perf_counter()
        print(f"snapshot time: {end_time - start_time}\n")
        # torch.save(self.state_dict_cpu, self.path)        # if snapshot_stream is not None:
        #     snapshot_stream.synchronize()
        # self.save_checkpoint(self.state_dict_cpu, self.path, snapshot_stream, start_time)
        #     timer_record_file.write(f"step: {step_cnt}\n")
        # timer_record_file.write(f"snapshot time: {end_time - start_time}\n")
    
    async def _make_snapshot_rec(self, state_dict, use_copy_, snapshot_stream, device):
        # no use
        def _copy_tensors_to_cpu(data, use_copy_):
            if isinstance(data, dict):
                cpu_data = {}
                for key, value in data.items():
                    if isinstance(value, torch.Tensor):
                        if use_copy_:
                            tensor_cpu = torch.empty_like(value, device='cpu').pin_memory()
                            tensor_cpu.copy_(value, non_blocking=True)
                            cpu_data[key] = tensor_cpu
                        else:
                            cpu_data[key] = value.cpu()
                    elif isinstance(value, (dict, list)):
                        cpu_data[key] = _copy_tensors_to_cpu(value, use_copy_)
                    else:
                        cpu_data[key] = value
                return cpu_data
            elif isinstance(data, list):
                cpu_data = []
                for item in data:
                    if isinstance(item, torch.Tensor):
                        if use_copy_:
                            tensor_cpu = torch.empty_like(item, device='cpu').pin_memory()
                            tensor_cpu.copy_(item, non_blocking=True)
                            cpu_data.append(tensor_cpu)
                        else:
                            cpu_data.append(item.cpu())
                    elif isinstance(item, (dict, list)):
                        cpu_data.append(_copy_tensors_to_cpu(item, use_copy_))
                    else:
                        cpu_data.append(item)
                return cpu_data
            else:
                return data
                        
        snapshot_stream.wait_stream(torch.cuda.default_stream(device))
        with torch.cuda.stream(snapshot_stream):
            self.state_dict_cpu = _copy_tensors_to_cpu(state_dict, use_copy_)
            
    def make_snapshot_sync(self, state_dict, use_copy_, snapshot_stream, device, shard_info_dict):
        def get_shard_layers(shard_info_dict):
            if shard_info_dict['pipeline_model_parallel_size'] > 1:
                start_num = 2
            else:
                start_num = 0
            total_snapshot_layer_num = shard_info_dict['num_layers'] // shard_info_dict['pipeline_model_parallel_size']
            per_node_snapshot_layer_num = total_snapshot_layer_num // shard_info_dict['data_parallel_size']
            shard_layers = range(start_num + shard_info_dict['data_parallel_rank'] * per_node_snapshot_layer_num, start_num + (shard_info_dict['data_parallel_rank'] + 1) * per_node_snapshot_layer_num)
            return shard_layers
        def is_encoder_layer(key, shard_info_dict): 
            if shard_info_dict['pipeline_model_parallel_size'] > 1:
                layer_num = key.split('.')[0]
                if not layer_num.isdigit():
                    return False
                if len(key.split('.')) <= 1:
                    return False
            else:
                layer_num = key.split('.')[1]
                if not layer_num.isdigit():
                    return False
                if key.split('.')[0] != "layers":
                    return False
            return True
        def is_snapshot_shard_tensor(key, shard_info_dict, shard_layers):
            if not isinstance(key, str) or '.' not in key:
                return False
            if shard_info_dict['pipeline_model_parallel_size'] > 1:
                layer_num = key.split('.')[0]
                if not layer_num.isdigit():
                    return False
                layer_num = int(layer_num)
            else:
                layer_num = key.split('.')[1]
                if not layer_num.isdigit():
                    return False
                layer_num = int(layer_num)
            
            if layer_num in shard_layers:
                return True
            else:
                return False
        
        def _prepare_cpu_buffers(state_dict, shard_info_dict):
            stack = [(state_dict, None, None)]
            root = None
            shard_layers = get_shard_layers(shard_info_dict)

            while stack:
                current, parent, key = stack.pop() # current is an object in state_dict, it could be a tensor, a list or a dict
                if isinstance(current, torch.Tensor) and current.device.type == 'cuda':
                    if is_encoder_layer(key, shard_info_dict):
                        if not is_snapshot_shard_tensor(key, shard_info_dict, shard_layers):
                            continue
                    cpu_buffer = torch.empty_like(current, device='cpu').pin_memory()
                    if parent is not None:
                        parent[key] = cpu_buffer
                    else:
                        root = cpu_buffer
                elif isinstance(current, dict):
                    cpu_data = {}
                    for k, v in current.items():
                        stack.append((v, cpu_data, k))
                    if parent is not None:
                        parent[key] = cpu_data
                    else:
                        root = cpu_data
                elif isinstance(current, list):
                    cpu_data = [None] * len(current)
                    for idx, item in enumerate(current):
                        stack.append((item, cpu_data, idx))
                    if parent is not None:
                        parent[key] = cpu_data
                    else:
                        root = cpu_data
                else:
                    if parent is not None:
                        parent[key] = current
                    else:
                        root = current

            return root

        def _copy_tensors_to_cpu_buffers(data, cpu_buffers, use_copy_, shard_info_dict):
            stack = [(data, cpu_buffers, None)]
            shard_layers = get_shard_layers(shard_info_dict)
        
            while stack:
                current, cpu_buffer, key = stack.pop()
                if isinstance(current, torch.Tensor) and current.device.type == 'cuda':
                    if is_encoder_layer(key, shard_info_dict, shard_layers):
                        if not is_snapshot_shard_tensor(key, shard_info_dict, shard_layers):
                            continue
                    cpu_buffer = cpu_buffer[key] if key is not None else cpu_buffer
                    if use_copy_:
                        cpu_buffer.copy_(current, non_blocking=True)
                    else:
                        cpu_buffer = current.cpu() # not tested
                elif isinstance(current, dict):
                    cpu_buffer = cpu_buffer[key] if key is not None else cpu_buffer
                    for k, v in current.items():
                        stack.append((v, cpu_buffer, k))
                elif isinstance(current, list):
                    cpu_buffer = cpu_buffer[key] if key is not None else cpu_buffer
                    for idx, item in enumerate(current):
                        stack.append((item, cpu_buffer, idx))
                        
        timestamp = datetime.now().strftime('%H:%M:%S:%X')
        print(f"[{timestamp}][{device}] start _prepare_cpu_buffers")
                  
        self.state_dict_cpu = _prepare_cpu_buffers(state_dict, shard_info_dict)
        
        timestamp = datetime.now().strftime('%H:%M:%S:%X')
        print(f"[{timestamp}][{device}] end _prepare_cpu_buffers")
        
        # print("buffer", self.state_dict_cpu)

        snapshot_stream.wait_stream(torch.cuda.default_stream(device))
        with torch.cuda.stream(snapshot_stream):
            # self.state_dict_cpu = _copy_tensors_to_cpu(state_dict, use_copy_)
            _copy_tensors_to_cpu_buffers(state_dict, self.state_dict_cpu, use_copy_, shard_info_dict)
            # torch.save(self.state_dict_cpu, self.path) 
            
        timestamp = datetime.now().strftime('%H:%M:%S:%X')
        print(f"[{timestamp}][{device}] end _copy_tensors_to_cpu_buffers")
 
    async def _make_snapshot(self, state_dict, use_copy_, snapshot_stream, device, shard_info_dict):
        def get_shard_layers(shard_info_dict):
            if shard_info_dict['pipeline_model_parallel_size'] > 1:
                start_num = 2
            else:
                start_num = 0
            total_snapshot_layer_num = shard_info_dict['num_layers'] // shard_info_dict['pipeline_model_parallel_size']
            per_node_snapshot_layer_num = total_snapshot_layer_num // shard_info_dict['data_parallel_size']
            shard_layers = range(start_num + shard_info_dict['data_parallel_rank'] * per_node_snapshot_layer_num, start_num + (shard_info_dict['data_parallel_rank'] + 1) * per_node_snapshot_layer_num)
            return shard_layers
        def is_snapshot_shard_tensor(key, shard_info_dict, shard_layers):
            if shard_info_dict['pipeline_model_parallel_size'] > 1:
                layer_num = key.split('.')[0]
                if not layer_num.isdigit():
                    return False
                layer_num = int(layer_num)
            else:
                layer_num = key.split('.')[1]
                if not layer_num.isdigit():
                    return False
                layer_num = int(layer_num)
            
            if layer_num in shard_layers:
                return True
            else:
                return False
        def is_encoder_layer(key, shard_info_dict): 
            # if key is not string
            if not isinstance(key, str) or '.' not in key:
                return False
            if shard_info_dict['pipeline_model_parallel_size'] > 1:
                layer_num = key.split('.')[0]
                if not layer_num.isdigit():
                    return False
                if len(key.split('.')) <= 1:
                    return False
            else:
                layer_num = key.split('.')[1]
                if not layer_num.isdigit():
                    return False
                if key.split('.')[0] != "layers":
                    return False
            return True
        def _prepare_cpu_buffers(state_dict, shard_info_dict):
            stack = [(state_dict, None, None)]
            root = None
            shard_layers = get_shard_layers(shard_info_dict)
            stored_keys = []

            while stack:
                current, parent, key = stack.pop() # current is an object in state_dict, it could be a tensor, a list or a dict
                if isinstance(current, torch.Tensor) and current.device.type == 'cuda':
                    if is_encoder_layer(key, shard_info_dict):
                        if not is_snapshot_shard_tensor(key, shard_info_dict, shard_layers):
                            continue
                    stored_keys.append(key)
                    cpu_buffer = torch.empty_like(current, device='cpu').pin_memory()
                    if parent is not None:
                        parent[key] = cpu_buffer
                    else:
                        root = cpu_buffer
                elif isinstance(current, dict):
                    cpu_data = {}
                    for k, v in current.items():
                        stack.append((v, cpu_data, k))
                    if parent is not None:
                        parent[key] = cpu_data
                    else:
                        root = cpu_data
                elif isinstance(current, list):
                    cpu_data = [None] * len(current)
                    for idx, item in enumerate(current):
                        stack.append((item, cpu_data, idx))
                    if parent is not None:
                        parent[key] = cpu_data
                    else:
                        root = cpu_data
                else:
                    if parent is not None:
                        parent[key] = current
                    else:
                        root = current
            info_dir = "/hpc2hdd/home/zli755/xueze/reft_ds/Megatron-DeepSpeed/examples_deepspeed/data_efficiency/gpt/info"
            time_stamp = datetime.now().strftime("%m%d-%H%M%S")
            info_path = os.path.join(info_dir, f"{time_stamp}_rank_{shard_info_dict['data_parallel_rank']}_stored_keys.txt")
            with open(info_path, "w") as f:
                f.write(str(stored_keys))
            return root

        def _copy_tensors_to_cpu_buffers(data, cpu_buffers, use_copy_, shard_info_dict):
            stack = [(data, cpu_buffers, None)]
            shard_layers = get_shard_layers(shard_info_dict)
        
            while stack:
                current, cpu_buffer, key = stack.pop()
                if isinstance(current, torch.Tensor) and current.device.type == 'cuda':
                    if is_encoder_layer(key, shard_info_dict):
                        if not is_snapshot_shard_tensor(key, shard_info_dict, shard_layers):
                            continue
                    cpu_buffer = cpu_buffer[key] if key is not None else cpu_buffer
                    if use_copy_:
                        cpu_buffer.copy_(current, non_blocking=True)
                    else:
                        cpu_buffer = current.cpu() # not tested
                elif isinstance(current, dict):
                    cpu_buffer = cpu_buffer[key] if key is not None else cpu_buffer
                    for k, v in current.items():
                        stack.append((v, cpu_buffer, k))
                elif isinstance(current, list):
                    cpu_buffer = cpu_buffer[key] if key is not None else cpu_buffer
                    for idx, item in enumerate(current):
                        stack.append((item, cpu_buffer, idx))
                        
        timestamp = datetime.now().strftime('%H:%M:%S:%X')
        print(f"[{timestamp}][{device}] start _prepare_cpu_buffers")
                  
        self.state_dict_cpu = _prepare_cpu_buffers(state_dict, shard_info_dict)
        
        timestamp = datetime.now().strftime('%H:%M:%S:%X')
        print(f"[{timestamp}][{device}] end _prepare_cpu_buffers")
        
        # print("buffer", self.state_dict_cpu)

        snapshot_stream.wait_stream(torch.cuda.default_stream(torch.cuda.current_device()))
        with torch.cuda.stream(snapshot_stream):
            # self.state_dict_cpu = _copy_tensors_to_cpu(state_dict, use_copy_)
            _copy_tensors_to_cpu_buffers(state_dict, self.state_dict_cpu, use_copy_, shard_info_dict)
            torch.save(self.state_dict_cpu, self.path) 
            saved_state_dict_info_dir = "/hpc2hdd/home/zli755/xueze/reft_ds/Megatron-DeepSpeed/examples_deepspeed/data_efficiency/gpt/info/saved_state_dict"
            time_stamp = datetime.now().strftime("%m%d-%H%M%S")
            saved_state_dict_info_path = os.path.join(saved_state_dict_info_dir, f"{time_stamp}_rank_{shard_info_dict['data_parallel_rank']}_state_dict_cpu.txt")
            with open(saved_state_dict_info_path, "w") as f:
                f.write(str(self.state_dict_cpu))
            
        timestamp = datetime.now().strftime('%H:%M:%S:%X')
        print(f"[{timestamp}][{device}] end _copy_tensors_to_cpu_buffers")
        # print("state_dict_cpu", self.state_dict_cpu)
        
        
                        
    def _checkpoint_thread(self, state_dict, path, snapshot_stream, start_time):
        # if snapshot_stream is not None:
        #     snapshot_stream.synchronize()
        logger.info(f"[AsyncCkpt] Saving {path}...")
        # torch.save(state_dict, path)
        start_time = time.time()
        
        timestamp = datetime.now().strftime('%H:%M:%S')
        print(f"[{timestamp}] save_checkpoint started...")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._save_checkpoint(state_dict, path))
        finally:
            loop.close()
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        print("path", path)
        ckpt_size = os.path.getsize(path) / (1024 * 1024)  # MB
        speed = ckpt_size / elapsed_time
        timestamp = datetime.now().strftime('%H:%M:%S')
        print(f"[{timestamp}] Checkpoint size {ckpt_size:.2f} MB, speed {speed:.2f} MB/s, elapsed {elapsed_time:.2f} sec.")

    async def _save_checkpoint(self, state_dict, path):
        # state = {
        #     'model': model.state_dict(),
        #     'optimizer': optimizer.state_dict()
        # }
        buffer = io.BytesIO()
        current_loop = asyncio.get_event_loop()
        
        await current_loop.run_in_executor(self.executor, torch.save, state_dict, buffer)
        buffer.seek(0)
        async with aiofiles.open(path, "wb") as f:
            await f.write(buffer.read())
        # timestamp = datetime.now().strftime('%H:%M:%S')
        # print(f"[{timestamp}] saved")