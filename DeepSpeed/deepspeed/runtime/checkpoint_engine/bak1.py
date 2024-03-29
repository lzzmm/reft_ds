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
from multiprocessing import Process
import time


   
class AsyncCheckpointEngine(CheckpointEngine):
    
    def __init__(self, config_params=None):
        super().__init__(config_params)
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.thread_lock = threading.Lock()
        self.state_dict_cpu = {}
        self.enable_concurrency = True
        self.enable_sharding = False
        logger.info(f"[AsyncCkpt] enable_sharding: {self.enable_sharding}, enable_concurrency: {self.enable_concurrency}")
        
    def __update_cpu_buffer(self, state_dict, ckpt_args_dict):
        stack = [(state_dict, None, None)]
        # self.state_dict_cpu = None
        shard_layers = self.get_shard_layers(ckpt_args_dict)
        while stack:
            current, parent, key = stack.pop()
            if isinstance(current, torch.Tensor) and current.device.type == 'cuda':
                if not ckpt_args_dict['save_embeddings']:
                    if not self.is_encoder_layer(key, ckpt_args_dict):
                        continue
                    if ckpt_args_dict['enable_sharding']:
                        if not self.is_snapshot_shard_tensor(key, ckpt_args_dict, shard_layers):
                            continue
                else:
                    if ckpt_args_dict['enable_sharding']:
                        if self.is_encoder_layer(key, ckpt_args_dict):
                            if not self.is_snapshot_shard_tensor(key, ckpt_args_dict, shard_layers):
                                continue
                cpu_buffer = torch.empty_like(current, device='cpu').pin_memory()
                if parent is not None:
                    parent[key] = cpu_buffer
                else:
                    self.state_dict_cpu = cpu_buffer
            elif isinstance(current, dict):
                cpu_data = {}
                for k, v in current.items():
                    stack.append((v, cpu_data, k))
                if parent is not None:
                    parent[key] = cpu_data
                else:
                    self.state_dict_cpu = cpu_data
            elif isinstance(current, list):
                cpu_data = [None] * len(current)
                for idx, item in enumerate(current):
                    stack.append((item, cpu_data, idx))
                if parent is not None:
                    parent[key] = cpu_data
                else:
                    self.state_dict_cpu = cpu_data
            else:
                if parent is not None:
                    parent[key] = current
                else:
                    self.state_dict_cpu = current

    def _init_cpu_buffer(self, state_dict, ckpt_args_dict):
        self.__update_cpu_buffer(state_dict, ckpt_args_dict)
        logger.info(f"[AsyncCkpt] CPU buffer initialized.")

    def create(self, tag):
        log_dist(f"[AsyncCkpt] Checkpoint {tag} is about to be saved!", ranks=[0])

    def save(self, state_dict, path: str, device='cuda:0', snapshot_=True, use_copy_=True, snapshot_stream=torch.cuda.Stream(torch.cuda.current_device()), parity_stream=torch.cuda.Stream(), ckpt_args_dict={}):
        # Prepare cpu buffer if ckpt_args_dict['init_cpu_buffer'] = True
        if 'init_cpu_buffer' in ckpt_args_dict and ckpt_args_dict['init_cpu_buffer'] == True:
            buffer_init_thread = threading.Thread(
                target=self._init_cpu_buffer,
                args=(state_dict, ckpt_args_dict)
            )
            buffer_init_thread.start()
            # self.__update_cpu_buffer(state_dict)
            return
        
        # print(f"rank: {dist.get_rank()}, in engine save")
        # info_dir = "/data2/share/md_test/reft_ds/Megatron-DeepSpeed/examples_deepspeed/data_efficiency/gpt/info"
        # info_path = os.path.join(info_dir, "info.txt")
        # with open(info_path, "w") as f:
        #     f.write(str(state_dict))
        # assert ckpt_args_dict != {}
        self.make_snapshot(state_dict, use_copy_, snapshot_stream, device, ckpt_args_dict)
        self.path = path
        logger.info(f"[AsyncCkpt] Saved {path}.")
        # self.calculate_parity(state_dict, parity_stream, ckpt_args_dict)
        return None

    def load(self, path: str, map_location=None):
        logger.info(f"[AsyncCkpt] Loading checkpoint from {path}...")
        partition = torch.load(path, map_location=map_location)
        logger.info(f"[AsyncCkpt] Loaded checkpoint from {path}.")
        return partition

    def commit(self, tag):
        logger.info(f"[AsyncCkpt] Checkpoint {tag} is ready now!")
        return True
    def calculate_parity(self, state_dict, parity_stream, ckpt_args_dict):
        logger.info(f"[AsyncCkpt] Calculating parity...")
        
        parity_thread = threading.Thread(
            target=self._calculate_parity_thread,
            args=(state_dict, parity_stream, ckpt_args_dict)
        )
        parity_thread.start()
        return parity_thread
    def _calculate_parity_thread(self, state_dict, parity_stream, ckpt_args_dict):
        start_time = time.time()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._calculate_parity(state_dict, parity_stream, ckpt_args_dict))
        finally:
            loop.close()
        end_time = time.time()
        print(f"parity time: {end_time - start_time}\n")
    async def _calculate_parity(self, state_dict, parity_stream, ckpt_args_dict):
        def compute_xor(*gpu_tensors): 
            int_tensors = [tensor.view(torch.int32) for tensor in gpu_tensors]   
            parity = torch.zeros_like(int_tensors[0])
            for tensor in int_tensors:
                parity ^= tensor
                
            return parity
        
        # print rank and ckpt_args_dict
        logger.info(f"rank: {dist.get_rank()}, ckpt_args_dict: {ckpt_args_dict}")
        if ckpt_args_dict['pipeline_model_parallel_size'] > 1:
            start_num = 2
        else:
            start_num = 0
        model_params_dict = state_dict['module']['language_model']['encoder']
        # Obtain the layers current rank is responsible for calculating parity
        dp_rank = ckpt_args_dict['data_parallel_rank']
        assert ckpt_args_dict['num_layers'] % ckpt_args_dict['data_parallel_size'] == 0
        layers_per_rank = ckpt_args_dict['num_layers'] // ckpt_args_dict['data_parallel_size']
        parity_layer_num_list = []
        for i in range(dp_rank - 1): # For the nodes' dp_rank smaller than current dp_rank
            parity_layer_num_list.append(start_num + i * layers_per_rank + dp_rank - 1)
        for i in range(dp_rank + 1, ckpt_args_dict['num_layers']): # For the nodes' dp_rank larger than current dp_rank
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
            
    
    def make_snapshot(self, state_dict, use_copy_, snapshot_stream, device, ckpt_args_dict):
        logger.info(f"[AsyncCkpt] Snapshoting...")

        if ckpt_args_dict['checkpoint_new_thread']:
            logger.info(f"[AsyncCkpt] Using concurrency.")
            snapshot_thread = threading.Thread(
                target=self._snapshot_thread,
                args=(state_dict, use_copy_, snapshot_stream, torch.cuda.current_device(), ckpt_args_dict)
            )
            snapshot_thread.start()
        else:
            logger.info(f"[AsyncCkpt] Not using concurrency.")
            self._snapshot_thread(state_dict, use_copy_, snapshot_stream, torch.cuda.current_device(), ckpt_args_dict)
        # self.make_snapshot_sync(state_dict, use_copy_, snapshot_stream, device, ckpt_args_dict)
        # return snapshot_thread
        
    def _snapshot_thread(self, state_dict, use_copy_, snapshot_stream, device, ckpt_args_dict):
        # if use_timer and step_cnt > 10:
        torch.cuda.set_device(device)
        start_time = time.perf_counter()
        self._make_snapshot(state_dict, use_copy_, snapshot_stream, device, ckpt_args_dict)
        end_time = time.perf_counter()
        print(f"snapshot time: {end_time - start_time}\n")

 
    def get_shard_layers(self, ckpt_args_dict):
        if ckpt_args_dict['pipeline_model_parallel_size'] > 1:
            start_num = 2
        else:
            start_num = 0
        total_snapshot_layer_num = ckpt_args_dict['num_layers'] // ckpt_args_dict['pipeline_model_parallel_size']
        per_node_snapshot_layer_num = total_snapshot_layer_num // ckpt_args_dict['data_parallel_size']
        shard_layers = range(start_num + ckpt_args_dict['data_parallel_rank'] * per_node_snapshot_layer_num, start_num + (ckpt_args_dict['data_parallel_rank'] + 1) * per_node_snapshot_layer_num)
        return shard_layers
    def is_snapshot_shard_tensor(self, key, ckpt_args_dict, shard_layers):
        if not isinstance(key, str) or '.' not in key:
            return False
        if ckpt_args_dict['pipeline_model_parallel_size'] > 1:
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
    def is_encoder_layer(self, key, ckpt_args_dict): 
        # if key is not string
        if not isinstance(key, str) or '.' not in key:
            return False
        if ckpt_args_dict['pipeline_model_parallel_size'] > 1:
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
    
    def _make_snapshot(self, state_dict, use_copy_, snapshot_stream, device, ckpt_args_dict):
        
        def _prepare_cpu_buffers(state_dict, ckpt_args_dict):
            stack = [(state_dict, None, None)]
            root = None
            shard_layers = self.get_shard_layers(ckpt_args_dict)
            stored_keys = []

            while stack:
                current, parent, key = stack.pop() # current is an object in state_dict, it could be a tensor, a list or a dict
                if isinstance(current, torch.Tensor) and current.device.type == 'cuda':
                    if not ckpt_args_dict['save_embeddings']:
                        if not self.is_encoder_layer(key, ckpt_args_dict):
                            continue
                        if ckpt_args_dict['enable_sharding']:
                            if not self.is_snapshot_shard_tensor(key, ckpt_args_dict, shard_layers):
                                continue
                    stored_keys.append(key)
                    cpu_buffer = torch.empty_like(current, device='cpu').pin_memory()
                    cpu_buffer.copy_(current, non_blocking=True)
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
                        parent[key] = None
                    else:
                        root = current
            # print("stored_keys", stored_keys)
            return root


        def _copy_tensors_to_cpu_buffers(data, cpu_buffers, use_copy_, ckpt_args_dict):
            stack = [(data, cpu_buffers, None)]
            shard_layers = self.get_shard_layers(ckpt_args_dict)
        
            while stack:
                current, cpu_buffer, key = stack.pop()
                if key is not None:
                    if key not in cpu_buffer:
                        continue
                if isinstance(current, torch.Tensor) and current.device.type == 'cuda':
                    if not ckpt_args_dict['save_embeddings']:
                        if not self.is_encoder_layer(key, ckpt_args_dict):
                            continue
                        if ckpt_args_dict['enable_sharding']:
                            if not self.is_snapshot_shard_tensor(key, ckpt_args_dict, shard_layers):
                                continue
                    cpu_buffer = cpu_buffer[key] if key is not None else cpu_buffer
                    if use_copy_ and cpu_buffer is not None:
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
                else: 
                    if cpu_buffer is not None:
                        cpu_buffer[key] = current
                        
        def _copy_tensors_to_cpu_buffers_prealloc(data, cpu_buffers, use_copy_, ckpt_args_dict):
            # DONE: update cpu buffers if key error
            stack = [(data, cpu_buffers, None)]
            shard_layers = self.get_shard_layers(ckpt_args_dict)
        
            while stack:
                current, cpu_buffer, key = stack.pop()
                if key is not None:
                    if key not in cpu_buffer:
                        # print("key not in cpu_buffer: ", key)
                        if isinstance(current, torch.Tensor) and current.device.type == 'cuda':
                            if not ckpt_args_dict['save_embeddings']:
                                if not self.is_encoder_layer(key, ckpt_args_dict):
                                    continue
                                if ckpt_args_dict['enable_sharding']:
                                    if not self.is_snapshot_shard_tensor(key, ckpt_args_dict, shard_layers):
                                        continue
                            if use_copy_:
                                if isinstance(cpu_buffer, dict):
                                    cpu_buffer[key] = torch.empty_like(current, device='cpu').pin_memory()
                                    cpu_buffer[key].copy_(current, non_blocking=True)
                                elif isinstance(cpu_buffer, list):
                                    # for i in range(key-len(cpu_buffer)):
                                    #     cpu_buffer.append(None)
                                    cpu_buffer.append(torch.empty_like(current, device='cpu').pin_memory())
                                    cpu_buffer[key].copy_(current, non_blocking=True)
                            else:
                                cpu_buffer[key] = current.cpu() # not tested
                        elif isinstance(current, dict):
                            if isinstance(cpu_buffer, dict):
                                cpu_buffer[key] = {}
                            elif isinstance(cpu_buffer, list):
                                # for i in range(key-len(cpu_buffer)):
                                #     cpu_buffer.append(None)
                                cpu_buffer.append({None})
                            cpu_buffer = cpu_buffer[key]
                            for k, v in current.items():
                                stack.append((v, cpu_buffer, k))
                        elif isinstance(current, list):
                            if isinstance(cpu_buffer, dict):
                                cpu_buffer[key] = [None] * len(current)
                            elif isinstance(cpu_buffer, list):
                                # for i in range(key-len(cpu_buffer)):
                                #     cpu_buffer.append(None)
                                cpu_buffer.append([None] * len(current))
                            cpu_buffer = cpu_buffer[key]
                            for idx, item in enumerate(current):
                                stack.append((item, cpu_buffer, idx))
                        continue
                    else:
                        cpu_buffer = cpu_buffer[key]
                if isinstance(current, torch.Tensor) and current.device.type == 'cuda':
                    if not ckpt_args_dict['save_embeddings']:
                        if not self.is_encoder_layer(key, ckpt_args_dict):
                            continue
                        if ckpt_args_dict['enable_sharding']:
                            if not self.is_snapshot_shard_tensor(key, ckpt_args_dict, shard_layers):
                                continue
                    if use_copy_ and cpu_buffer.device.type == 'cpu':
                        cpu_buffer.copy_(current, non_blocking=True)
                    else:
                        cpu_buffer = current.cpu() # not tested
                elif isinstance(current, dict):
                    for k, v in current.items():
                        stack.append((v, cpu_buffer, k))
                elif isinstance(current, list):
                    for idx, item in enumerate(current):
                        stack.append((item, cpu_buffer, idx))
                else:
                    pass
                    # print("not dict or list", type(current))
                        
        # timestamp = datetime.now().strftime('%H:%M:%S:%X')
        # print(f"[{timestamp}][{device}] start _prepare_cpu_buffers")
                  
        # self.state_dict_cpu = _prepare_cpu_buffers(state_dict, ckpt_args_dict)
        
        # timestamp = datetime.now().strftime('%H:%M:%S:%X')
        # print(f"[{timestamp}][{device}] end _prepare_cpu_buffers")
        
        # print("buffer", self.state_dict_cpu)

        if ckpt_args_dict['checkpoint_new_stream']:
            snapshot_stream.wait_stream(torch.cuda.default_stream(device))
            with torch.cuda.stream(snapshot_stream):
                if 'pre_alloc' in ckpt_args_dict and ckpt_args_dict['pre_alloc'] == True:
                    _copy_tensors_to_cpu_buffers_prealloc(state_dict, self.state_dict_cpu, use_copy_, ckpt_args_dict)
                else:
                    self.state_dict_cpu = _prepare_cpu_buffers(state_dict, ckpt_args_dict)
                    _copy_tensors_to_cpu_buffers(state_dict, self.state_dict_cpu, use_copy_, ckpt_args_dict)
                # save_process = Process(target=torch.save, args=(self.state_dict_cpu, self.path))
                # save_process.start()
        else:
            if 'pre_alloc' in ckpt_args_dict and ckpt_args_dict['pre_alloc'] == True:
                _copy_tensors_to_cpu_buffers_prealloc(state_dict, self.state_dict_cpu, use_copy_, ckpt_args_dict)
            else:
                self.state_dict_cpu = _prepare_cpu_buffers(state_dict, ckpt_args_dict)
                _copy_tensors_to_cpu_buffers(state_dict, self.state_dict_cpu, use_copy_, ckpt_args_dict)
            # torch.save(self.state_dict_cpu, self.path)
            # torch.save(self.state_dict_cpu, self.path) 