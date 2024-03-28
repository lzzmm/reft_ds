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
import multiprocessing as mp


   
class AsyncCheckpointEngine(CheckpointEngine):
    
    def __init__(self, config_params=None):
        super().__init__(config_params)
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.thread_lock = threading.Lock()
        self.state_dict_cpu = {}
        self.enable_concurrency = True
        self.enable_sharding = False
        logger.info(f"[AsyncCkpt] enable_sharding: {self.enable_sharding}, enable_concurrency: {self.enable_concurrency}")

    def create(self, tag):
        log_dist(f"[AsyncCkpt] Checkpoint {tag} is about to be saved!", ranks=[0])

    def save(self, state_dict, path: str, use_copy_=True, snapshot_stream=None, parity_stream=torch.cuda.Stream(), shard_info_dict={}):
        # print(f"rank: {dist.get_rank()}, in engine save")
        assert snapshot_stream is not None
        # logger.info(f"rank: {dist.get_rank()}, current_device: {torch.cuda.current_device()} in engine save")
        # if shard_info_dict["data_parallel_rank"] == 0:
        #     info_dir = "/hpc2hdd/home/zli755/xueze/reft_ds/Megatron-DeepSpeed/examples_deepspeed/data_efficiency/gpt/info"
        #     time_stamp = datetime.now().strftime("%m%d-%H%M%S")
        #     info_path = os.path.join(info_dir, f"{time_stamp}_tp_{shard_info_dict['tensor_model_parallel_rank']}_pp_{shard_info_dict['pipeline_model_parallel_rank']}_state_dict.txt")
        #     with open(info_path, "w") as f:
        #         f.write(str(state_dict))
        # assert shard_info_dict != {}
        self.state_dict_cpu = {}
        self.path = path
        self.make_snapshot(state_dict, use_copy_, snapshot_stream, shard_info_dict)
        # torch.save(state_dict, path)
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
            
            
        
    
    def make_snapshot(self, state_dict, use_copy_, snapshot_stream, shard_info_dict):
        logger.info(f"[AsyncCkpt] Snapshoting...")
        # logger.info(f"rank: {dist.get_rank()}, current_device: {torch.cuda.current_device()} in make_snapshot")
        if self.enable_concurrency:
            snapshot_thread = threading.Thread(
                target=self._snapshot_thread,
                args=(state_dict, use_copy_, snapshot_stream, torch.cuda.current_device(), shard_info_dict)
            )
            snapshot_thread.start()
        else:
            self._snapshot_thread(state_dict, use_copy_, snapshot_stream, torch.cuda.current_device(), shard_info_dict)
        # mp.set_start_method('spawn', force=True)
        # snapshot_process = Process(target=self._snapshot_thread, 
        #                            args=(state_dict, use_copy_, snapshot_stream, torch.cuda.current_device(), shard_info_dict))
        # snapshot_process.start()
        
        
    def _snapshot_thread(self, state_dict, use_copy_, snapshot_stream, device, shard_info_dict):
        # if use_timer and step_cnt > 10:
        torch.cuda.set_device(device)
        logger.info(f"rank: {dist.get_rank()}, current_device: {torch.cuda.current_device()} in _snapshot_thread")
        print("snapshot thread sleep")
        time.sleep(0.3)
        print("snapshot thread wake up")
        start_time = time.perf_counter()
        self.make_snapshot_sync(state_dict, use_copy_, snapshot_stream, device, shard_info_dict)
        # if use_timer and step_cnt > 10:
        end_time = time.perf_counter()
        # print(f"snapshot time: {end_time - start_time}\n")
        print(f"dp_{shard_info_dict['data_parallel_rank']}_pp_{shard_info_dict['pipeline_model_parallel_rank']}_tp_{shard_info_dict['tensor_model_parallel_rank']} snapshot time: {end_time - start_time}\n")
        
    def get_shard_layers(self, shard_info_dict):
        if shard_info_dict['pipeline_model_parallel_size'] > 1:
            start_num = 2
        else:
            start_num = 0
        total_snapshot_layer_num = shard_info_dict['num_layers'] // shard_info_dict['pipeline_model_parallel_size']
        per_node_snapshot_layer_num = total_snapshot_layer_num // shard_info_dict['data_parallel_size']
        shard_layers = range(start_num + shard_info_dict['data_parallel_rank'] * per_node_snapshot_layer_num, start_num + (shard_info_dict['data_parallel_rank'] + 1) * per_node_snapshot_layer_num)
        return shard_layers
    def is_snapshot_shard_tensor(self, key, shard_info_dict, shard_layers):
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
    
    def is_encoder_layer(self, key, shard_info_dict): 
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
        
    def make_snapshot_sync(self, state_dict, use_copy_, snapshot_stream, device, shard_info_dict):
        logger.info(f"rank: {dist.get_rank()}, current_device: {torch.cuda.current_device()} in _make_snapshot")
        def _prepare_cpu_buffers(state_dict, shard_info_dict):
            stack = [(state_dict, None, None)]
            root = None
            shard_layers = self.get_shard_layers(shard_info_dict)
            # stored_keys = []

            while stack:
                current, parent, key = stack.pop() # current is an object in state_dict, it could be a tensor, a list or a dict
                if isinstance(current, torch.Tensor) and current.device.type == 'cuda':
                    if not self.is_encoder_layer(key, shard_info_dict):
                        continue
                    if self.enable_sharding:
                        if not self.is_snapshot_shard_tensor(key, shard_info_dict, shard_layers):
                            continue
                    # stored_keys.append(key)
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
                        parent[key] = current
                    else:
                        root = current
            return root

        def _copy_tensors_to_cpu_buffers(data, cpu_buffers, use_copy_, shard_info_dict):
            stack = [(data, cpu_buffers, None)]
            shard_layers = self.get_shard_layers(shard_info_dict)
        
            while stack:
                current, cpu_buffer, key = stack.pop()
                if isinstance(current, torch.Tensor) and current.device.type == 'cuda':
                    if self.is_encoder_layer(key, shard_info_dict):
                        if not self.is_snapshot_shard_tensor(key, shard_info_dict, shard_layers):
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
        if self.enable_concurrency:
            snapshot_stream.wait_stream(torch.cuda.default_stream(torch.cuda.current_device()))
            with torch.cuda.stream(snapshot_stream):
                # self.state_dict_cpu = _copy_tensors_to_cpu(state_dict, use_copy_)
                self.state_dict_cpu = _prepare_cpu_buffers(state_dict, shard_info_dict)
                # _copy_tensors_to_cpu_buffers(state_dict, self.state_dict_cpu, use_copy_, shard_info_dict)
                # With process
                # save_process = Process(target=torch.save, args=(self.state_dict_cpu, self.path))
                # save_process.start()
                
                # With thread
                # save_thread = threading.Thread(target=torch.save, args=(self.state_dict_cpu, self.path))
                # save_thread.start()
                
                # Direct
                # torch.save(self.state_dict_cpu, self.path) 
        else:
            self.state_dict_cpu = _prepare_cpu_buffers(state_dict, shard_info_dict)
            # _copy_tensors_to_cpu_buffers(state_dict, self.state_dict_cpu, use_copy_, shard_info_dict)
            # torch.save(self.state_dict_cpu, self.path)
                