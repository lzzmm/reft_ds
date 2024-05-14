# lzzmm

import sys
import torch
import torch.distributed
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
import multiprocessing
import time
import math
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
sys.path.append(root_path)
from output import get_state_dict_shape, nprint

   
class AsyncCheckpointEngine(CheckpointEngine):
    
    def __init__(self, config_params=None, dp_group_ranks=None):
        super().__init__(config_params)
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.thread_lock = threading.Lock()
        self.state_dict_cpu = {}
        self.zero_state_dict_cpu = {}
        self.print_flag = False
        self.init_state_dict_buffer = True
        self.init_zero_state_dict_buffer = True
        self.dp_group_cpu = None
        self.save_dir = None
        # Is using pipeline parallel
        self.is_pipeline = False
        self.copied_tensor_numel = 0
        self.zero_copied_tensor_numel = 0
        self.total_tensor_numel = 0
        self.zero_total_tensor_numel = 0
        self.total_bubble_time = 0
        self.bubble_time_list = []
        self.snapshot_thread_list = []
        self.snapshot_size = 0
        self.zero_snapshot_size = 0
        self.saved_ckpt_template = False
        
    def get_tensor_shard_cpu_buffer(self, tensor, chunk_num):
        # A tensor and chunk_num is sent inside, our target is to get the corresponding chunk of this tensor
        # Then pad it to the uniform size, and send this shard back
        # We don't need to locate the shard, we just need to find out the size.
        original_shape = tensor.shape
        new_dim_0_size = math.ceil(original_shape[0] / (chunk_num * (chunk_num - 1))) * (chunk_num - 1)
        new_shape = (new_dim_0_size, *original_shape[1:])
        return torch.empty(new_shape, device='cpu', dtype=tensor.dtype)
        
    def __update_cpu_buffer(self, state_dict, ckpt_args_dict, is_zero):
        stack = [(state_dict, None, None, None)]
        # self.state_dict_cpu = None
        # timestamp = datetime.now().strftime('%m%d-%H%M')
        # shard_layers = self.get_shard_layers(ckpt_args_dict)
        # info_dir = "/hpc2hdd/home/zli755/xueze/reft_ds/Megatron-DeepSpeed/examples_deepspeed/data_efficiency/gpt/info"
        # info_path = os.path.join(info_dir, f"{timestamp}_dp_{ckpt_args_dict['data_parallel_rank']}_pp_{ckpt_args_dict['pipeline_model_parallel_rank']}_tp_{ckpt_args_dict['tensor_model_parallel_rank']}_state_dict_for_init.txt")
        # with open(info_path, "w") as f:
        #     f.write(str(state_dict))
        while stack:
            current, parent, key, tag = stack.pop(0)
            if isinstance(current, torch.Tensor) and current.device.type == 'cuda' and tag != "rng_state":
                if ckpt_args_dict['zero_stage'] != 0 and tag == "optimizer": 
                    continue
                if not is_zero:
                    self.total_tensor_numel += current.numel()
                    self.snapshot_size += current.element_size() * current.numel()
                else:
                    self.zero_total_tensor_numel += current.numel()
                    self.zero_snapshot_size += current.element_size() * current.numel()
                chunk_num = ckpt_args_dict["data_parallel_size"]
                if ckpt_args_dict["enable_sharding"] and ckpt_args_dict["data_parallel_size"] > 1 and not is_zero:
                    if ckpt_args_dict['enable_pin_memory']:
                        cpu_buffer = self.get_tensor_shard_cpu_buffer(current, chunk_num).pin_memory()
                    else:
                        cpu_buffer = self.get_tensor_shard_cpu_buffer(current, chunk_num)
                else:
                    if ckpt_args_dict['enable_pin_memory']:
                        cpu_buffer = torch.empty_like(current, device='cpu').pin_memory()
                    else:
                        cpu_buffer = torch.empty_like(current, device='cpu')
                    
                if parent is not None:
                    parent[key] = (cpu_buffer, current.shape)
                else:
                    if is_zero == False:
                        self.state_dict_cpu = (cpu_buffer, current.shape)
                    else:
                        self.zero_state_dict_cpu = (cpu_buffer, current.shape)
            elif isinstance(current, dict):
                cpu_data = {}
                if type(key) == str:
                    if "embedding" in key:
                        tag = "embedding"
                    if "optimizer" == key:
                        tag = "optimizer"
                    if "rng_state" in key:
                        tag = "rng_state"
                for k, v in current.items():
                    stack.append((v, cpu_data, k, tag))
                if parent is not None:
                    parent[key] = cpu_data
                else:
                    if is_zero == False:
                        self.state_dict_cpu = cpu_data
                    else:
                        self.zero_state_dict_cpu = cpu_data
            elif isinstance(current, list):
                if type(key) == str:
                    if "rng_state" in key:
                        tag = "rng_state"
                cpu_data = [None] * len(current)
                for idx, item in enumerate(current):
                    stack.append((item, cpu_data, idx, tag))
                if parent is not None:
                    parent[key] = cpu_data
                else:
                    if is_zero == False:
                        self.state_dict_cpu = cpu_data
                    else:
                        self.zero_state_dict_cpu = cpu_data
            else:
                if parent is not None:
                    parent[key] = current # wait for copy
                    # parent[key] = current
                else:
                    if is_zero == False:
                        self.state_dict_cpu = current
                    else:
                        self.zero_state_dict_cpu = current

    def _init_cpu_buffer(self, state_dict, ckpt_args_dict, is_zero):
        if ckpt_args_dict['get_state_dict_shape']:
            get_state_dict_shape(state_dict, "prealloc", ckpt_args_dict["data_parallel_rank"], ckpt_args_dict["pipeline_model_parallel_rank"], ckpt_args_dict["tensor_model_parallel_rank"], ckpt_args_dict["zero_stage"])
            sys.exit()
        else:
            self.__update_cpu_buffer(state_dict, ckpt_args_dict, is_zero)
            
            
        logger.info(f"[AsyncCkpt] CPU buffer initialized.")
        

    def create(self, tag):
        log_dist(f"[AsyncCkpt] Checkpoint {tag} is about to be saved!", ranks=[0])

    def save(self, state_dict, path: str, use_copy_=True, snapshot_stream=torch.cuda.Stream(torch.cuda.current_device()), ckpt_args_dict={}, is_zero=False, dp_group_cpu=None, save_dir=None, iteration=None, is_pipeline=False, bubble_id=None):
        # Prepare cpu buffer if ckpt_args_dict['init_cpu_buffer'] = True
        self.save_dir = save_dir
        if not ckpt_args_dict['enable_snapshot']:
            return
        
        if self.init_state_dict_buffer == True:
            # buffer_init_thread = threading.Thread(
            #     target=self._init_cpu_buffer,
            #     args=(state_dict, ckpt_args_dict)
            # )
            # buffer_init_thread.start()
            self._init_cpu_buffer(state_dict, ckpt_args_dict, is_zero)
            self.init_state_dict_buffer = False
            # self.__update_cpu_buffer(state_dict)
        if is_zero and self.init_zero_state_dict_buffer == True:
            self._init_cpu_buffer(state_dict, ckpt_args_dict, is_zero)
            self.init_zero_state_dict_buffer = False
            
        if not self.saved_ckpt_template:
            ckpt_template_path = os.path.join(ckpt_args_dict["recovery_dir"], f"dp_{ckpt_args_dict['data_parallel_rank']}_pp_{ckpt_args_dict['pipeline_model_parallel_rank']}_tp_{ckpt_args_dict['tensor_model_parallel_rank']}_state_dict_template.pt")
            if ckpt_args_dict["enable_save"]:
                torch.save(self.state_dict_cpu, ckpt_template_path)
            
            self.saved_ckpt_template = True

        timestamp = datetime.now().strftime('%m%d-%H%M')
        # time stamp with month day hour minute second
        if self.print_flag:
            info_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../Megatron-DeepSpeed/examples_deepspeed/data_efficiency/gpt/info'))
            info_path = os.path.join(info_dir, f"{timestamp}_dp_{ckpt_args_dict['data_parallel_rank']}_pp_{ckpt_args_dict['pipeline_model_parallel_rank']}_tp_{ckpt_args_dict['tensor_model_parallel_rank']}_state_dict.txt")
            with open(info_path, "w") as f:
                f.write(str(state_dict))
            state_dict_cpu_path = os.path.join(info_dir, "state_dict_cpu", f"{timestamp}_dp_{ckpt_args_dict['data_parallel_rank']}_pp_{ckpt_args_dict['pipeline_model_parallel_rank']}_tp_{ckpt_args_dict['tensor_model_parallel_rank']}_state_dict_cpu.txt")
            with open(state_dict_cpu_path, "w") as f:
                f.write(str(self.state_dict_cpu))
            self.print_flag = False
            sys.exit()
        assert ckpt_args_dict != {}
        self.path = path
        self.make_snapshot(state_dict, use_copy_, snapshot_stream, ckpt_args_dict, is_zero, dp_group_cpu, iteration, is_pipeline, bubble_id)
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
    def compute_parity(self, state_dict, ckpt_args_dict, is_optimizer, iteration) :
        parity_thread = threading.Thread(
            target=self.compute_parity_thread,
            args=(state_dict, ckpt_args_dict, is_optimizer, iteration)
        )
        parity_thread.start()
        self.snapshot_thread_list.append(parity_thread)
        
    def compute_parity_thread(self, state_dict, ckpt_args_dict, is_optimizer, iteration):
        # nprint("Into compute parity", "yellow")
        def compute_xor(*gpu_tensors):
            assert torch.cuda.is_available(), "CUDA is not available. This function requires a GPU."
            if gpu_tensors[0].dtype == torch.float32:
                parity_dtype = torch.int32
            else:
                assert gpu_tensors[0].dtype == torch.float16
                parity_dtype = torch.int16  
            int_tensors = [tensor.view(parity_dtype) for tensor in gpu_tensors]   
            parity = torch.zeros_like(int_tensors[0])
            for tensor in int_tensors:
                parity ^= tensor
                
            return parity
        root = None
        dp_size = ckpt_args_dict["data_parallel_size"]
        dp_rank = ckpt_args_dict["data_parallel_rank"]
        stack = [(state_dict, None, None, None)]
        while stack:
            current, parent, key, tag = stack.pop(0)
            if isinstance(current, torch.Tensor) and tag != "rng_state":
                # calculate the parity of this tensor for the current dp rank
                # nprint("Into compute parity tensor", "red")
                shape = current.shape
                # if key == "layers.0.self_attention.query_key_value.weight":
                #     nprint(f"current shape: {shape}, current type: {current.dtype}", "blue")
                padded_shape_dim_0 = math.ceil(shape[0] / (dp_size * (dp_size - 1))) * (dp_size * (dp_size - 1))
                padding = [0] * (2 * (current.dim() - 1))
                padding = padding + [0, padded_shape_dim_0 - shape[0]]
                padded_current_tensor = torch.nn.functional.pad(current, padding)
                # if key == "layers.0.self_attention.query_key_value.weight":
                #     nprint(f"padded_current_tensor shape: {padded_current_tensor.shape}", "blue")
                tensor_shards = padded_current_tensor.chunk(dp_size * (dp_size - 1), dim=0)
                # if key == "layers.0.self_attention.query_key_value.weight":
                #     nprint(f"tensor_shards shape: {tensor_shards[0].shape}", "blue")
                # Figure out the shards participating the parity computation based on dp rank
                current_parity_shards = []
                # i is the dp rank to be taken shards
                for i in range(dp_size):
                    if i != dp_rank:
                        current_parity_shards.append(tensor_shards[i * (dp_size - 1) + (dp_rank if i > dp_rank else dp_rank - 1)])
                parity_buffer = compute_xor(*current_parity_shards)
                # if key == "layers.0.self_attention.query_key_value.weight":
                #     nprint(f"parity_buffer shape: {parity_buffer.shape}", "blue")
                #     sys.exit()
                    
                if parent is not None:
                    parent[key] = parity_buffer.cpu()
                else:
                    root = parity_buffer
            elif isinstance(current, dict):
                parity_buffer = {}
                if type(key) == str:
                    if "embedding" in key:
                        tag = "embedding"
                    if "optimizer" == key:
                        tag = "optimizer"
                    if "rng_state" in key:
                        tag = "rng_state"
                for k, v in current.items():
                    stack.append((v, parity_buffer, k, tag))
                if parent is not None:
                    parent[key] = parity_buffer
                else:
                    root = parity_buffer
            elif isinstance(current, list):
                if type(key) == str:
                    if "rng_state" in key:
                        tag = "rng_state"
                parity_buffer = [None] * len(current)
                for idx, item in enumerate(current):
                    stack.append((item, parity_buffer, idx, tag))
                if parent is not None:
                    parent[key] = parity_buffer
                else:
                    root = parity_buffer
            else:
                if parent is not None:
                    parent[key] = None # wait for copy
                    # parent[key] = current
                else:
                    root = current
                    
        if ckpt_args_dict["enable_save"]:
            tag = f"global_step{iteration}"
            if not os.path.exists(os.path.join(ckpt_args_dict["recovery_dir"], tag)):
                os.makedirs(os.path.join(ckpt_args_dict["recovery_dir"], tag), exist_ok=True)
            if not is_optimizer:
                param_save_path = os.path.join(ckpt_args_dict["recovery_dir"], tag, f"dp_{ckpt_args_dict['data_parallel_rank']}_pp_{ckpt_args_dict['pipeline_model_parallel_rank']}_tp_{ckpt_args_dict['tensor_model_parallel_rank']}_param_parity.pt")
                param_save_process = multiprocessing.Process(target=torch.save, args=(root, param_save_path))
                param_save_process.start()
            else:
                optimizer_save_path = os.path.join(ckpt_args_dict["recovery_dir"], tag, f"dp_{ckpt_args_dict['data_parallel_rank']}_pp_{ckpt_args_dict['pipeline_model_parallel_rank']}_tp_{ckpt_args_dict['tensor_model_parallel_rank']}_optimizer_parity.pt")
                optimizer_save_process = multiprocessing.Process(target=torch.save, args=(root, optimizer_save_path))
                optimizer_save_process.start()
            
    def construct_scatter_dict(self, ckpt_args_dict):
        stack = [(self.zero_state_dict_cpu, None, None, None)]
        split_num = ckpt_args_dict['data_parallel_size'] - 1
        scatter_dict_list = [None for _ in range(split_num)]
        while stack:
            current, parent_list, key, tag = stack.pop(0)
            if isinstance(current, tuple) and isinstance(current[0], torch.Tensor):
                # Use torch.tensor_split to split current[0] into ckpt_args_dict['data_parallel_size'] parts
                # Then parent_list[i][key] = current[0]_split[i]
                current_tensor = current[0]
                split_tensors = torch.tensor_split(current_tensor, split_num)                                    
                # sys.exit()
                assert(parent_list is not None)
                for i in range(split_num):
                    parent_list[i][key] = split_tensors[i]
            elif isinstance(current, dict):
                buffer_list = [{} for _ in range(split_num)]
                if type(key) == str:
                    if "embedding" in key:
                        tag = "embedding"
                    if "optimizer" == key:
                        tag = "optimizer"
                for k, v in current.items():
                    stack.append((v, buffer_list, k, tag))
                if parent_list is not None:
                    for i in range(split_num):
                        parent_list[i][key] = buffer_list[i]
                else:
                    for i in range(split_num):
                        scatter_dict_list[i] = buffer_list[i]
            elif isinstance(current, list):
                buffer_list = [[None] * len(current) for _ in range(split_num)]
                for idx, item in enumerate(current):
                    stack.append((item, buffer_list, idx, tag))
                if parent_list is not None:
                    for i in range(split_num):
                        parent_list[i][key] = buffer_list[i]
                else:
                    for i in range(split_num):
                        scatter_dict_list[i] = buffer_list[i]
            else:
                if parent_list is not None:
                    for parent in parent_list:
                        parent[key] = current
                else:
                    for i in range(split_num):
                        scatter_dict_list[i] = current
                    
        return scatter_dict_list
    
    def save_scatter_dict_list(self, scatter_dict_list, ckpt_args_dict):
        for i in range(ckpt_args_dict['data_parallel_size']):
            if i != ckpt_args_dict['data_parallel_rank']:
                scatter_save_path = os.path.join(self.save_dir, f"dp_{ckpt_args_dict['data_parallel_rank']}_pp_{ckpt_args_dict['pipeline_model_parallel_rank']}_tp_{ckpt_args_dict['tensor_model_parallel_rank']}_scatter_dict_{i}.pt")
                print(f"scatter_save_path: {scatter_save_path}")
                torch.save(scatter_dict_list[i], scatter_save_path)
    
    def make_snapshot(self, state_dict, use_copy_, snapshot_stream, ckpt_args_dict, is_zero, dp_group_cpu, iteration, is_pipeline, bubble_id):

        if ckpt_args_dict['checkpoint_new_thread']:
            snapshot_thread = threading.Thread(
                target=self._snapshot_thread,
                args=(state_dict, use_copy_, snapshot_stream, torch.cuda.current_device(), ckpt_args_dict, is_zero, dp_group_cpu, iteration, is_pipeline, bubble_id)
            )
            snapshot_thread.start()
            self.snapshot_thread_list.append(snapshot_thread)
        else:
            self._snapshot_thread(state_dict, use_copy_, snapshot_stream, torch.cuda.current_device(), ckpt_args_dict, is_zero, dp_group_cpu, iteration, is_pipeline, bubble_id)
        # self.make_snapshot_sync(state_dict, use_copy_, snapshot_stream, device, ckpt_args_dict)
        # return snapshot_thread
        
    def _snapshot_thread(self, state_dict, use_copy_, snapshot_stream, device, ckpt_args_dict, is_zero, dp_group_cpu, iteration, is_pipeline, bubble_id):
        # if use_timer and step_cnt > 10:
        torch.cuda.set_device(device)
        # time.sleep(0.8)
        start_time = time.perf_counter()
        if ckpt_args_dict['pure_torch_save']:
            torch.save(state_dict, self.path)
            snapshot_size_in_MB = os.path.getsize(self.path) / 1024 / 1024
        else:
            self._make_snapshot(state_dict, use_copy_, snapshot_stream, device, ckpt_args_dict, is_zero, is_pipeline, bubble_id)
            # if is_zero:
            #     print(f"Iteration {iteration} Into zero scatter logic")
            #     optimizer_state_dict_scatter_list = self.construct_scatter_dict(ckpt_args_dict)
            #     # optimizer_state_dict_scatter_list = [torch.randn(375422976) for _ in range(ckpt_args_dict['data_parallel_size'])]
            #     # print(f"Iteration {iteration} After construct scatter dict")
            #     optimizer_state_dict_scatter_list.insert(ckpt_args_dict['data_parallel_rank'], None)
            #     scatter_dict_list = []
            #     for i in range(ckpt_args_dict['data_parallel_size']):
            #         if i == ckpt_args_dict['data_parallel_rank']:
            #             scatter_input_list = optimizer_state_dict_scatter_list
            #         else:
            #             scatter_input_list = [None for _ in range(ckpt_args_dict['data_parallel_size'])]
            #             # scatter_input_list = None
            #         scatter_output_list = [None]
            #         dist.scatter_object_list(scatter_output_list, scatter_input_list, group=dp_group_cpu, src=i)
            #         scatter_dict_list.append(scatter_output_list[0])
            #         # scatter_dict_list.append(scatter_output_list)
                    
            #     print(f"Iteration {iteration} After scatter object list")
                
            #     self.save_scatter_dict_list(scatter_dict_list, ckpt_args_dict)
            #     print(f"Iteration {iteration} After save scatter dict list")
        end_time = time.perf_counter()
        if not is_zero:
            snapshot_size_in_MB = self.snapshot_size / 1024 / 1024
            print(f"Iteration {iteration} dp_{ckpt_args_dict['data_parallel_rank']}_pp_{ckpt_args_dict['pipeline_model_parallel_rank']}_tp_{ckpt_args_dict['tensor_model_parallel_rank']} snapshot time: {end_time - start_time}, snapshot_size: {snapshot_size_in_MB} MB, snapshot_speed: {snapshot_size_in_MB / (end_time - start_time)} MB/s")
            print(f"[AsyncCkpt] Iteration {iteration} Snapshot done.")
        else:
            snapshot_size_in_MB = self.zero_snapshot_size / 1024 / 1024
            print(f"Zero Iteration {iteration} dp_{ckpt_args_dict['data_parallel_rank']}_pp_{ckpt_args_dict['pipeline_model_parallel_rank']}_tp_{ckpt_args_dict['tensor_model_parallel_rank']} snapshot time: {end_time - start_time}, snapshot_size: {snapshot_size_in_MB} MB, snapshot_speed: {snapshot_size_in_MB / (end_time - start_time)} MB/s")
            print(f"[AsyncCkpt] Iteration {iteration} Zero checkpoint done.")
 
    def get_shard_layers(self, ckpt_args_dict):
        if ckpt_args_dict['pipeline_model_parallel_size'] > 1:
            start_num = 2
        else:
            start_num = 0
        total_snapshot_layer_num_per_stage = ckpt_args_dict['num_layers'] // ckpt_args_dict['pipeline_model_parallel_size']
        start_num += ckpt_args_dict['pipeline_model_parallel_rank'] * total_snapshot_layer_num_per_stage
        per_dp_node_snapshot_layer_num = total_snapshot_layer_num_per_stage // ckpt_args_dict['data_parallel_size']
        shard_layers = range(start_num + ckpt_args_dict['data_parallel_rank'] * per_dp_node_snapshot_layer_num, start_num + (ckpt_args_dict['data_parallel_rank'] + 1) * per_dp_node_snapshot_layer_num)
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
            if len(key.split('.')) <= 2:
                return False
        else:
            layer_num = key.split('.')[1]
            if not layer_num.isdigit():
                return False
            if key.split('.')[0] != "layers":
                return False
        return True
    
    def _prepare_cpu_buffers(self, state_dict, ckpt_args_dict):
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
                else:
                    if ckpt_args_dict['enable_sharding']:
                        if self.is_encoder_layer(key, ckpt_args_dict):
                            if not self.is_snapshot_shard_tensor(key, ckpt_args_dict, shard_layers):
                                continue
                stored_keys.append(key)
                if ckpt_args_dict['enable_pin_memory']:
                    cpu_buffer = torch.empty_like(current, device='cpu').pin_memory()
                else:
                    cpu_buffer = torch.empty_like(current, device='cpu')
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
    
    def _copy_tensors_to_cpu_buffers(self, data, cpu_buffers, use_copy_, ckpt_args_dict):
        stack = [(data, cpu_buffers, None)]
        shard_layers = self.get_shard_layers(ckpt_args_dict)
        snapshot_size = 0
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
                else:
                    if ckpt_args_dict['enable_sharding']:
                        if self.is_encoder_layer(key, ckpt_args_dict):
                            if not self.is_snapshot_shard_tensor(key, ckpt_args_dict, shard_layers):
                                continue
                snapshot_size += current.element_size() * current.numel()
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
                    
        return snapshot_size
    
    def _copy_tensors_to_cpu_buffers_prealloc(self, state_dict, state_dict_buffer, ckpt_args_dict, is_zero):
        # TODO: update cpu buffers if key error bug fix
        # if ckpt_args_dict['pipeline_model_parallel_rank'] == 1:
        #     print(data, cpu_buffers)
        stack = [(state_dict, state_dict_buffer, None, None)]
        snapshot_size = 0
        # log_dist(f"data {data} \n cpu_buffers {cpu_buffers}", ranks=[0])
        while stack:
            current, cpu_buffer, key, tag = stack.pop(0)
            if key is not None:
                if (isinstance(cpu_buffer, dict) and key not in cpu_buffer) or (isinstance(cpu_buffer, list) and key >= len(cpu_buffer)):
                    if isinstance(current, torch.Tensor):
                        if ckpt_args_dict['zero_stage'] != 0 and tag == "optimizer":
                            continue
                    if torch.distributed.get_rank() == 1:
                        print(f"current: {current}")
                        if isinstance(cpu_buffer, dict):
                            print(f"cpu_buffer: {cpu_buffer.keys()}")
                        print(f"key: {key}")
                    raise KeyError(f"key {key} not in cpu_buffer")
                else:
                    cpu_buffer = cpu_buffer[key]
            
            if isinstance(current, torch.Tensor) and current.device.type == 'cuda' and tag != "rng_state":
                if ckpt_args_dict['zero_stage'] != 0 and tag == "optimizer":
                    continue
                if cpu_buffer[0].device.type == 'cpu':
                    if ckpt_args_dict["enable_sharding"] and ckpt_args_dict["data_parallel_size"] > 1 and not is_zero:
                        # I need to locate the correct position of the shard
                        # and copy pad it to the correct size
                        # then copy the shard to the cpu_buffer
                        shard_dim_0_size = cpu_buffer[0].shape[0]
                        shard_id = ckpt_args_dict["data_parallel_rank"]
                        # There are 3 possibilities
                        if (shard_id + 1) * shard_dim_0_size <= current.shape[0]:
                            shard = current[shard_id * shard_dim_0_size : (shard_id + 1) * shard_dim_0_size]
                        else:
                            if shard_id * shard_dim_0_size < current.shape[0]:
                                shard = current[shard_id * shard_dim_0_size :]
                                shard = torch.cat((shard, torch.zeros(shard_dim_0_size - shard.shape[0], *shard.shape[1:], device=torch.cuda.current_device())), dim=0)
                            else:
                                shard = torch.zeros(shard_dim_0_size, *current.shape[1:], device=torch.cuda.current_device())
                        snapshot_size += shard.element_size() * shard.numel()
                        cpu_buffer[0].copy_(shard, non_blocking=True)
                    else:
                        snapshot_size += current.element_size() * current.numel()
                        cpu_buffer[0].copy_(current, non_blocking=True)
            elif isinstance(current, dict):
                if type(key) == str:
                    if "embedding" in key:
                        tag = "embedding"
                    if "optimizer" == key:
                        tag = "optimizer"
                    if "rng_state" in key:
                        tag = "rng_state"
                for k, v in current.items():
                    stack.append((v, cpu_buffer, k, tag))
            elif isinstance(current, list):
                if type(key) == str:
                    if "rng_state" in key:
                        tag = "rng_state"
                for idx, item in enumerate(current):
                    stack.append((item, cpu_buffer, idx, tag))
            else:
                cpu_buffer = current
                # pass
                # print("not dict or list", type(current))
        # print("snapshot_size", snapshot_size)
        return snapshot_size
    
    def _copy_tensors_to_cpu_buffers_prealloc_with_pipeline(self, state_dict, state_dict_buffer, ckpt_args_dict, current_bubble_id, is_zero):
        stack = [(state_dict, state_dict_buffer, None, None)]
        # log_dist(f"data {data} \n cpu_buffers {cpu_buffers}", ranks=[0])
        bubble_tensor_numel_list = []
        for i in range(len(self.bubble_time_list) - 1):
            if not is_zero:
                bubble_tensor_numel_list.append(math.ceil(self.bubble_time_list[i] / self.total_bubble_time * self.total_tensor_numel))
            else:
                bubble_tensor_numel_list.append(math.ceil(self.bubble_time_list[i] / self.total_bubble_time * self.zero_total_tensor_numel))
                
        if not is_zero:
            bubble_tensor_numel_list.append(self.total_tensor_numel - sum(bubble_tensor_numel_list))
        else:
            bubble_tensor_numel_list.append(self.zero_total_tensor_numel - sum(bubble_tensor_numel_list))
            
        # The tensor that has been processed, when it reaches current_alloc_tensor_numel, we need to stop
        current_processed_tensor_numel = 0
        current_jumped_tensor_numel = 0
        jumped_bubble_num = 0
        while stack:
            current, cpu_buffer, key, tag = stack.pop(0)
            if key is not None:
                if (isinstance(cpu_buffer, dict) and key not in cpu_buffer) or (isinstance(cpu_buffer, list) and key >= len(cpu_buffer)):
                    if isinstance(current, torch.Tensor):
                        if ckpt_args_dict['zero_stage'] != 0 and tag == "optimizer":
                            continue
                    if torch.distributed.get_rank() == 1:
                        print(f"current: {current}")
                        if isinstance(cpu_buffer, dict):
                            print(f"cpu_buffer: {cpu_buffer.keys()}")
                        print(f"key: {key}")
                    raise KeyError(f"key {key} not in cpu_buffer")
                else:
                    cpu_buffer = cpu_buffer[key]
            
            if isinstance(current, torch.Tensor) and current.device.type == 'cuda' and tag != "rng_state":
                if current_bubble_id > jumped_bubble_num:
                    current_jumped_tensor_numel += current.numel()
                    if current_jumped_tensor_numel >= bubble_tensor_numel_list[jumped_bubble_num]:
                        jumped_bubble_num += 1
                        current_jumped_tensor_numel = 0
                    continue
                
                if ckpt_args_dict['zero_stage'] != 0 and tag == "optimizer":
                    continue
                if cpu_buffer[0].device.type == 'cpu':
                    if ckpt_args_dict["enable_sharding"] and ckpt_args_dict["data_parallel_size"] > 1 and not is_zero:
                        # I need to locate the correct position of the shard
                        # and copy pad it to the correct size
                        # then copy the shard to the cpu_buffer
                        shard_dim_0_size = cpu_buffer[0].shape[0]
                        shard_id = ckpt_args_dict["data_parallel_rank"]
                        # There are 3 possibilities
                        if (shard_id + 1) * shard_dim_0_size <= current.shape[0]:
                            shard = current[shard_id * shard_dim_0_size : (shard_id + 1) * shard_dim_0_size]
                        else:
                            if shard_id * shard_dim_0_size < current.shape[0]:
                                shard = current[shard_id * shard_dim_0_size :]
                                shard = torch.cat((shard, torch.zeros(shard_dim_0_size - shard.shape[0], *shard.shape[1:], device=torch.cuda.current_device())), dim=0)
                            else:
                                shard = torch.zeros(shard_dim_0_size, *current.shape[1:], device=torch.cuda.current_device())
                        cpu_buffer[0].copy_(shard, non_blocking=True)
                    else:
                        cpu_buffer[0].copy_(current, non_blocking=True)
                        
                current_processed_tensor_numel += current.numel()
                if current_bubble_id != (len(self.bubble_time_list) - 1) and current_processed_tensor_numel >= bubble_tensor_numel_list[current_bubble_id]:
                    break
            elif isinstance(current, dict):
                if type(key) == str:
                    if "embedding" in key:
                        tag = "embedding"
                    if "optimizer" == key:
                        tag = "optimizer"
                    if "rng_state" in key:
                        tag = "rng_state"
                for k, v in current.items():
                    stack.append((v, cpu_buffer, k, tag))
            elif isinstance(current, list):
                if type(key) == str:
                    if "rng_state" in key:
                        tag = "rng_state"
                for idx, item in enumerate(current):
                    stack.append((item, cpu_buffer, idx, tag))
            else:
                cpu_buffer = current
                # pass
                # print("not dict or list", type(current))
        # print("snapshot_size", snapshot_size)
    
    def _make_snapshot(self, state_dict, use_copy_, snapshot_stream, device, ckpt_args_dict, is_zero, is_pipeline, bubble_id):
        if not is_zero:
            state_dict_buffer = self.state_dict_cpu
        else:
            state_dict_buffer = self.zero_state_dict_cpu
        
        if ckpt_args_dict['checkpoint_new_stream']:
            snapshot_stream.wait_stream(torch.cuda.default_stream(device))
            with torch.cuda.stream(snapshot_stream):
                if 'pre_alloc' in ckpt_args_dict and ckpt_args_dict['pre_alloc'] == True:
                    if not is_pipeline:
                        self._copy_tensors_to_cpu_buffers_prealloc(state_dict, state_dict_buffer, ckpt_args_dict, is_zero)
                        if ckpt_args_dict['enable_save']:
                            save_process = multiprocessing.Process(target=torch.save, args=(state_dict_buffer, self.path))
                            save_process.start()
                    else:
                        self._copy_tensors_to_cpu_buffers_prealloc_with_pipeline(state_dict, state_dict_buffer, ckpt_args_dict, bubble_id, is_zero)
                        if bubble_id == len(self.bubble_time_list) - 1:
                            if ckpt_args_dict['enable_save']:
                                nprint(f"dp_{ckpt_args_dict['data_parallel_rank']}_pp_{ckpt_args_dict['pipeline_model_parallel_rank']}_tp_{ckpt_args_dict['tensor_model_parallel_rank']} save_path {self.path}", "blue")
                                save_process = multiprocessing.Process(target=torch.save, args=(self.state_dict_cpu, self.path))
                                save_process.start()
                    # timestamp = datetime.now().strftime('%m%d-%H%M')
                    # info_dir = "/hpc2hdd/home/zli755/xueze/reft_ds/Megatron-DeepSpeed/examples_deepspeed/data_efficiency/gpt/info"
                    # info_path0 = os.path.join(info_dir, "saved_state_dict", f"{timestamp}_dp_{ckpt_args_dict['data_parallel_rank']}_pp_{ckpt_args_dict['pipeline_model_parallel_rank']}_tp_{ckpt_args_dict['tensor_model_parallel_rank']}_saved_state_dict.txt")
                    # with open(info_path0, "w") as f:
                    #     f.write(str(self.state_dict_cpu))
                    # info_path1 = os.path.join(info_dir, "state_dict", f"{timestamp}_dp_{ckpt_args_dict['data_parallel_rank']}_pp_{ckpt_args_dict['pipeline_model_parallel_rank']}_tp_{ckpt_args_dict['tensor_model_parallel_rank']}_state_dict.txt")
                    # with open(info_path1, "w") as f:
                    #     f.write(str(state_dict))
                else:
                    self.state_dict_cpu = self._prepare_cpu_buffers(state_dict, ckpt_args_dict)
                    self._copy_tensors_to_cpu_buffers(state_dict, self.state_dict_cpu, use_copy_, ckpt_args_dict)
                # Get the size of self.state_dict_cpu
                
        else:
            if 'pre_alloc' in ckpt_args_dict and ckpt_args_dict['pre_alloc'] == True:
                self._copy_tensors_to_cpu_buffers_prealloc(state_dict, state_dict_buffer, ckpt_args_dict)
            else:
                self.state_dict_cpu = self._prepare_cpu_buffers(state_dict, ckpt_args_dict)
                self._copy_tensors_to_cpu_buffers(state_dict, self.state_dict_cpu, use_copy_, ckpt_args_dict)
            if ckpt_args_dict['enable_save']:
                torch.save(self.state_dict_cpu, self.path)
    
    def _copy_tensors_to_cpu_buffers_prealloc_old(self, data, cpu_buffers, use_copy_, ckpt_args_dict):
        # TODO: update cpu buffers if key error bug fix
        # if ckpt_args_dict['pipeline_model_parallel_rank'] == 1:
        #     print(data, cpu_buffers)
        stack = [(data, cpu_buffers, None)]
        shard_layers = self.get_shard_layers(ckpt_args_dict)
        snapshot_size = 0
        # log_dist(f"data {data} \n cpu_buffers {cpu_buffers}", ranks=[0])
        while stack:
            current, cpu_buffer, key = stack.pop()
            if key is not None:
                if key not in cpu_buffer:
                    continue
                    # print("key not in cpu_buffer: ", key)
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
                else:
                    if ckpt_args_dict['enable_sharding']:
                        if self.is_encoder_layer(key, ckpt_args_dict):
                            if not self.is_snapshot_shard_tensor(key, ckpt_args_dict, shard_layers):
                                continue
                snapshot_size += current.element_size() * current.numel()
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
                cpu_buffer = current
                # pass
                # print("not dict or list", type(current))
        # print("snapshot_size", snapshot_size)
        return snapshot_size