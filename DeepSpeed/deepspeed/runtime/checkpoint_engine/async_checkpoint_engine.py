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
from output import get_state_dict_shape, nprint, log_info
import config as global_config


class CPUAdamOptimizer:
    def __init__(self, model, dp_rank, dp_size, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = {}
        self.v = {}
        self.param_fp32 = {}
        self.t = 0
        self.generated_grad = {}
        self.total_tensors_size = 0
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                stored_param = torch.tensor_split(param, dp_size, dim=0)[(dp_rank + 1) % dp_size]
                self.m[name] = torch.zeros_like(stored_param, device="cpu", dtype=torch.float32)
                self.v[name] = torch.zeros_like(stored_param, device="cpu", dtype=torch.float32)
                self.total_tensors_size += (stored_param.numel() * stored_param.element_size() * 2)
            # self.generated_grad[name] = torch.randn_like(param, device="cpu", dtype=torch.float32)
            # self.param_fp32[param] = torch.zeros_like(param, device="cpu", dtype=torch.float32)

    def step(self, cpu_grads, dp_rank, dp_size):
        # nprint(f"Into step", "cyan")
        # def is_cpu_available(cpu_id):
        #     cpu_utilization = psutil.cpu_percent(interval=0.1, percpu=True)
        #     return cpu_utilization[cpu_id] < 10
        # total_cpus = multiprocessing.cpu_count()
        # available_cpu = None
        # for cpu_id in range(total_cpus):
        #     if is_cpu_available(cpu_id):
        #         available_cpu = cpu_id
        #         break
        # p = psutil.Process()
        # p.cpu_affinity(list(range(16,63)))

        self.t += 1
        
        for name, grad in cpu_grads.items():
            # grad = torch.tensor_split(grad, dp_size)[(dp_rank + 1) % dp_size]
            self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * grad
            self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * (grad ** 2)
            
            # m_hat = self.m[param] / (1 - self.beta1 ** self.t)
            # v_hat = self.v[param] / (1 - self.beta2 ** self.t)
            
            # self.param_fp32[param] -= self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)
        # nprint(f"dp_{global_config.data_parallel_rank} pp_{global_config.pipeline_parallel_rank} tp_{global_config.tensor_parallel_rank} CPUAdamOptimizer.step time: {end_time - start_time}", "cyan")
   
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
        self.bubble_snapshot_time = 0
        self.grad_buffer = {}
        self.cpu_optimizer_stream = torch.cuda.Stream(torch.cuda.current_device())

        
    def init_grad_buffer(self, model, ckpt_args_dict):
        dp_rank = ckpt_args_dict["data_parallel_rank"]
        dp_size = ckpt_args_dict["data_parallel_size"]
        named_parameters = model.named_parameters()
        for name, param in named_parameters:
            if param.requires_grad:
                record_grad = torch.tensor_split(param, dp_size)[(dp_rank + 1) % dp_size]
                self.grad_buffer[name] = torch.empty_like(record_grad, device='cpu').pin_memory()
        self.cpu_optimizer = CPUAdamOptimizer(model, dp_rank, dp_size)
        
                    
    def cpu_optimizer_step(self, model, ckpt_args_dict):
        with torch.cuda.stream(self.cpu_optimizer_stream):
            cpu_optimizer_step_thread = threading.Thread(
                target=self.cpu_optimizer_step_thread,
                args=(model, ckpt_args_dict)
            )
            cpu_optimizer_step_thread.start()
        
    def cpu_optimizer_step_thread(self, model, ckpt_args_dict):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        dp_rank = ckpt_args_dict["data_parallel_rank"]
        dp_size = ckpt_args_dict["data_parallel_size"]
        named_parameters = model.named_parameters()
        for name, param in named_parameters:
            if param.requires_grad:
                grad = param.grad
                record_grad = torch.tensor_split(grad, dp_size)[(dp_rank + 1) % dp_size]
                self.grad_buffer[name].copy_(record_grad, non_blocking=True)
                
        self.cpu_optimizer.step(self.grad_buffer, dp_rank, dp_size)
        end_event.record()
        if ckpt_args_dict['enable_test_snapshot_time']:
            torch.cuda.synchronize()
            elapsed_time = start_event.elapsed_time(end_event) / 1000
            nprint(f"dp_{ckpt_args_dict['data_parallel_rank']} optimizer step time: {elapsed_time}, optimizer step speed: {self.cpu_optimizer.total_tensors_size / 1024 / 1024 / elapsed_time} MB/s", "magenta")
        
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
        while stack:
            current, parent, key, tag = stack.pop(0)
            if isinstance(current, torch.Tensor) and current.device.type == 'cuda':
                if ckpt_args_dict['zero_stage'] != 0 and tag == "optimizer": 
                    continue
                
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
                        
                if not is_zero:
                    self.total_tensor_numel += cpu_buffer.numel()
                    self.snapshot_size += cpu_buffer.element_size() * cpu_buffer.numel()
                else:
                    self.zero_total_tensor_numel += cpu_buffer.numel()
                    self.zero_snapshot_size += cpu_buffer.element_size() * cpu_buffer.numel()
                    
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
                    # if "rng_state" in key:
                    #     tag = "rng_state"
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
            get_state_dict_shape(state_dict, "get_state_dict_shape", ckpt_args_dict["data_parallel_rank"], ckpt_args_dict["pipeline_model_parallel_rank"], ckpt_args_dict["tensor_model_parallel_rank"], ckpt_args_dict["zero_stage"])
            sys.exit(0)
        else:
            self.__update_cpu_buffer(state_dict, ckpt_args_dict, is_zero)
            if not is_zero:
                # get_state_dict_shape(state_dict, "prealloc bubble", ckpt_args_dict["data_parallel_rank"], ckpt_args_dict["pipeline_model_parallel_rank"], ckpt_args_dict["tensor_model_parallel_rank"], ckpt_args_dict["zero_stage"])
                print(f"dp_{ckpt_args_dict['data_parallel_rank']}_pp_{ckpt_args_dict['pipeline_model_parallel_rank']}_tp_{ckpt_args_dict['tensor_model_parallel_rank']} snapshot_size: {self.snapshot_size / 1024 / 1024} MB")
            else:
                print(f"Zero dp_{ckpt_args_dict['data_parallel_rank']}_pp_{ckpt_args_dict['pipeline_model_parallel_rank']}_tp_{ckpt_args_dict['tensor_model_parallel_rank']} snapshot_size: {self.zero_snapshot_size / 1024 / 1024} MB")
            
            
        logger.info(f"[AsyncCkpt] CPU buffer initialized.")
        

    def create(self, tag):
        # log_dist(f"[AsyncCkpt] Checkpoint {tag} is about to be saved!", ranks=[0])
        pass

    def save(self, state_dict, path: str, use_copy_=True, snapshot_stream=torch.cuda.Stream(torch.cuda.current_device()), ckpt_args_dict={}, is_zero=False, dp_group_cpu=None, save_dir=None, iteration=None, is_pipeline=False, bubble_id=None):
        # Prepare cpu buffer if ckpt_args_dict['init_cpu_buffer'] = True
        self.save_dir = save_dir
        if not ckpt_args_dict['enable_snapshot']:
            return
        
        if self.init_state_dict_buffer == True:
            self._init_cpu_buffer(state_dict, ckpt_args_dict, is_zero)
            self.init_state_dict_buffer = False
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
        
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
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
                # torch.save(root, param_save_path)
            else:
                optimizer_save_path = os.path.join(ckpt_args_dict["recovery_dir"], tag, f"dp_{ckpt_args_dict['data_parallel_rank']}_pp_{ckpt_args_dict['pipeline_model_parallel_rank']}_tp_{ckpt_args_dict['tensor_model_parallel_rank']}_optimizer_parity.pt")
                optimizer_save_process = multiprocessing.Process(target=torch.save, args=(root, optimizer_save_path))
                optimizer_save_process.start()
                # torch.save(root, optimizer_save_path)
                
            end_event.record()
            if ckpt_args_dict['enable_test_snapshot_time']:
                torch.cuda.synchronize()
                elapsed_time = start_event.elapsed_time(end_event) / 1000
                nprint(f"Iteration {iteration} dp_{ckpt_args_dict['data_parallel_rank']}_pp_{ckpt_args_dict['pipeline_model_parallel_rank']}_tp_{ckpt_args_dict['tensor_model_parallel_rank']} parity computation time: {elapsed_time} seconds, parity computation speed: {self.snapshot_size / 1024 / 1024 / elapsed_time} MB/s", "magenta")
    
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
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

        if ckpt_args_dict['enable_snapshot'] and ckpt_args_dict['save_checkpoint_in_bubble'] and bubble_id == 0:
            self.bubble_snapshot_time = 0
        
        if ckpt_args_dict['pure_torch_save']:
            torch.save(state_dict, self.path)
            snapshot_size_in_MB = os.path.getsize(self.path) / 1024 / 1024
        else:
            self._make_snapshot(state_dict, use_copy_, snapshot_stream, device, ckpt_args_dict, is_zero, is_pipeline, bubble_id, iteration)
        end_event.record()
        if ckpt_args_dict['enable_test_snapshot_time']:
            torch.cuda.synchronize()
            elapsed_time = start_event.elapsed_time(end_event) / 1000
            if not is_pipeline:
                if not is_zero:
                    snapshot_size_in_MB = self.snapshot_size / 1024 / 1024
                    nprint(f"Iteration {iteration} dp_{ckpt_args_dict['data_parallel_rank']}_pp_{ckpt_args_dict['pipeline_model_parallel_rank']}_tp_{ckpt_args_dict['tensor_model_parallel_rank']} snapshot time: {elapsed_time}, snapshot_size: {snapshot_size_in_MB} MB, snapshot_speed: {snapshot_size_in_MB / (elapsed_time)} MB/s", "magenta")
                else:
                    snapshot_size_in_MB = self.zero_snapshot_size / 1024 / 1024
                    nprint(f"Zero Iteration {iteration} dp_{ckpt_args_dict['data_parallel_rank']}_pp_{ckpt_args_dict['pipeline_model_parallel_rank']}_tp_{ckpt_args_dict['tensor_model_parallel_rank']} snapshot time: {elapsed_time}, snapshot_size: {snapshot_size_in_MB} MB, snapshot_speed: {snapshot_size_in_MB / (elapsed_time)} MB/s", "magenta")
            else:
                self.bubble_snapshot_time += elapsed_time
                if bubble_id == len(self.bubble_time_list) - 1:
                    snapshot_size_in_MB = self.snapshot_size / 1024 / 1024
                    nprint(f"Iteration {iteration} dp_{ckpt_args_dict['data_parallel_rank']}_pp_{ckpt_args_dict['pipeline_model_parallel_rank']}_tp_{ckpt_args_dict['tensor_model_parallel_rank']} snapshot time: {self.bubble_snapshot_time}, snapshot_size: {snapshot_size_in_MB} MB, snapshot_speed: {snapshot_size_in_MB / self.bubble_snapshot_time} MB/s", "magenta")
 
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
            
            if isinstance(current, torch.Tensor) and current.device.type == 'cuda':
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
                        cpu_buffer[0].copy_(shard, non_blocking=ckpt_args_dict["enable_non_blocking"])
                        
                        if ckpt_args_dict["double_snapshot"]:
                            next_shard_id = (shard_id + 1) % ckpt_args_dict["data_parallel_size"]
                            if (next_shard_id + 1) * shard_dim_0_size <= current.shape[0]:
                                next_shard = current[next_shard_id * shard_dim_0_size : (next_shard_id + 1) * shard_dim_0_size]
                            else:
                                if next_shard_id * shard_dim_0_size < current.shape[0]:
                                    next_shard = current[next_shard_id * shard_dim_0_size :]
                                    next_shard = torch.cat((next_shard, torch.zeros(shard_dim_0_size - next_shard.shape[0], *next_shard.shape[1:], device=torch.cuda.current_device())), dim=0)
                                else:
                                    next_shard = torch.zeros(shard_dim_0_size, *current.shape[1:], device=torch.cuda.current_device())
                                    
                            cpu_buffer[0].copy_(next_shard, non_blocking=ckpt_args_dict["enable_non_blocking"])
                    else:
                        cpu_buffer[0].copy_(current, non_blocking=ckpt_args_dict["enable_non_blocking"])
                        if ckpt_args_dict["double_snapshot"]:
                            cpu_buffer[0].copy_(current, non_blocking=ckpt_args_dict["enable_non_blocking"])
                            
            elif isinstance(current, dict):
                if type(key) == str:
                    if "embedding" in key:
                        tag = "embedding"
                    if "optimizer" == key:
                        tag = "optimizer"
                    # if "rng_state" in key:
                    #     tag = "rng_state"
                for k, v in current.items():
                    stack.append((v, cpu_buffer, k, tag))
            elif isinstance(current, list):
                # if type(key) == str:
                #     if "rng_state" in key:
                #         tag = "rng_state"
                for idx, item in enumerate(current):
                    stack.append((item, cpu_buffer, idx, tag))
            else:
                cpu_buffer = current
                # pass
                # print("not dict or list", type(current))
        # print("snapshot_size", snapshot_size)
    
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
            
            if isinstance(current, torch.Tensor) and current.device.type == 'cuda':
                if current_bubble_id > jumped_bubble_num:
                    current_jumped_tensor_numel += current.numel()
                    try:
                        if current_jumped_tensor_numel >= bubble_tensor_numel_list[jumped_bubble_num]:
                            jumped_bubble_num += 1
                            current_jumped_tensor_numel = 0
                    except Exception as e:
                        nprint(f"dp_{ckpt_args_dict['data_parallel_rank']}_pp_{ckpt_args_dict['pipeline_model_parallel_rank']}_tp_{ckpt_args_dict['tensor_model_parallel_rank']}_current_bubble_id: {current_bubble_id}, jumped_bubble_num: {jumped_bubble_num}, bubble_tensor_numel_list: {len(bubble_tensor_numel_list)}, bubble_time_list: {len(self.bubble_time_list)}", "blue")
                        raise e
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
                        cpu_buffer[0].copy_(shard, non_blocking=ckpt_args_dict["enable_non_blocking"])
                        
                        if ckpt_args_dict["double_snapshot"]:
                            next_shard_id = (shard_id + 1) % ckpt_args_dict["data_parallel_size"]
                            if (next_shard_id + 1) * shard_dim_0_size <= current.shape[0]:
                                next_shard = current[next_shard_id * shard_dim_0_size : (next_shard_id + 1) * shard_dim_0_size]
                            else:
                                if next_shard_id * shard_dim_0_size < current.shape[0]:
                                    next_shard = current[next_shard_id * shard_dim_0_size :]
                                    next_shard = torch.cat((next_shard, torch.zeros(shard_dim_0_size - next_shard.shape[0], *next_shard.shape[1:], device=torch.cuda.current_device())), dim=0)
                                else:
                                    next_shard = torch.zeros(shard_dim_0_size, *current.shape[1:], device=torch.cuda.current_device())
                                    
                            cpu_buffer[0].copy_(next_shard, non_blocking=ckpt_args_dict["enable_non_blocking"])
                            
                        
                    else:
                        cpu_buffer[0].copy_(current, non_blocking=True)
                        if ckpt_args_dict["double_snapshot"]:
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
                    # if "rng_state" in key:
                    #     tag = "rng_state"
                for k, v in current.items():
                    stack.append((v, cpu_buffer, k, tag))
            elif isinstance(current, list):
                # if type(key) == str:
                    # if "rng_state" in key:
                    #     tag = "rng_state"
                for idx, item in enumerate(current):
                    stack.append((item, cpu_buffer, idx, tag))
            else:
                cpu_buffer = current
                # pass
                # print("not dict or list", type(current))
        # print("snapshot_size", snapshot_size)
    
    def _make_snapshot(self, state_dict, use_copy_, snapshot_stream, device, ckpt_args_dict, is_zero, is_pipeline, bubble_id, iteration):
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
                        # get_state_dict_shape(state_dict_buffer, "prealloc", ckpt_args_dict["data_parallel_rank"], ckpt_args_dict["pipeline_model_parallel_rank"], ckpt_args_dict["tensor_model_parallel_rank"], ckpt_args_dict["zero_stage"])
                        if ckpt_args_dict['enable_save']:
                            nprint(f"save snapshot to {self.path}", "magenta")
                            save_process = multiprocessing.Process(target=torch.save, args=(state_dict_buffer, self.path))
                            save_process.start()
                            # torch.save(state_dict_buffer, self.path)
                    else:
                        self._copy_tensors_to_cpu_buffers_prealloc_with_pipeline(state_dict, state_dict_buffer, ckpt_args_dict, bubble_id, is_zero)
                        # get_state_dict_shape(state_dict_buffer, "prealloc_bubble", ckpt_args_dict["data_parallel_rank"], ckpt_args_dict["pipeline_model_parallel_rank"], ckpt_args_dict["tensor_model_parallel_rank"], ckpt_args_dict["zero_stage"])
                        if bubble_id == len(self.bubble_time_list) - 1:
                            if ckpt_args_dict['enable_save']:
                                nprint(f"save snapshot to {self.path}", "magenta")
                                save_process = multiprocessing.Process(target=torch.save, args=(state_dict_buffer, self.path))
                                save_process.start()
                                # torch.save(state_dict_buffer, self.path)
                else:
                    self.state_dict_cpu = self._prepare_cpu_buffers(state_dict, ckpt_args_dict)
                    self._copy_tensors_to_cpu_buffers(state_dict, self.state_dict_cpu, use_copy_, ckpt_args_dict)
                # Get the size of self.state_dict_cpu
                
        else:
            if 'pre_alloc' in ckpt_args_dict and ckpt_args_dict['pre_alloc'] == True:
                self._copy_tensors_to_cpu_buffers_prealloc(state_dict, state_dict_buffer, ckpt_args_dict, is_zero)
            else:
                self.state_dict_cpu = self._prepare_cpu_buffers(state_dict, ckpt_args_dict)
                self._copy_tensors_to_cpu_buffers(state_dict, self.state_dict_cpu, use_copy_, ckpt_args_dict)
            if ckpt_args_dict['enable_save']:
                torch.save(self.state_dict_cpu, self.path)