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

   
class AsyncCheckpointEngine(CheckpointEngine):
    
    def __init__(self, config_params=None):
        super().__init__(config_params)
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.thread_lock = threading.Lock()
        self.state_dict_cpu = {}

    def create(self, tag):
        log_dist(f"[AsyncCkpt] Checkpoint {tag} is about to be saved!", ranks=[0])

    def save(self, state_dict, path: str, device='cuda:0', snapshot_=True, use_copy_=True, snapshot_stream=torch.cuda.Stream()):
        start_time = time.time()
        self.state_dict_cpu = {}
        if snapshot_:
            self.make_snapshot(state_dict, use_copy_, snapshot_stream, device)
            self.path = path
            # self.save_checkpoint(self.state_dict_cpu, path, snapshot_stream, start_time)
        else:
            self.save_checkpoint(state_dict, path, start_time)
        logger.info(f"[AsyncCkpt] Saved {path}.")
        return None

    def load(self, path: str, map_location=None):
        logger.info(f"[AsyncCkpt] Loading checkpoint from {path}...")
        partition = torch.load(path, map_location=map_location)
        logger.info(f"[AsyncCkpt] Loaded checkpoint from {path}.")
        return partition

    def commit(self, tag):
        logger.info(f"[AsyncCkpt] Checkpoint {tag} is ready now!")
        return True
    
    def make_snapshot(self, state_dict, use_copy_, snapshot_stream, device):
        logger.info(f"[AsyncCkpt] Snapshoting...")

        snapshot_thread = threading.Thread(
            target=self._snapshot_thread,
            args=(state_dict, use_copy_, snapshot_stream, device)
        )
        snapshot_thread.start()
        return snapshot_thread
    
    def save_checkpoint(self, state_dict, path: str, snapshot_stream, start_time):
        checkpoint_thread = threading.Thread(
            target=self._checkpoint_thread,
            args=(state_dict, path, snapshot_stream, start_time)
        )
        checkpoint_thread.start()

    
    def snapshot_and_save():
        pass
        
    def _snapshot_thread(self, state_dict, use_copy_, snapshot_stream, device):
        # if use_timer and step_cnt > 10:
        start_time = time.perf_counter()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._make_snapshot(state_dict, use_copy_, snapshot_stream, device))
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
            


   
    async def _make_snapshot(self, state_dict, use_copy_, snapshot_stream, device):
        def _prepare_cpu_buffers(state_dict):
            stack = [(state_dict, None, None)]
            root = None

            while stack:
                current, parent, key = stack.pop()
                if isinstance(current, torch.Tensor) and current.device.type == 'cuda':
                    if torch.tensor(current.size()).prod().item() > 16777216: # 4096*4096
                        current = current.chunk(8)[0].clone()
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


        def _copy_tensors_to_cpu(data, use_copy_):
            # no use
            stack = [(data, None, None)]
            root = None
            
            while stack:
                current, parent, key = stack.pop()
                if isinstance(current, torch.Tensor):
                    if torch.tensor(current.size()).prod().item() > 16777216: # 4096*4096
                        current = current.chunk(10)[0].clone()
                        # print("chunked")
                    # print("tensor", current.shape)
                    if use_copy_:
                        tensor_cpu = torch.empty_like(current, device='cpu').pin_memory()
                        tensor_cpu.copy_(current, non_blocking=True)
                        result = tensor_cpu
                    else:
                        result = current.cpu()
                    if parent is not None:
                        parent[key] = result
                    else:
                        root = result 
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


        def _copy_tensors_to_cpu_buffers(data, cpu_buffers, use_copy_):
            stack = [(data, cpu_buffers, None)]
        
            while stack:
                current, cpu_buffer, key = stack.pop()
                cpu_buffer = cpu_buffer[key] if key is not None else cpu_buffer
                    
                if isinstance(current, torch.Tensor) and current.device.type == 'cuda':
                    if use_copy_:
                        if torch.tensor(current.size()).prod().item() > 16777216:
                            current = current.chunk(8)[0].clone()
                        cpu_buffer.copy_(current, non_blocking=True)
                    else:
                        cpu_buffer = current.cpu() # not tested
                elif isinstance(current, dict):
                    for k, v in current.items():
                        stack.append((v, cpu_buffer, k))
                elif isinstance(current, list):
                    for idx, item in enumerate(current):
                        stack.append((item, cpu_buffer, idx))
                        
        timestamp = datetime.now().strftime('%H:%M:%S:%X')
        print(f"[{timestamp}][{device}] start _prepare_cpu_buffers")
                  
        self.state_dict_cpu = _prepare_cpu_buffers(state_dict)
        
        timestamp = datetime.now().strftime('%H:%M:%S:%X')
        print(f"[{timestamp}][{device}] end _prepare_cpu_buffers")
        
        # print("buffer", self.state_dict_cpu)

        snapshot_stream.wait_stream(torch.cuda.default_stream(device))
        with torch.cuda.stream(snapshot_stream):
            # self.state_dict_cpu = _copy_tensors_to_cpu(state_dict, use_copy_)
            _copy_tensors_to_cpu_buffers(state_dict, self.state_dict_cpu, use_copy_)
            
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
