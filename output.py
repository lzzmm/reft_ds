import torch
from datetime import datetime
import os
import config
import torch.distributed as dist
from colorama import init, Fore, Style

logger_file = None
init_time_stamp = None

def nprint(message, color):
    """
    Prints a message with a specified color using the colorama library.

    Args:
        message (str): The message to be printed.
        color (str): The name of the color ('red', 'green', 'yellow', etc.).
    """
    # Define a dictionary mapping color names to colorama styles
    colors = {
        'red': Fore.RED,
        'green': Fore.GREEN,
        'yellow': Fore.YELLOW,
        'blue': Fore.BLUE,
        'magenta': Fore.MAGENTA,
        'cyan': Fore.CYAN,
        'white': Fore.WHITE,
        'black': Fore.BLACK
    }

    # Get the selected color or default to white if the color is not found
    selected_color = colors.get(color.lower(), Fore.WHITE)

    # Print the message in the selected color
    print(f"{selected_color}{message}{Style.RESET_ALL}")


def init_logger(init_msg=None):
    global logger_file, init_time_stamp
    info_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'Megatron-DeepSpeed/examples_deepspeed/data_efficiency/gpt/info/log_info'))
    if dist.get_rank() == 0:
        init_time_stamp = datetime.now().strftime('%m%d-%H%M')
        if not os.path.exists(os.path.join(info_dir, init_time_stamp)):
            os.makedirs(os.path.join(info_dir, init_time_stamp))
        broadcast_objects = [init_time_stamp]
        dist.broadcast_object_list(broadcast_objects, src=0)
    else:
        broadcast_objects = [None]
        dist.broadcast_object_list(broadcast_objects, src=0)
        init_time_stamp = broadcast_objects[0]
            
    dist.barrier()
    info_path = os.path.join(info_dir, init_time_stamp, f"{init_time_stamp}_dp_{config.data_parallel_rank}_pp_{config.pipeline_parallel_rank}_tp_{config.tensor_parallel_rank}_zero_{config.zero_stage}_log_info.txt")
    logger_file = open(info_path, "w")
    if init_msg is not None:
        log_info(init_msg)
    
def log_info(msg):
    time_stamp = datetime.now().strftime('%m%d-%H%M%S.%f')
    logger_file.write(f"[{time_stamp} dp: {config.data_parallel_rank} pp: {config.pipeline_parallel_rank} tp: {config.tensor_parallel_rank}] {msg}\n")
    


def get_state_dict_shape(state_dict, info_name, dp_rank, pp_rank, tp_rank, zero_stage):
    root = None
    stack = [(state_dict, None, None, None)]
    while stack:
        current, parent, key, tag = stack.pop(0)
        if isinstance(current, torch.Tensor):
            cpu_buffer = current.shape

            if parent is not None:
                parent[key] = cpu_buffer
            else:
                root = cpu_buffer
        elif isinstance(current, tuple) and isinstance(current[0], torch.Tensor):
            if parent is not None:
                parent[key] = (current[0].shape, current[0].device)
            else:
                root = cpu_buffer
        elif isinstance(current, dict):
            cpu_data = {}
            if type(key) == str:
                if "embedding" in key:
                    tag = "embedding"
                if "optimizer" == key:
                    tag = "optimizer"
            for k, v in current.items():
                stack.append((v, cpu_data, k, tag))
            if parent is not None:
                parent[key] = cpu_data
            else:
                root = cpu_data
        elif isinstance(current, list):
            cpu_data = [None] * len(current)
            for idx, item in enumerate(current):
                stack.append((item, cpu_data, idx, tag))
            if parent is not None:
                parent[key] = cpu_data
            else:
                root = cpu_data
        else:
            if parent is not None:
                parent[key] = current # wait for copy
                # parent[key] = current
            else:
                root = current
                
                
    timestamp = datetime.now().strftime('%m%d-%H%M')
    info_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'Megatron-DeepSpeed/examples_deepspeed/data_efficiency/gpt/info/log_info', info_name))
    # create this directory if it does not exist
    if not os.path.exists(info_dir):
        os.makedirs(info_dir)
    info_path = os.path.join(info_dir, f"{timestamp}_dp_{dp_rank}_pp_{pp_rank}_tp_{tp_rank}_zero_{zero_stage}_{info_name}_state_dict_shape.txt")
  
    with open(info_path, "w") as f:
        f.write(str(root))
        
        
def write_state_dict(prefix, state_dict, dp_rank, pp_rank, tp_rank, zero_stage):
    timestamp = datetime.now().strftime('%m%d-%H%M')
    save_dir = os.path.join("/hpc2hdd/home/zli755/xueze/reft_ds/Megatron-DeepSpeed/examples_deepspeed/data_efficiency/gpt/info", prefix)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    path = os.path.join(save_dir, f"{timestamp}_dp_{dp_rank}_pp_{pp_rank}_tp_{tp_rank}_zero_{zero_stage}_{prefix}_state_dict.txt")
    
    with open(path, "w") as f:
        f.write(str(state_dict))