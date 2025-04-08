import torch
from threading import Thread

def get_gpu_memory_usage(device=None):
    """Return GPU memory usage in MB for a specific device"""
    if torch.cuda.is_available():
        if device is None:
            # Get memory for all devices
            memory_stats = {}
            for i in range(torch.cuda.device_count()):
                torch.cuda.synchronize(i)
                memory_stats[f"GPU {i}"] = torch.cuda.memory_allocated(i) / 1024 / 1024
            return memory_stats
        else:
            # Get memory for specific device
            device_idx = device if isinstance(device, int) else int(device.split(':')[-1])
            torch.cuda.synchronize(device_idx)
            return torch.cuda.memory_allocated(device_idx) / 1024 / 1024
    return 0

def get_tensor_size_mb(tensor):
    """Return tensor size in MB"""
    if tensor is None:
        return 0
    
    # Handle tuple or list of tensors
    if isinstance(tensor, (tuple, list)):
        return sum(get_tensor_size_mb(t) for t in tensor)
    
    # Handle regular tensor
    return tensor.element_size() * tensor.nelement() / 1024 / 1024

def log_memory_usage(tag, tensors_dict=None, device=None, verbose=True):
    """Log memory usage and tensor sizes"""
    memory_usage = get_gpu_memory_usage(device)
    
    if verbose:
        if isinstance(memory_usage, dict):
            # Multiple GPUs
            log_str = f"[{tag}] GPU Memory: " + ", ".join([f"{k}: {v:.2f} MB" for k, v in memory_usage.items()])
        else:
            # Single GPU
            log_str = f"[{tag}] GPU Memory: {memory_usage:.2f} MB"
        
        if tensors_dict:
            log_str += " | Tensors: "
            for name, tensor in tensors_dict.items():
                if tensor is not None:
                    size_mb = get_tensor_size_mb(tensor)
                    
                    # Handle shape display for different types
                    if isinstance(tensor, (tuple, list)):
                        shape_str = "[" + ", ".join(f"{t.shape}" for t in tensor) + "]"
                    else:
                        shape_str = 'x'.join(str(dim) for dim in tensor.shape)
                    
                    log_str += f"{name}({shape_str}): {size_mb:.2f}MB, "
        
        print(log_str, flush=True)
    
    return memory_usage

def log_batch_memory(data_shape, tag="Batch"):
    """Log memory usage for a specific data batch"""
    memory_allocated = torch.cuda.memory_allocated() / 1e9  # Convert to GB
    print(f"{tag} shape: {data_shape}, Memory allocated: {memory_allocated:.2f} GB")
    return memory_allocated

class AsyncLogger:
    """Helper class for asynchronous logging of metrics"""
    
    @staticmethod
    def log_metrics(metrics_dict, wandb_module=None):
        """Log metrics to wandb asynchronously"""
        if wandb_module is None:
            try:
                import wandb as wandb_module
            except ImportError:
                print("wandb not installed, cannot log metrics")
                return
        
        # Copy metrics to avoid reference issues
        metrics_copy = metrics_dict.copy()
        
        def log_thread_fn(metrics_to_log, wb):
            try:
                wb.log(metrics_to_log)
            except Exception as e:
                print(f"Error in wandb logging: {e}")
            finally:
                # Ensure the dictionary is deleted
                del metrics_to_log
        
        # Start thread for logging
        log_thread = Thread(target=log_thread_fn, args=(metrics_copy, wandb_module))
        log_thread.daemon = True  # Make thread daemon so it doesn't prevent program exit
        log_thread.start() 