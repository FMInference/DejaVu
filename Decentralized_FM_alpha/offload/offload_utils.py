import numpy as np
import torch
import cupy


def pin_memory(array):
    mem = cupy.cuda.alloc_pinned_memory(array.nbytes)
    ret = np.frombuffer(mem, array.dtype, array.size).reshape(array.shape)
    ret[...] = array
    return ret


def copy_torch_gpu_tensor2numpy_cpu_array(gpu_tensor: torch.Tensor, cpu_array: np.ndarray):
    cupy_gpu_tensor = cupy.asarray(gpu_tensor)
    cupy_gpu_tensor.get(out=cpu_array)


def copy_numpy_cpu_array2torch_gpu_tensor(cpu_array: np.ndarray, gpu_tensor: torch.Tensor):
    cupy_gpu_tensor = cupy.asarray(gpu_tensor)
    cupy_gpu_tensor.set(cpu_array)