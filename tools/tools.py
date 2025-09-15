import numpy as np
from sklearn.metrics import accuracy_score, f1_score,mean_squared_error, r2_score,precision_score,recall_score
import torch
import torch.nn as nn
import pynvml
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import random
def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def assign_gpu(gpu_ids, memory_limit=1e16):
    if len(gpu_ids) == 0 and torch.cuda.is_available():
        pynvml.nvmlInit()
        n_gpus = pynvml.nvmlDeviceGetCount()
        dst_gpu_id, min_mem_used = 0, memory_limit
        for g_id in range(n_gpus):
            handle = pynvml.nvmlDeviceGetHandleByIndex(g_id)
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            mem_used = meminfo.used
            if mem_used < min_mem_used:
                min_mem_used = mem_used
                dst_gpu_id = g_id
        gpu_ids.append(dst_gpu_id)

    using_cuda = len(gpu_ids) > 0 and torch.cuda.is_available()

    device = torch.device('cuda:%d' % int(gpu_ids[0]) if using_cuda else 'cpu')
    return device
