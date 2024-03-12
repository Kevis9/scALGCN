import dgl
import random
import torch
import os
import numpy as np

def setup_seed(seed=32):   
    dgl.seed(seed)    
    dgl.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)       
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed) #所有GPU
        torch.cuda.manual_seed(seed)     # 当前GPU    
        # CUDA有些算法是non deterministic, 需要限制    
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8' # CUDA >= 10.2版本会提示设置这个环境变量
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print("set up seed!")
