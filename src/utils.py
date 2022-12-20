import numpy as np

import time
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

from src.models import ALS, eALS
from src.metrics import hit_ratio, nDCG, MRR


def log_experiment(users, items, target, k):
    
    metrics = [hit_ratio, nDCG, MRR]
    
    result = []
    
    for metric in metrics:
        
        result.append(metric(users, items, target, k).cpu().numpy())
        
    return result
    
    
def plot_visualization(logs, labels):
    fig, ax = plt.subplots(1, 3, figsize=(24, 6))
    names = ['Hit-ratio', 'NDCG', 'MRR']
    for i in range(3):
        for log, label in zip(logs, labels): 
            ax[i].plot(np.arange(iterations), log[:, i], label=label)
        ax[i].grid()
        ax[i].set_title(names[i], fontsize=30)
        ax[i].legend()


def measure_time(data,
                 model_classes=[ALS, eALS], 
                 hidden_sizes=[24, 32, 48, 64, 96, 128],
                 iterations=10,
                 device='cpu', callback=None):
    
    times = []
    
    for hidden_size in tqdm(hidden_sizes):
        
        time_hidden_size = []
        
        for model_class in model_classes:
            
            model = model_class(hidden_size, iterations=iterations, device=device, callback=callback)
            
            start_time = time.time()
            
            model.fit(data)
            
            working_time = time.time() - start_time
            
            time_hidden_size.append(working_time / iterations)
            
        times.append(time_hidden_size)
        
    return times 

