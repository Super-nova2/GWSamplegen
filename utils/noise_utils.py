import os
import numpy as np
from typing import Iterator, List, Optional, Sequence, Tuple
import h5py
import matplotlib.pyplot as plt

import time
#length of noise the waveform is injected into. for BNS, I use 1000 seconds
noise_len = 1000

noise_path = "../real_noise/"


#TODO: handle arbitrary groups of interferometers. Assume each noise file in a dir has the same ifos

#function to retrieve valid start times for injecting gravitational wave samples into. 

    
def get_valid_noise_times(
    noise_dir: str,
    noise_len: int
) -> List[int]:
    
    valid_times = np.array([])
    
    #add on 10 seconds to account for post-whitening truncation
    noise_len += 10
    
    #get all strain file paths from the noise directory, then extract their start time and duration
    paths = os.listdir(noise_dir)
    paths = [path.split("-")[1:] for path in paths if path[:6] == 'strain']
    
    for path in paths:
        path[0] = int(path[0])
        path[1] = int(path[1][:-5])
    
        
        if path[1] <= noise_len:
            print("file length is shorter than desired noise segment length, skipping...")
            continue
        
        times = np.arange(path[0], path[0]+path[1] - noise_len)
        valid_times = np.concatenate((valid_times,times))
        
    #ensure the file paths are in chronological order
    paths = np.array(paths)
    paths = paths[np.argsort(paths[:,0])]
    
    #now that we have all the noise times, we load them into a list 
    np.random.shuffle(valid_times)
    
    
    return valid_times, paths


    
def load_noise_timeseries(    
    paths: np.ndarray
) -> List[np.ndarray]:
    
    noise_list = []
    for path in paths:
        
        f = np.load(noise_path+"HL-"+str(path[0])+"-"+str(path[1])+".npy")

        noise_list.append(f)
        print("loaded a noise file")
        #noise_samples = np.concatenate((noise_samples,f['L1'][()]))
        
    return noise_list
    
    

def fetch_noise_loaded(    
    noise_list: List[np.ndarray],
    noise_len: int,
    noise_start_time: int,
    sample_rate: int,
    paths: np.ndarray
) -> List[float]:
    
    #fetch from a directory of noise files an array of noise for sample generation

    noises = np.empty(shape = (2,noise_len*sample_rate))
    
    f_idx = np.searchsorted(paths[:,0], noise_start_time,side='right') -1
    
    #print(f_idx)
    start_idx = int((noise_start_time - paths[f_idx,0]))*sample_rate

    noises = noise_list[f_idx][:,start_idx:start_idx + noise_len * sample_rate]
    
    return noises