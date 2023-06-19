from pycbc.waveform import get_td_waveform, get_fd_waveform
import multiprocessing as mp
#import h5py
import numpy as np
import os
import json
import time

#os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

#arguments for waveform generator

n_cpus = 20

approximant = "SpinTaylorT4"
approximant = "TaylorF2"
f_lower = 18.0
delta_t = 1/2048
duration = 1024
f_final = (1/delta_t)/2
delta_f = 1/duration

flen = int(f_final/delta_f) + 1

#if possible, make templates_per_file a multiple of n_cpus
templates_per_file = 1000


#directory of all GWSamplegen template banks
main_dir = "./template_banks/"


#name of template bank directory
bank_dir = "BNS_lowspin_freqseries"


#priors of the template bank. 

mass1_min = 1.0
mass1_max = 3.0

mass2_min = 1.0
mass2_max = 3.0

q_min = 0.1
q_max = 1.0

#rather than selecting for spins, we scale the template spins by a factor. set to 0 for no spins, 1 for full spins
spin_scale = 0.05

#check if directory exists
if not os.path.exists(main_dir+bank_dir):
    os.mkdir(main_dir+bank_dir)

args = {'approximant': approximant,'f_lower': f_lower,'delta_t': delta_t, 'delta_f': delta_f, 'templates_per_file': templates_per_file,
        'mass1_min': mass1_min,'mass1_max': mass1_max,'mass2_min': mass2_min,'mass2_max': mass2_max,
        'q_min': q_min,'q_max': q_max}


#templates are stored in the form: chirp mass, mass 1, mass 2, spin 1z, spin 2z
templates = np.load(main_dir+"GSTLal_templates.npy")

#select only templates that are within the specified range

#templates = templates[(templates[:,1] >= mass1_min) & (templates[:,1] <= mass1_max) & (templates[:,2] >= mass2_min) & 
#    (templates[:,2] <= mass2_max) & (templates[:,3] >= spin1z_min) & (templates[:,3] <= spin1z_max) & 
#    (templates[:,4] >= spin2z_min) & (templates[:,4] <= spin2z_max) & (templates[:,2]/templates[:,1] >= q_min) & 
#    (templates[:,2]/templates[:,1] <= q_max)]

templates = templates[(templates[:,1] >= mass1_min) & (templates[:,1] <= mass1_max) & (templates[:,2] >= mass2_min) & 
    (templates[:,2] <= mass2_max) & (templates[:,2]/templates[:,1] >= q_min) & (templates[:,2]/templates[:,1] <= q_max)]

templates[:,3] *= spin_scale
templates[:,4] *= spin_scale

#sort the templates by chirp mass
templates = templates[templates[:,0].argsort()]

#save the template waveform params and waveform generation args for future reference
np.save(main_dir+bank_dir+"/params.npy",templates)

with open(main_dir+bank_dir+"/args.json", 'w') as fp:
    json.dump(args, fp, sort_keys=False, indent=4)



def get_td_waveform_mp(args):
    hp, _ = get_td_waveform(mass1 = args[1], mass2 = args[2], spin1z = args[3], spin2z = args[4],
            approximant = approximant, f_lower = f_lower, delta_t = delta_t)
    #hp = hp.to_frequencyseries()
    return hp

def get_fd_waveform_mp(args):
    hp, _ = get_fd_waveform(mass1 = args[1], mass2 = args[2], spin1z = args[3], spin2z = args[4],
            approximant = approximant, f_lower = f_lower, delta_f = delta_f, f_final = f_final)
    #hp = hp.to_frequencyseries()
    return hp



savetime = 0
waveformtime = 0

for j in range(int(np.ceil(len(templates)/templates_per_file))):


    #handle the case of templates not being divisible by templates_per_file
    if j == len(templates)//templates_per_file:
        param_list = templates[j*templates_per_file:,:]
    else:
        param_list = templates[j*templates_per_file:(j+1)*templates_per_file,:]

    #print(param_list)

    fname = main_dir+bank_dir+"/"+str(j*templates_per_file)+".npy"
    #fp = np.memmap(fname, dtype=np.complex128, mode='w+', 
    #           shape=(len(param_list),flen), offset=128)

    start = time.time()
    with mp.Pool(processes = n_cpus) as pool:
        waveforms = pool.map(get_fd_waveform_mp, param_list)
    waveformtime += time.time() - start

    start = time.time()
    #fp[:] = waveforms
    #fp.flush()
    #header = np.lib.format.header_data_from_array_1_0(fp)
    #with open(fname, 'r+b') as f:
    #    np.lib.format.write_array_header_1_0(f, header)
    np.save(fname, np.array(waveforms))

    

    #np.savez(fname, *waveforms)
    savetime += time.time() - start

    print(fname)
    
print("Save time: ", savetime)
print("Waveform time: ", waveformtime)