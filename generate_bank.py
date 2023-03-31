from pycbc.waveform import get_td_waveform
import multiprocessing as mp
import h5py
import numpy as np
import os
import json

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

#arguments for waveform generator

approximant = "SpinTaylorT4"
f_lower = 15.0
delta_t = 1/2048

templates_per_file = 100


#directory of all GWSamplegen template banks
main_dir = "/home/alistair.mcleod/GWSamplegen/template_banks/"


#name of template bank directory
bank_dir = "full_test"


#check if directory exists
if not os.path.exists(main_dir+bank_dir):
    os.mkdir(main_dir+bank_dir)

#priors of the template bank

mass1_min = 1.0
mass1_max = 1.1

mass2_min = 1.0
mass2_max = 1.1

spin1z_min = -1.0
spin1z_max = 1.0

spin2z_min = -1.0
spin2z_max = 1.0

q_min = 0.1
q_max = 1.0





args = {'approximant': approximant,'f_lower': f_lower,'delta_t': delta_t, 'templates_per_file': templates_per_file,
        'mass1_min': mass1_min,'mass1_max': mass1_max,'mass2_min': mass2_min,'mass2_max': mass2_max,
        'spin1z_min': spin1z_min,'spin1z_max': spin1z_max,'spin2z_min': spin2z_min,'spin2z_max': spin2z_max,
        'q_min': q_min,'q_max': q_max}


#templates are stored in the form: chirp mass, mass 1, mass 2, spin 1z, spin 2z
templates = np.load(main_dir+"GSTLal_templates.npy")

#select only templates that are within the specified range

templates = templates[(templates[:,1] >= mass1_min) & (templates[:,1] <= mass1_max) & (templates[:,2] >= mass2_min) & 
    (templates[:,2] <= mass2_max) & (templates[:,3] >= spin1z_min) & (templates[:,3] <= spin1z_max) & 
    (templates[:,4] >= spin2z_min) & (templates[:,4] <= spin2z_max) & (templates[:,2]/templates[:,1] >= q_min) & 
    (templates[:,2]/templates[:,1] <= q_max)]

#sort the templates by chrirp mass
templates = templates[templates[:,0].argsort()]

#save the template params and args for future reference
np.save(main_dir+bank_dir+"/params.npy",templates)

with open(main_dir+bank_dir+"/args.json", 'w') as fp:
    json.dump(args, fp, sort_keys=False, indent=4)


#templates = templates[:,1:3]


def get_fd_waveform_mp(args):
    hp, _ = get_td_waveform(mass1 = args[0], mass2 = args[1], spin1z = args[2], spin2z = args[3],
            approximant = approximant, f_lower = f_lower, delta_t = delta_t)
    hp = hp.to_frequencyseries()
    return hp


for j in range(int(np.ceil(len(templates)/templates_per_file))):

    #handle the case of templates not being divisible by templates_per_file
    if j == len(templates)//templates_per_file:
        param_list = templates[j*templates_per_file:,:]
    else:
        param_list = templates[j*templates_per_file:(j+1)*templates_per_file,:]

    #print(param_list)


    with mp.Pool(processes = 20) as pool:
        waveforms = pool.map(get_fd_waveform_mp, param_list)

    fname = main_dir+bank_dir+"/"+str(j*templates_per_file)+".hdf5"

    print(fname)
    f = h5py.File(fname,'w')
    for i in range(len(waveforms)):
        f.create_dataset(str(i), data = np.abs(waveforms[i]))

    f.close()