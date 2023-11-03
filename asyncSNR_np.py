import os
import numpy as np
from typing import Iterator, List, Optional, Sequence, Tuple
from pycbc.filter import highpass, lowpass
from GWSamplegen.noise_utils import get_valid_noise_times, load_noise, fetch_noise_loaded, load_psd
#rom pycbc.filter import matched_filter
from pycbc.detector import Detector
from pycbc.psd import interpolate, inverse_spectrum_truncation
from pycbc.types import FrequencySeries
from pycbc.types.timeseries import TimeSeries
#from GWSamplegen.SNR_utils import array_matched_filter, tf_get_cutoff_indices
from GWSamplegen.snr_utils_np import numpy_matched_filter, np_get_cutoff_indices
#import tensorflow as tf
from pycbc.waveform import get_fd_waveform
import pycbc.noise

#import asyncio
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, wait

import multiprocessing as mp

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--index', type=int)
parser.add_argument('--totaljobs', type=int, default=1)
parser.add_argument('--config-file', type=str, default=None)
args = parser.parse_args()

total_jobs = args.totaljobs
index = args.index
config_file = args.config_file

print("NOW STARTING JOB",index,"OF",total_jobs)


#defining some configs. some of these need to come from config files in the future.
duration = 1024
delta_t = 1.0/2048
sample_rate = int(1/delta_t)
f_lower = 18.0
delta_f = 1/duration
f_final = duration
fd_approximant = 'TaylorF2'
td_approximant = "SpinTaylorT4"

ifos = ['H1', 'L1']

seconds_before = 100
seconds_after = 400
offset = 0

fname = 'SNR.npy'

project_dir = "./configs/train1"
noise_dir = "./noise/test"
#template_dir = "./template_banks/BNS_lowspin_freqseries"

#waveforms_per_file = 100
#templates_per_file = 1000

#samples_per_batch is limited by how many samples we can fit into a 2 Gb tensor
#mp_batch is limited by the amount of memory available.

samples_per_batch = 10
#samples_per_file = 10000


#number of batches to process in parallel. determined by the available cores and memory.
mp_batch = 10

#n_cpus = 10
#set n_cpus from os
n_cpus = int(os.environ['SLURM_CPUS_PER_TASK'])
print("n_cpus:",n_cpus)


offset = np.min((offset*sample_rate, duration//2))

import json
if config_file:
	print("loading args from a config file")
	with open(config_file) as json_file:
		config = json.load(json_file)
		project_dir = config['project_dir']
		noise_dir = config['noise_dir']
		seed = config['seed']
		fd_approximant = config['fd_approximant']
		td_approximant = config['td_approximant']
		noise_type = config['noise_type']
		n_signal_samples = config['n_signal_samples']
		n_noise_samples = config['n_noise_samples']
		ifos = config['detectors']
		seconds_before = config['seconds_before']
		seconds_after = config['seconds_after']
		f_lower = config['f_lower']
		
	for key, value in config.items():
		print(key, value)

samples_per_batch = min(100//(config['templates_per_waveform']),50)
print("SAMPLES_PER_BATCH:",samples_per_batch)

###################################################load noise segments
valid_times, paths, files = get_valid_noise_times(noise_dir,0)
segments = load_noise(noise_dir)

params = np.load(project_dir + "/params.npy", allow_pickle=True).item(0)
template_ids = np.array(params['template_waveforms'])
gps = params['gps']
n_templates = len(template_ids[0])

templates = np.load(project_dir + "/template_params.npy")

#Damon's definition of N. from testing, it's just the total length of the segment in samples
#N = (len(sample1)-1) * 2
N = int(duration/delta_t)
kmin, kmax = np_get_cutoff_indices(f_lower, None, delta_f, N)


##CLEAN UP: JOB ARRAY STUFF GOING HERE FOR NOW

samples_per_file = len(params['mass1'])//total_jobs
print("samples per file is",samples_per_file)


##################################################load PSD
psd = np.load(noise_dir + "/psd.npy")

#since psd[0] is the sample frequencies, and the first frequency is always 0 Hz, psd[0][1] is sample frequency
psds = {}
t_psds = {}
import matplotlib.pyplot as plt
from GWSamplegen.noise_utils import load_psd

psds = load_psd(noise_dir, duration, ifos, f_lower, int(1/delta_t))

for psd in psds:
	psds[psd] = psds[psd][kmin:kmax]

#for i in range(len(ifos)):
#	t_psds[ifos[i]] = tf.convert_to_tensor(psds[ifos[i]], dtype=tf.complex128)
#	t_psds[ifos[i]] = tf.slice(t_psds[ifos[i]], begin=[kmin], size=[kmax-kmin])


#create an array on disk that we will save the samples to.
if not os.path.exists(project_dir + "/" + fname):
	print("FILE DOES NOT EXIST, PROCESS {} IS CREATING IT".format(index))
	fp = np.memmap(project_dir + "/" + fname, dtype=np.complex64, mode='w+', 
				shape=(len(ifos),n_templates*len(params['mass1']), (seconds_before + seconds_after)*sample_rate), offset=128)
else:
	print("FILE EXISTS, PROCESS {} IS LOADING IT".format(index))
	fp = np.memmap(project_dir + "/" + fname, dtype=np.complex64, mode='r+', 
				shape=(len(ifos),n_templates*len(params['mass1']), (seconds_before + seconds_after)*sample_rate), offset=128)

#detectors = {'H1': Detector('H1'), 'L1': Detector('L1'), 'V1': Detector('V1'), 'K1': Detector('K1')}
print("file will have shape ", fp.shape)

##################################################calculate the SNR

print("finished loading data, starting SNR calculation")
import time 

allstart = time.time()

template_time = 0
template_load_time = 0
waveform_time = 0
SNR_time = 0
convert_time = 0
repeat_time = 0

import gc

from pycbc.waveform import get_td_waveform
all_detectors = {'H1': Detector('H1'), 'L1': Detector('L1'), 'V1': Detector('V1'), 'K1': Detector('K1')}

def get_projected_waveform_mp(args):
	
	hp, hc = get_td_waveform(mass1 = args['mass1'], mass2 = args['mass2'], 
							 spin1z = args['spin1z'], spin2z = args['spin2z'],
							 inclination = args['i'], distance = args['d'],
							 approximant = td_approximant, f_lower = f_lower, delta_t = delta_t)
	
	waveforms = np.empty(shape=(len(ifos), len(hp)))

	for detector in ifos:
		f_plus, f_cross = all_detectors[detector].antenna_pattern(
			right_ascension=args['ra'], declination=args['dec'],
			polarization=args['pol'],
			t_gps=args['gps'][0])
		
		detector_signal = f_plus * hp + f_cross * hc

		detector_index = ifos.index(detector)
		waveforms[detector_index] = detector_signal

	return waveforms

def run_batch(n):
	#file_idx = templates_per_file * (np.ravel(template_ids[n:n+samples_per_batch])//templates_per_file)
	#template_idx = np.ravel(template_ids[n:n+samples_per_batch]) % templates_per_file

	t_ids = np.ravel(template_ids[n:n+samples_per_batch])
	batch_template_params = templates[t_ids]

	t_templates = np.empty((n_templates * samples_per_batch, kmax-kmin), dtype=np.complex128)
	#start = time.time()
	
	for i in range(n_templates * samples_per_batch):
		t_templates[i] = get_fd_waveform(mass1 = batch_template_params[i,1], mass2 = batch_template_params[i,2], 
				   		spin1z = batch_template_params[i,3], spin2z = batch_template_params[i,4],
						approximant = fd_approximant, f_lower = f_lower, delta_f = delta_f, f_final = f_final)[0].data[kmin:kmax]
		  				#spin1z = 0, spin2z = 0,

	#create this batch's strains
	#TODO: optimise memory usage. we're creating the waveform and strain arrays separately, which is inefficient.
	strains = {}
	for ifo in ifos:
		strains[ifo] = np.zeros((samples_per_batch, duration*sample_rate))

	for i in range(samples_per_batch):
		#print("sample:", n+i, "template", t_ids[i])
		if noise_type == "Gaussian":
			noise = np.zeros((len(ifos), duration*sample_rate))
			for j in range(len(ifos)):
				noise[j] = pycbc.noise.gaussian.noise_from_psd(duration*int(1/delta_t),delta_t,psds[ifos[j]],
						   seed=seed+n+i*len(ifos)+j)
		else:
			noise = fetch_noise_loaded(segments,duration,gps[n+i],sample_rate,paths)

		if params["injection"][n+i]:
			
			args = {'mass1': params['mass1'][n+i], 'mass2': params['mass2'][n+i],
	   				'spin1z': params['spin1z'][n+i], 'spin2z': params['spin2z'][n+i],	
					'i': params['i'][n+i], 'd': params['d'][n+i],
					'ra': params['ra'][n+i], 'dec': params['dec'][n+i],
					'pol': params['pol'][n+i], 'gps': params['gps'][n+i]}
			
			temp = get_projected_waveform_mp(args)
						
			for ifo in ifos:
				#this shouldn't be used, waveforms should be shorter than the noise.
				w_len = np.min([len(temp[ifos.index(ifo)]), duration*sample_rate//2])
				strains[ifo][i,duration*sample_rate//2 - w_len + offset: duration*sample_rate//2 + offset] = temp[ifos.index(ifo)]
				delta_t_h1 = all_detectors[ifo].time_delay_from_detector(other_detector=all_detectors[ifos[0]],
													right_ascension=params['ra'][n+i],
													declination=params['dec'][n+i],
													t_gps=params['gps'][n+i][0])

				strains[ifo][i] = np.roll(strains[ifo][i], round(delta_t_h1*sample_rate))

				strains[ifo][i] += noise[ifos.index(ifo)]
		else:
			#print("no injection in sample ", n+i)
			for ifo in ifos:
				strains[ifo][i] = noise[ifos.index(ifo)]
	#waveform_time += time.time() - start
	ret = {}
	for ifo in ifos:
		strain = [TimeSeries(strains[ifo][i], delta_t=delta_t) for i in range(samples_per_batch)]
		strain = [highpass(i,f_lower).to_frequencyseries(delta_f=delta_f).data for i in strain]

		strain = np.array(strain)[:,kmin:kmax]
		#strain = tf.convert_to_tensor(strain, dtype=tf.complex128)
		#strains[ifo] = strain

		#strain = np.array([strain])[:,kmin:kmax]
		strain_np = np.repeat(strain, n_templates, axis=0)

		x = numpy_matched_filter(strain_np, t_templates, psds[ifo], N, kmin, kmax, duration, delta_t = delta_t, flow = f_lower)
		ret[ifo] = x[:,len(x[0])//2-seconds_before*sample_rate+offset:len(x[0])//2+seconds_after*sample_rate+offset]

	return ret


for n in range(index*samples_per_file,(index+1)*samples_per_file,samples_per_batch*mp_batch):
	#print("batch:", n//samples_per_batch)
	#print(n)
	end = min(n+mp_batch*samples_per_batch, (index+1)*samples_per_file)

	print("starting batches",[j for j in range(n,end,samples_per_batch)])

	start = time.time()
	with mp.Pool(n_cpus) as p:
		results = p.map(run_batch, [j for j in range(n,end,samples_per_batch)])
		#results = p.map(run_batch, [j for j in range(n,min(n+mp_batch*samples_per_batch, samples_per_file),samples_per_batch)])
	template_time += time.time() - start

	#TODO: ensure we can handle the case where the number of samples is not divisible by samples_per_batch,
	#and where samples_per_batch*mp_batch is not divisible by samples_per_file

	for i in range(mp_batch):
		#t_templates, strains = results[i]
		for ifo in ifos:

			fp[ifos.index(ifo)][(i)*n_templates*samples_per_batch + n_templates*n:(i+1)*n_templates*samples_per_batch + n_templates*n] = results[i][ifo]

	#garbage collect
	del results
	gc.collect()


print("template time + waveform load + convert:", template_time)
print("SNR time:", SNR_time)
print("convert time:", convert_time)
print("repeat time:", repeat_time)
print("total time:", time.time() - allstart)


t_time = time.time() - allstart
print("it would take ", (25000 * t_time/(samples_per_file*total_jobs))/3600, "hours to process 25000 samples.")


fp.flush()

#memmap'd files don't have a header describing the shape of the array, so we add one here

header = np.lib.format.header_data_from_array_1_0(fp)
with open(project_dir + "/" + fname, 'r+b') as f:
	np.lib.format.write_array_header_1_0(f, header)

#pool.close()

print("done!")
#sanity check on the SNR values

"""
import matplotlib.pyplot as plt

for i in range(len(fp)):
	if np.min(np.max(np.abs(fp[i]), axis = 1)) < 0.1:
		print("SNR is too low. not all jobs have necessarily finished.")


	plt.plot(np.max(np.abs(fp[i]), axis = 1), alpha=0.5)

plt.savefig(project_dir + "/max_SNRs.png")

plt.clf()


if np.min(np.max(np.abs(fp[0]), axis = 1)[:n_signal_samples * n_templates]) > 0.1:
	print("all jobs should have finished. ")		
	for ifo in ifos:
		plt.hist(np.max(np.abs(fp[ifos.index(ifo)]), axis = 1)[:n_signal_samples *n_templates: n_templates]/ params[ifo+'_snr'][:n_signal_samples], bins=30, alpha=0.5)
	plt.xlabel("recovered/injected SNR")
	plt.savefig(project_dir + "/recovered_SNR.png")
"""