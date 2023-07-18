import os
import numpy as np
from typing import Iterator, List, Optional, Sequence, Tuple
from pycbc.filter import highpass, lowpass
from utils.noise_utils import get_valid_noise_times, load_noise, fetch_noise_loaded
#rom pycbc.filter import matched_filter
from pycbc.detector import Detector
from pycbc.psd import interpolate, inverse_spectrum_truncation
from pycbc.types import FrequencySeries
from pycbc.types.timeseries import TimeSeries
from utils.SNR_utils import array_matched_filter, tf_get_cutoff_indices
import tensorflow as tf
from pycbc.waveform import get_fd_waveform

#import asyncio
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, wait

import multiprocessing as mp


#Ideas for speedup:
#smaller sized template bank files. not sure this is necessary since we are using a memmap
#do ALL tf operations on GPU, including converting the templates to tensors.


#defining some configs. some of these need to come from config files in the future.
duration = 1024
delta_t = 1.0/2048
sample_rate = int(1/delta_t)
f_lower = 18.0
delta_f = 1/duration
f_final = duration
approximant = 'TaylorF2'

ifos = ['H1', 'L1']

seconds_before = 1
seconds_after = 1
offset = 0

fname = 'test.npy'

config_dir = "./configs/50k_signoise_8SNR_noglitches"
noise_dir = "./noise/test"
#template_dir = "./template_banks/BNS_lowspin_freqseries"

waveforms_per_file = 100
templates_per_file = 1000

#samples_per_batch is limited by how many samples we can fit into a 2 Gb tensor
#mp_batch is limited by the amount of memory available.

samples_per_batch = 100
samples_per_file = 50000

mp_batch = 10
n_cpus = 20

offset = np.min((offset*sample_rate, duration//2))


###################################################load noise segments
valid_times, paths, files = get_valid_noise_times(noise_dir,0)
segments = load_noise(noise_dir)

params = np.load(config_dir + "/params.npy", allow_pickle=True).item(0)
template_ids = np.array(params['template_waveforms'])
gps = params['gps']
n_templates = len(template_ids[0])

templates = np.load(config_dir + "/template_params.npy")

#Damon's definition of N. from testing, it's just the total length of the segment in samples
#N = (len(sample1)-1) * 2
N = int(duration/delta_t)
kmin, kmax = tf_get_cutoff_indices(f_lower, None, delta_f, N)



##################################################load PSD
psd = np.load(noise_dir + "/psd.npy")

#since psd[0] is the sample frequencies, and the first frequency is always 0 Hz, psd[0][1] is sample frequency
psds = {}

for i in range(len(ifos)):
	psds[ifos[i]] = FrequencySeries(psd[i+1], delta_f = psd[0][1], dtype = np.complex128)
	psds[ifos[i]] = interpolate(psds[ifos[i]], delta_f= 1/(duration))
	#TODO: not sure if inverse spectrum truncation is required. I think we do since a 4 second window size 
	#is used when creating the PSD.
	psds[ifos[i]] = inverse_spectrum_truncation(psds[ifos[i]], int(4 * sample_rate),
									low_frequency_cutoff=f_lower)
	psds[ifos[i]] = tf.convert_to_tensor(psds[ifos[i]], dtype=tf.complex128)
	psds[ifos[i]] = tf.slice(psds[ifos[i]], begin=[kmin], size=[kmax-kmin])


#create an array on disk that we will save the samples to.
fp = np.memmap(config_dir + "/" + fname, dtype=np.complex64, mode='w+', 
               shape=(len(ifos),n_templates*samples_per_file, (seconds_before + seconds_after)*sample_rate), offset=128)

detectors = {'H1': Detector('H1'), 'L1': Detector('L1'), 'V1': Detector('V1'), 'K1': Detector('K1')}

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

#def read_file_mp(args):
#	temp = np.load(template_dir + "/" + str(args[0]) +".npy", mmap_mode='r')
#	return np.copy(temp[args[1]][kmin:kmax])

#pool = mp.Pool(processes = 10)

#def read_file_mp2(args):
#	file_idx, template_ids, i = args
#	temp = np.load(template_dir + "/"+ str(file_idx) +".npy", mmap_mode='r')
#	ret = []
#	for id in template_ids:
#		ret.append(np.copy(temp[id][kmin:kmax]))
#	return (ret,i)

#def get_fd_waveform(args):
#    return get_fd_waveform(mass1 = args[1], mass2 = args[2], spin1z = args[3], spin2z = args[4],
#            approximant = approximant, f_lower = f_lower, delta_f = delta_f, f_final = f_final)[0].data



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
            			approximant = approximant, f_lower = f_lower, delta_f = delta_f, f_final = f_final)[0].data[kmin:kmax]

	
	#for i in range(n_templates * samples_per_batch):
	#	temp = np.load(template_dir + "/"+ str(file_idx[i]) +".npy",mmap_mode='r')
	#	t_templates.append(np.copy(temp[template_idx[i]][kmin:kmax]))

	#template_load_time += time.time() - start

	#t_templates = tf.convert_to_tensor(t_templates, dtype=tf.complex128)
	
	file_idx = waveforms_per_file * (np.arange(n,n+samples_per_batch)//waveforms_per_file)
	waveform_idx = np.arange(n,n+samples_per_batch) % waveforms_per_file

	#start = time.time()

	#create this batch's strains
	strains = {}
	for ifo in ifos:
		strains[ifo] = np.zeros((samples_per_batch, duration*sample_rate))

	for i in range(samples_per_batch):
		print("sample:", n+i)
		noise = fetch_noise_loaded(segments,duration,gps[n+i],sample_rate,paths)

		if params["injection"][n+i]:
			#TODO: speed up file loading, as the batch's waveforms should be in 1 or 2 files.
			with np.load(config_dir + "/"+str(file_idx[i])+".npz") as data:

				temp = data['arr_'+str(waveform_idx[i])]
						
			#changed temp to temp[i]
			for ifo in ifos:
				#this shouldn't be used, waveforms should be shorter than the noise.
				w_len = np.min([len(temp[ifos.index(ifo)]), duration*sample_rate//2])
				strains[ifo][i,duration*sample_rate//2 - w_len + offset: duration*sample_rate//2 + offset] = temp[ifos.index(ifo)]
				delta_t_h1 = detectors[ifo].time_delay_from_detector(other_detector=detectors[ifos[0]],
													right_ascension=params['ra'][n+i],
													declination=params['dec'][n+i],
													t_gps=params['gps'][n+i][0])

				strains[ifo][i] = np.roll(strains[ifo][i], round(delta_t_h1*sample_rate))

				strains[ifo][i] += noise[ifos.index(ifo)]
		else:
			print("no injection in sample ", n+i)
			for ifo in ifos:
				strains[ifo][i] = noise[ifos.index(ifo)]
	#waveform_time += time.time() - start

	for ifo in ifos:
		strain = [TimeSeries(strains[ifo][i], delta_t=delta_t) for i in range(samples_per_batch)]
		strain = [highpass(i,f_lower).to_frequencyseries(delta_f=delta_f).data for i in strain]

		strain = np.array(strain)[:,kmin:kmax]
		#strain = tf.convert_to_tensor(strain, dtype=tf.complex128)
		strains[ifo] = strain

	return (t_templates, strains)



for n in range(0,samples_per_file,samples_per_batch*mp_batch):
	#print("batch:", n//samples_per_batch)
	#print(n)
	#for i in range(0,samples_per_file,samples_per_batch*mp_batch):
	print("starting batches",[j for j in range(n,min(n+mp_batch*samples_per_batch, samples_per_file),samples_per_batch)])

	start = time.time()
	with mp.Pool(n_cpus) as p:
		results = p.map(run_batch, [j for j in range(n,min(n+mp_batch*samples_per_batch, samples_per_file),samples_per_batch)])
	template_time += time.time() - start

	#TODO: ensure we can handle the case where the number of samples is not divisible by samples_per_batch,
	#and where samples_per_batch*mp_batch is not divisible by samples_per_file

	for i in range(mp_batch):
		t_templates, strains = results[i]
		for ifo in ifos:
			with tf.device('/GPU:0'):
			
				start = time.time()
				if ifo == 'H1':
					t_templates = tf.convert_to_tensor(t_templates, dtype=tf.complex128)

				strain = tf.convert_to_tensor(strains[ifo])
				convert_time += time.time() - start	
				start = time.time()
				strain = tf.repeat(strains[ifo], n_templates, axis=0)
				repeat_time += time.time() - start

				start = time.time()
				x = array_matched_filter(strain, t_templates, psds[ifo], N, kmin, kmax, duration, delta_t = delta_t, flow = f_lower)
				SNR_time += time.time() - start

			fp[ifos.index(ifo)][(i)*n_templates*samples_per_batch + n_templates*n:(i+1)*n_templates*samples_per_batch + n_templates*n] = \
				x[:,len(x[0])//2-seconds_before*sample_rate+offset:len(x[0])//2+seconds_after*sample_rate+offset]

#nthreads = 10

#with ThreadPoolExecutor(max_workers=nthreads) as exc:
#	for n in range(0,samples_per_file, samples_per_batch):
#		print("batch:", n//samples_per_batch)
#		exc.submit(run_batch,n)


print("template time + waveform load + convert:", template_time)
print("SNR time:", SNR_time)
print("convert time:", convert_time)
print("repeat time:", repeat_time)
print("total time:", time.time() - allstart)


t_time = time.time() - allstart
print("it would take ", (25000 * t_time/samples_per_file)/3600, "hours to process 25000 samples.")

#snrlist = []

#mp_snr(n)


fp.flush()

#memmap'd files don't have a header describing the shape of the array, so we add one here

header = np.lib.format.header_data_from_array_1_0(fp)
with open(config_dir + "/" + fname, 'r+b') as f:
	np.lib.format.write_array_header_1_0(f, header)

#pool.close()