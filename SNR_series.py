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

import multiprocessing as mp

#defining some configs. some of these need to come from config files in the future.
duration = 1024
delta_t = 1.0/2048
sample_rate = int(1/delta_t)
f_lower = 18.0
delta_f = 1/duration

ifos = ['H1', 'L1']

seconds_before = 1
seconds_after = 1
offset = 0

fname = 'test.npy'

config_dir = "./configs/test2"
noise_dir = "./noise/test"
template_dir = "./template_banks/BNS_lowspin"

waveforms_per_file = 100
templates_per_file = 1000

samples_per_batch = 10
samples_per_file = 100



real_offset = np.min((offset*sample_rate, duration//2))


###################################################load noise segments
valid_times, paths, files = get_valid_noise_times("noise/test/",0)
segments = load_noise("noise/test/")

params = np.load(config_dir + "/params.npy", allow_pickle=True).item(0)
template_ids = np.array(params['template_waveforms'])
gps = params['gps']
n_templates = len(template_ids[0])

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
	#TODO: not sure if inverse spectrum truncation is required. I think we do since a 4 second window size is used when creating the PSD.
	psds[ifos[i]] = inverse_spectrum_truncation(psds[ifos[i]], int(4 * sample_rate),
									low_frequency_cutoff=f_lower)
	psds[ifos[i]] = tf.convert_to_tensor(psds[ifos[i]], dtype=tf.complex128)
	psds[ifos[i]] = tf.slice(psds[ifos[i]], begin=[kmin], size=[kmax-kmin])


#create an array on disk that we will save the samples to.
fp = np.memmap(config_dir + "/" + fname, dtype=np.complex128, mode='w+', 
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

for n in range(0,samples_per_file, samples_per_batch):

	#load this batch's templates

	file_idx = templates_per_file * (np.ravel(template_ids[n:n+samples_per_batch])//templates_per_file)
	template_idx = np.ravel(template_ids[n:n+samples_per_batch]) % templates_per_file

	t_templates = []

	

	for i in range(n_templates * samples_per_batch):
		with np.load(template_dir + "/"+ str(file_idx[i]) +".npz") as data:
			start = time.time()
			template = np.zeros((duration*sample_rate))
			template[-len(data['arr_'+str(template_idx[i])]):] = data['arr_'+str(template_idx[i])][-(duration)*sample_rate:]
			template_load_time += time.time() - start

			start = time.time()
			template = TimeSeries(template, delta_t=delta_t).to_frequencyseries(delta_f=delta_f)
			template_time += time.time() - start
			
			template = tf.convert_to_tensor(template)
			template = tf.slice(template, begin=[kmin], size=[kmax-kmin])
			t_templates.append(template)

	t_templates = tf.stack(t_templates)

	#template_time += time.time() - start

	file_idx = waveforms_per_file * (np.arange(n,n+samples_per_batch)//waveforms_per_file)
	waveform_idx = np.arange(n,n+samples_per_batch) % waveforms_per_file

	start = time.time()

	#load this batch's strains
	strains = {}
	for ifo in ifos:
		strains[ifo] = np.zeros((samples_per_batch, duration*sample_rate))

	for i in range(samples_per_batch):
		print("sample:", n+i)
		noise = fetch_noise_loaded(segments,duration,gps[n+i],sample_rate,paths)

		with np.load(config_dir + "/"+str(file_idx[i])+".npz") as data:
			#waveform = np.zeros((samples_per_batch,duration * sample_rate))
			temp = data['arr_'+str(waveform_idx[i])]
			#this shouldn't be used, waveforms should be shorter than the noise.
			w_len = np.min([len(temp[0]), duration*sample_rate//2])

			for ifo in ifos:
				strains[ifo][i,duration*sample_rate//2 - w_len + offset: duration*sample_rate//2 + offset] = temp[ifos.index(ifo)][:]

				delta_t_h1 = detectors[ifo].time_delay_from_detector(other_detector=detectors['H1'],
													right_ascension=params['ra'][n+i],
													declination=params['dec'][n+i],
													t_gps=params['gps'][n+i])
				#print(delta_t_h1)
				strains[ifo][i] = np.roll(strains[ifo][i], round(delta_t_h1*sample_rate))

				strains[ifo][i] += noise[ifos.index(ifo)]
	
	waveform_time += time.time() - start
	#compute SNRs
	#gpuSNRs = {}
	

	for ifo in ifos:
		start = time.time()
		strain = [TimeSeries(strains[ifo][i], delta_t=delta_t) for i in range(samples_per_batch)]
		strain = [highpass(i,f_lower).to_frequencyseries(delta_f=delta_f).data for i in strain]
		waveform_time += time.time() - start

		strain = np.array(strain)
		with tf.device('/GPU:0'):
			start = time.time()
			strain = tf.convert_to_tensor(strain[:,kmin:kmax])
			convert_time += time.time() - start
			start = time.time()
			strain = tf.repeat(strain, n_templates, axis=0)
			repeat_time += time.time() - start
			start = time.time()
			x = array_matched_filter(strain, t_templates, psds[ifo], N, kmin, kmax, duration, delta_t = delta_t, flow = f_lower)
			SNR_time += time.time() - start

		#start = time.time()
		#strain = tf.convert_to_tensor(strain)[:,kmin:kmax]
		#strain = tf.repeat(strain, n_templates, axis=0)
		
		#x = array_matched_filter(strain, t_templates, psds[ifo], N, kmin, kmax, duration, delta_t = delta_t, flow = f_lower)
		#SNR_time += time.time() - start

		fp[ifos.index(ifo)][n*n_templates:(n+samples_per_batch)* n_templates] = x[:,len(x[0])//2-seconds_before*sample_rate+offset:len(x[0])//2+seconds_after*sample_rate+offset]
		#return x[:,len(x[0])//2-seconds_before*sample_rate+offset:len(x[0])//2+seconds_after*sample_rate+offset]
	
	print(tf.config.experimental.get_memory_info('GPU:0'))


print("template time:", template_time)
print("template load time:", template_load_time)
print("waveform time:", waveform_time)
print("SNR time:", SNR_time)
print("convert time:", convert_time)
print("repeat time:", repeat_time)
print("total time:", time.time() - allstart)

#snrlist = []

#mp_snr(n)


#with mp.Pool(10) as p:
#	snrlist = p.map(mp_snr, range(0,samples_per_file, samples_per_batch))

#	for i in range(samples_per_file//samples_per_batch):
#		fp[:,i*samples_per_batch*n_templates:(i+1)*samples_per_batch*n_templates] = snrlist[i]

fp.flush()

#memmap'd files don't have a header describing the shape of the array, so we add one here

header = np.lib.format.header_data_from_array_1_0(fp)
with open(config_dir + "/" + fname, 'r+b') as f:
	np.lib.format.write_array_header_1_0(f, header)

