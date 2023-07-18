import os
import numpy as np
from typing import Iterator, List, Optional, Sequence, Tuple
import h5py
import matplotlib.pyplot as plt
from pycbc.types.timeseries import TimeSeries

import time
#length of noise the waveform is injected into. for BNS, I use 1000 seconds
noise_len = 1000

noise_path = "../real_noise/"


#TODO: handle arbitrary groups of interferometers. Assume each noise file in a dir has the same ifos

#function to retrieve valid start times for injecting gravitational wave samples into. 

def load_noise_paths(
	noise_dir: str
) -> List[str]:
	
	paths = os.listdir(noise_dir)
	paths = [path for path in paths if len(path.split("-")) == 3]
	paths = [noise_dir + path for path in paths]
	
	return paths
	
	
def get_valid_noise_times(
	noise_dir: str,
	noise_len: int
) -> List[int]:
	"""multipurpose function to return a list of valid start times, list of noise file paths and deconstructed file names """

	valid_times = np.array([])
	
	
	#get all strain file paths from the noise directory, then extract their start time and duration
	paths = os.listdir(noise_dir)
	paths = [path.split("-") for path in paths if len(path.split("-")) == 3]

	#paths[0] is the interferometer list
	#paths[1] is the start time
	#paths[2] is the duration

	ifo_list = paths[0][0]
	

	for path in paths:
		path[1] = int(path[1])
		path[2] = int(path[2][:-4])
		
		if path[2] <= noise_len:
			print("file length is shorter than desired noise segment length, skipping...")
			continue
		
		times = np.arange(path[1], path[1]+path[2] - noise_len)
		valid_times = np.concatenate((valid_times,times))
		
	#ensure the file paths are in chronological order
	paths = np.array(paths)
	paths = paths[np.argsort(paths[:,1])]

	valid_times = np.sort(valid_times)

	#reconstruct the file paths from the start times and ifo_list

	file_list = [noise_dir +"/"+ ifo_list +"-"+ path[1] +"-"+ path[2] +".npy" for path in paths]

	#paths = [noise_dir + path for path in paths]
	
	#now that we have all the noise times, we load them into a list 
	#np.random.shuffle(valid_times)
	
	#for each noise time, which file is it in and how far into the file is it?
	#print(paths)
	
	return valid_times, paths, file_list


def generate_time_slides(detector_data, min_distance):
	num_detectors = len(detector_data)
	data_lengths = [len(data) for data in detector_data]
	indices = [list(range(length)) for length in data_lengths]
	used_combinations = set()

	while True:
		min_length = min(data_lengths)
		# Limits the number of possible samples we draw from the generator
		if len(used_combinations) == (min_length - (min_distance - 1)) * (min_length - min_distance):
			print("No more unique combinations available.")
			return
		
		sample_indices = [np.random.choice(indices[i]) for i in range(num_detectors)]
		
		if all(abs(sample_indices[i] - sample_indices[j]) >= min_distance for i in range(num_detectors) for j in range(i+1, num_detectors)):
			combination = tuple(sample_indices)
			
			if combination not in used_combinations:
				used_combinations.add(combination)
				yield tuple(detector_data[i][sample_indices[i]] for i in range(num_detectors))

	
def load_noise_timeseries(    
	paths: np.ndarray
) -> List[np.ndarray]:
	
	noise_list = []
	for path in paths:
		#TODO: handle arbitrary groups of interferometers. Assume each noise file in a dir has the same ifos
		f = np.load(noise_path+"HL-"+str(path[0])+"-"+str(path[1])+".npy")

		noise_list.append(f)
		print("loaded a noise file")
		#noise_samples = np.concatenate((noise_samples,f['L1'][()]))
		
	return noise_list
	
def load_noise(noise_dir):
	_,_, fps = get_valid_noise_times(noise_dir,0)

	segments = []

	for fp in fps:
		segments.append(np.load(fp))

	return segments


def fetch_noise_loaded(    
	noise_list: List[np.ndarray],
	noise_len: int,
	noise_start_time: List[int],
	sample_rate: int,
	paths: np.ndarray
) -> np.ndarray[float]:

	"""Fetch an array of noise segments from a list of noise files.
	Supports timeslides by taking a list of start times.
	
	noise_list: list of noise files loaded into memory
	noise_len: length of noise segment to fetch
	noise_start_time: list of start times for noise segments
	sample_rate: sample rate of noise
	paths: array of noise file paths"""

	noises = np.empty(shape = (len(noise_start_time),noise_len*sample_rate))
	
	for i in range(len(noise_start_time)):
		f_idx = np.searchsorted(paths[:,1].astype('int'), noise_start_time[i],side='right') -1
		#to be able to fetch noise from ANY time, not just in integer steps, include sample_rate in the int()
		start_idx = int((noise_start_time[i] - paths[f_idx,1].astype('int'))*sample_rate)
		noises[i] = np.copy(noise_list[f_idx][i,start_idx:start_idx + noise_len * sample_rate])

	return noises




#utilities for downloading noise for later use

def overlapping_intervals(arr1, arr2):
	"""
	Find overlapping segments between two lists of segments from different
	detectors.

	Designed to be used with the outputs of get_seg_list as the inputs.

	Keyword arguments:
	arr1 -- segment list of first detector
	arr1 -- segment list of second detector
	"""
	res = []
	arr1_pos = 0
	arr2_pos = 0
	len_arr1 = len(arr1)
	len_arr2 = len(arr2)
	# //Iterate over all intervals and store answer
	while arr1_pos < len_arr1 and arr2_pos < len_arr2:
		arr1_seg = arr1[arr1_pos]
		arr2_seg = arr2[arr2_pos]

		# print(arr1_seg)
		# print(arr2_seg)

		# arr1_seg fully inside of arr2_seg
		if arr1_seg[0] >= arr2_seg[0] and arr1_seg[1] <= arr2_seg[1]:
			res.append(arr1_seg)
			arr1_pos += 1

		# arr2_seg fully inside of arr1_seg
		elif arr2_seg[0] >= arr1_seg[0] and arr2_seg[1] <= arr1_seg[1]:
			res.append(arr2_seg)
			arr2_pos += 1

		# arr1_seg fully below arr2_seg
		elif arr1_seg[0] <= arr2_seg[0] and arr1_seg[1] <= arr2_seg[0]:
			arr1_pos += 1

		# arr2_seg fully below arr1_seg
		elif arr2_seg[0] <= arr1_seg[0] and arr2_seg[1] <= arr1_seg[0]:
			arr2_pos += 1

		# arr1_seg overlaps start of arr2_seg
		elif arr1_seg[0] <= arr2_seg[0] <= arr1_seg[1] <= arr2_seg[1]:
			res.append([arr2_seg[0], arr1_seg[1]])
			arr1_pos += 1

		# arr2_seg overlaps start of arr1_seg
		elif arr2_seg[0] <= arr1_seg[0] <= arr2_seg[1] <= arr1_seg[1]:
			res.append([arr1_seg[0], arr2_seg[1]])
			arr2_pos += 1

	if not res:
		return [[]]

	return res


def get_seg_list(file_name, macrostart, macroend):
	"""
	Get a list of segments from a single detector segment file bounded
	by a GPS time window

	Keyword arguments:
	file_name -- path to detector's segment list
	macrostart -- Start GPS time of overlap window
	macroend -- End GPS time of overlap window
	"""
	file = open(file_name, "r")

	good_segs = []

	for i in file.readlines():
		times = i.split(" ")
		if int(times[1]) < macrostart or int(times[0]) > macroend:
			continue
		else:
			good_segs.append([int(times[0]), int(times[1])])

	file.close()

	if good_segs[0][0] < int(macrostart):
		good_segs[0][0] = int(macrostart)
	if good_segs[-1][1] > int(macroend):
		good_segs[-1][1] = int(macroend)

	return good_segs


def combine_seg_list(file_h1, file_l1, macrostart, macroend, min_duration):
	"""
	Find overlapping segments between two detectors within a window
	defined by two GPS times.

	Keyword arguments:
	file_h1 -- path to H1 complete segment list
	file_l1 -- path to L1 complete segment list
	macrostart -- Start GPS time of overlap window
	macroend -- End GPS time of overlap window
	"""
	good_segs_h1 = get_seg_list(file_h1, macrostart, macroend)
	good_segs_l1 = get_seg_list(file_l1, macrostart, macroend)

	good_segs = overlapping_intervals(good_segs_h1, good_segs_l1)

	#remove segments shorter than min_duration
	good_segs = [x for x in good_segs if x[1] - x[0] > min_duration]

	return good_segs, good_segs_h1, good_segs_l1



def get_noise_PSD(
	#segments: List[np.ndarray],
	noise_paths: List[str]
	#path: str
):
	"""Create an averaged PSD of noise segments. This way of doing things is fine so long as the noise is stationary
	(which it is provided the segments do not span longer than ~1 week.)
	"""

	segments = []

	for noise_path in noise_paths:
		segments.append(np.load(noise_path))

	ifo_psds = []

	#iterate through interferometers
	for ifo in range(len(segments[0])):
		
		psds = []
		for i in range(len(segments)):
			#setting NaNs in segments to 0. This hopefully shouldn't be needed!
			segments[i][ifo][np.isnan(segments[i][ifo])] = 0
			seg = TimeSeries(segments[i][ifo],delta_t=1/2048)
			psd = seg.psd(4)
			psds.append(psd.data * len(segments[i][ifo]))
			
		if ifo == 0:
			ifo_psds.append(psd.sample_frequencies.data)
		
		total_length = 0
		for segment in segments:
			total_length += len(segment[ifo])

		psd_avg = np.sum(np.array(psds), axis = 0)/total_length
		ifo_psds.append(psd_avg)

	ifo_psds = np.array(ifo_psds)

	#return ifo_psds
	np.save(os.path.dirname(noise_paths[0]) + "/psd.npy", ifo_psds)