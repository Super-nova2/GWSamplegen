"""
============
Functions for fetching, loading and selecting noise segments.
============

`load_noise` loads a directory of noise into memory.

`get_valid_noise_times` is a multipurpose function for returning a list of GPS times to inject samples into, and a list of sorted file paths.

`load_psd` loads the PSD of each interferometer from a noise directory.

`fetch_noise_loaded` fetches noise segments from a list of noise files loaded into memory.

`generate_time_slides` generates a list of time slides from a list of noise segments (vital if you want to generate more than ~10000 samples).
"""

import os
import numpy as np
from typing import Iterator, List, Optional, Sequence, Tuple
import h5py
import matplotlib.pyplot as plt
from pycbc.types.timeseries import TimeSeries
from pycbc.types import FrequencySeries
from pycbc.psd import interpolate, inverse_spectrum_truncation
import json

import time


#TODO: handle arbitrary groups of interferometers. Assume each noise file in a dir has the same ifos
	
from GWSamplegen.waveform_utils import t_at_f

def load_gps_blacklist(f_lower, event_file = '../noise/segments_event_gpstimes.json'):
	with open(event_file) as f:
		events = json.load(f)['events']
	gps_blacklist = []
	for event in events:

		if events[event]['mass_1_source'] is not None and events[event]['mass_2_source'] is not None \
			and events[event]['redshift'] is not None and events[event]['GPS'] is not None:
			
			m1_det = events[event]['mass_1_source'] * (1 + events[event]['redshift'])
			m2_det = events[event]['mass_2_source'] * (1 + events[event]['redshift'])
			start = np.floor(events[event]['GPS'] - t_at_f(m1_det, m2_det, f_lower))
			end = np.ceil(events[event]['GPS'] + 1)
			gps_blacklist.append(np.arange(start, end))

	return np.sort(np.concatenate(gps_blacklist))



def get_valid_noise_times(
	noise_dir: str,
	noise_len: int,
	min_step: int = 1,
	start_time: int = None,
	end_time: int = None,
	blacklisting: bool = True,
	f_lower = 30
) -> (List[int], np.ndarray, List[str]):
	"""multipurpose function to return a list of valid start times, list of noise file paths and deconstructed file names 
	
	noise_dir: directory containing noise files
	noise_len: minimum length of noise segments to consider
	start_time: if specified, the start of the time window to consider. Otherwise, all noise in noise_dir will be used.
	end_time: if specified, the end of the time window to consider

	returns:

	valid_times: list of valid start times for noise segments
	paths: array of deconstructed file names, giving detector info, segment start time and duration
	file_list: list of noise file paths in chronological order
	"""

	valid_times = np.array([])
	
	#get all strain file paths from the noise directory, then extract their start time and duration
	paths = os.listdir(noise_dir)
	paths = [path.split("-") for path in paths if len(path.split("-")) == 3]

	#paths[0] is the interferometer list
	#paths[1] is the start time
	#paths[2] is the duration

	ifo_list = paths[0][0]
	
	valid_paths = []
	for path in paths:
		if int(path[2][:-4]) >= noise_len:
			if start_time is not None and end_time is not None:
				if int(path[1]) <= start_time and int(path[1]) + int(path[2][:-4]) - start_time >= noise_len:
					valid_paths.append(path)
					print("path valid, starts before", path)
				
				elif int(path[1]) >= start_time and int(path[1]) + int(path[2][:-4]) <= end_time:
					valid_paths.append(path)
					print("path valid, contained", path)

				
				elif int(path[1]) < end_time and int(path[1]) + int(path[2][:-4]) - end_time >= noise_len:
					
					valid_paths.append(path)
					print("path valid, ends after", path)

				else:
					pass
					#print("path not valid", path)
			
			else:
				valid_paths.append(path)

	paths = valid_paths
	for path in paths:
		path[1] = int(path[1])
		path[2] = int(path[2][:-4])

		#print(path[1], path[2])

		times = np.arange(path[1], path[1]+path[2] - noise_len, min_step)
		if path[1] + path[2] - noise_len not in times:

			if int((path[1] + path[2] - noise_len) - times[-1]) != 1:
				#this additional if condition is to solve the edge case of a 1 second noise segment.
				times = np.append(times, path[1] + path[2] - noise_len)
			
			else:
				print("ignoring a 1 second segment")

		valid_times = np.concatenate((valid_times,times))

		if start_time is not None and end_time is not None:
			valid_times = valid_times[(valid_times >= start_time) & (valid_times + noise_len <= end_time) ]
		
	#ensure the file paths are in chronological order
	paths = np.array(paths)
	paths = paths[np.argsort(paths[:,1])]

	valid_times = np.sort(valid_times)
	
	#remove any GPS times that are too close to detected events

	if blacklisting:
		
		gps_blacklist = load_gps_blacklist(f_lower, "../noise/segments/event_gpstimes.json")
		#gps_blacklist = np.loadtxt("/fred/oz016/alistair/GWSamplegen/noise/segments/gps_blacklist.txt")
		n_blacklisted = len(np.where(np.isin(valid_times, gps_blacklist-noise_len//2))[0])
		print("{} GPS times are too close to detected events and have been removed".format(n_blacklisted))
		valid_times = np.delete(valid_times, np.where(np.isin(valid_times, gps_blacklist-noise_len//2)))

	#reconstruct the file paths from the start times and ifo_list
	file_list = [noise_dir +"/"+ ifo_list +"-"+ path[1] +"-"+ path[2] +".npy" for path in paths]

	return valid_times, paths, file_list


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

	

def two_det_timeslide(detector_data, min_distance):
	used_combinations = set()

	data_lengths = [len(data) for data in detector_data]
	divisor = data_lengths[1] 
	min_length = min(data_lengths)
	max_combos = (min_length - (min_distance - 1)) * (min_length - min_distance)
	while True:
		idx = np.random.randint(0,np.prod(data_lengths))
		sample_indicies = (idx//divisor, idx%divisor)

		# Limits the number of possible samples we draw from the generator
		if len(used_combinations) == max_combos:
			print("No more unique combinations available.")
			return

		if abs(detector_data[0][sample_indicies[0]] - detector_data[1][sample_indicies[1]]) >= min_distance:
			if sample_indicies not in used_combinations:

				used_combinations.add(sample_indicies)
				#yield sample_indicies
				yield (detector_data[0][sample_indicies[0]], detector_data[1][sample_indicies[1]])


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


#PSD UTILS

def construct_noise_PSD(
	#segments: List[np.ndarray],
	noise_paths: List[str]
	#path: str
):
	"""Create an averaged PSD of noise segments. This way of doing things is fine so long as the noise is stationary
	(which it is provided the segments do not span longer than ~1-2 weeks.)
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

def load_psd(
	noise_dir: str,
	duration: int,
	ifos: List[str],
	f_lower: int,
	sample_rate: int
):
	with open(noise_dir+ '/args.json') as f:
		args = json.load(f)
		ifo_list = args['detectors']

	psd = np.load(noise_dir + "/psd.npy")
	psds = {}
	for i in range(len(ifos)):
		psds[ifos[i]] = FrequencySeries(psd[ifo_list.index(ifos[i])+1], delta_f = psd[0][1], dtype = np.complex128)
		psds[ifos[i]] = interpolate(psds[ifos[i]], delta_f= 1/(duration))
		psds[ifos[i]] = inverse_spectrum_truncation(psds[ifos[i]], int(4 * sample_rate),
										low_frequency_cutoff=f_lower)
		
	return psds