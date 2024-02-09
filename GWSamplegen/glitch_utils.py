"""
============
Helper functions for inserting waveforms into glitchy noise
============

`get_glitchy_times` returns a list of glitchy times and glitchless times from a glitch file and a list of valid times.

`get_glitchy_gps_time` takes a list of valid times from `get_valid_noise_times` and a glitchy time from `get_glitchy_times` 
and offsets the glitchy time to ensure the glitch occurs in the SNR time series. This is necessary for BNS and NSBH samples,
as the peak frequency of the glitch can be tens of seconds from the merger.
"""


import numpy as np
from GWSamplegen.waveform_utils import t_at_f
from typing import Iterator, List, Optional, Sequence, Tuple
from pathlib import Path

def get_glitchy_times(
		glitch_file: Path, 
		duration: int,
		valid_times: List[int], 
		longest_waveform: int, 
		SNR_cutoff: float = 5.0, 
		freq_cutoff: float = 0.0, 
		seconds_before: int = 1, 
		seconds_after: int = 1
) -> (np.ndarray, np.ndarray, np.ndarray):
	"""
	Given a list of valid times and a glitch file from find_glitches.py, return a list of glitchy times and glitchless times.

	Parameters
	----------

	glitch_file: Path
		Path to a glitch file from find_glitches.py
	duration: int
		length of noise the waveforms are inserted into
	valid_times: List[int]
		List of valid times from get_valid_noise_times
	longest_waveform: int
		Length of the longest waveform in seconds
	SNR_cutoff: float
		Minimum SNR of glitches to consider
	freq_cutoff: float
		Minimum frequency of the glitches to consider
	seconds_before: int
		Number of seconds before the glitch to exclude from the glitchy times
	seconds_after: int
		Number of seconds after the glitch to exclude from the glitchy times

	Returns
	-------
	glitchy_times: np.ndarray
		List of glitchy times
	glitchless_times: np.ndarray
		List of glitchless times
	frequency_list: np.ndarray
		List of frequencies of the glitches

	"""
	glitch_data = np.load(glitch_file, allow_pickle=True).item()

	glitch_array = np.array([glitch_data['time'], glitch_data['frequency'], glitch_data['snr'], 
				glitch_data['tstart'], glitch_data['tend'], glitch_data['fstart'], glitch_data["fend"]]).T

	# select only times with SNR and end frequency above cutoff.
	glitch_array = glitch_array[(glitch_array[:,2] > SNR_cutoff) & (glitch_array[:,6] > freq_cutoff)]

	no_glitch = np.array([])
	glitch = np.array([])
	frequency_list = np.array([])
	glitch_idxs = []

	for i in range(len(glitch_array)):
		#exclude is not backwards, the higher the index the *earlier* the glitch appears in the data. longest_waveform therefore excludes higher indices.
		exclude = np.arange(int(glitch_array[i,3]-seconds_after - duration//2), int(glitch_array[i,4]+longest_waveform + seconds_before - duration//2))
		#TODO: expand include based on the length of the waveform. longer samples can be injected multiple times into a glitch.
		include = int(glitch_array[i,0] - duration//2 +1)

		no_glitch = np.hstack((no_glitch, exclude))

		if include in valid_times:
			glitch = np.hstack((glitch, include))
			#deal with the edge case where the peak glitch frequency is below the cutoff
			frequency_list = np.hstack((frequency_list, max(glitch_array[i,1], freq_cutoff))) 
			glitch_idxs.append(i)
		
	no_glitch = np.unique(no_glitch)
	
	glitchmask = np.zeros(len(valid_times), dtype=bool)

	mask = np.ones(len(valid_times), dtype=bool)
	for i in range(len(valid_times)):
		if valid_times[i] in no_glitch:
			mask[i] = False

	glitchless_times = valid_times[mask]
	glitchy_times = glitch

	print("There are {} glitchy times and {} glitchless times in {}".format(len(glitchy_times), len(glitchless_times), glitch_file[-15:]))

	return glitchy_times, glitchless_times, frequency_list

def get_glitchy_gps_time(
		valid_times: List[int],
		mass1: float,
		mass2: float, 
		glitch_time: int, 
		frequency: float
)-> int:
	"""
	Offset a glitchy time to ensure the glitch occurs in the SNR time series. This is necessary for BNS and NSBH samples,
	as the peak frequency of the glitch can be tens of seconds from the merger.

	Parameters
	----------

	valid_times: List[int]
		List of valid times from get_valid_noise_times
	mass1: float
		Primary mass of the waveform
	mass2: float
		Secondary mass of the waveform
	glitch_time: int
		Glitch time from get_glitchy_times

	Returns
	-------
	glitch_time: int
		Offset glitch time

	Warning
	-------
	If the glitch cannot be offset enough to ensure it appears in the SNR time series, 
	and a warning is printed and the closest valid time is returned.
	"""

	t_offset = int(t_at_f(mass1,mass2,frequency))
	if glitch_time + t_offset in valid_times:
		return glitch_time + t_offset
	else:
		print("WARNING: CANNOT OFFSET GLITCH ENOUGH TO ENSURE GLITCH APPEARS IN SNR TIME SERIES")
		return valid_times[np.argmin(np.abs(valid_times - (glitch_time + t_offset)))]