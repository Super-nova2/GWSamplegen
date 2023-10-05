import numpy as np

from .waveform_utils import t_at_f



def get_glitchy_times(glitch_file, duration, valid_times, longest_waveform, SNR_cutoff = 5, freq_cutoff = 0, seconds_before = 1, seconds_after = 1):
	"""
	Given a list of valid times and a glitch file from find_glitches.py, return a list of glitchy times and glitchless times.
	"""
	#longest_waveform is the rough length of the longest waveform in seconds
	glitch_data = np.load(glitch_file, allow_pickle=True).item()

	glitch_array = np.array([glitch_data['time'], glitch_data['frequency'], glitch_data['snr'], 
				glitch_data['tstart'], glitch_data['tend'], glitch_data['fstart'], glitch_data["fend"]]).T

	# select only times with SNR and frequency above cutoff. It is important to consider the end frequency of the glitch, not just the peak frequency.
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

	#print(len(no_glitch))
	mask = np.ones(len(valid_times), dtype=bool)
	for i in range(len(valid_times)):
		if valid_times[i] in no_glitch:
			mask[i] = False

	glitchless_times = valid_times[mask]
	glitchy_times = glitch

	print("There are {} glitchy times and {} glitchless times in {}".format(len(glitchy_times), len(glitchless_times), glitch_file[-15:]))

	return glitchy_times, glitchless_times, frequency_list#, glitch_idxs


def get_glitchy_gps_time(valid_times, mass1, mass2, glitch_time, frequency):
	#move a glitchy sample's start time to ensure the peak frequency of the glitch will be in the SNR time series.
	#not relevant for most BBH samples, but for BNS and NSBH the glitch's peak frequency can be tens of seconds from the merger.
	t_offset = int(t_at_f(mass1,mass2,frequency))
	if glitch_time + t_offset in valid_times:
		return glitch_time + t_offset
	else:
		#print(int(t_offset), glitch_time- valid_times[np.argmin(np.abs(valid_times - (glitch_time + t_offset)))] )
		print("WARNING: CANNOT OFFSET GLITCH ENOUGH TO ENSURE GLITCH APPEARS IN SNR TIME SERIES")
		return valid_times[np.argmin(np.abs(valid_times - (glitch_time + t_offset)))]