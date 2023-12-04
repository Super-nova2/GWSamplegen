import os
import h5py
import numpy as np
from gwpy.timeseries import TimeSeries, TimeSeriesDict
from GWSamplegen.mldatafind.find import find_data
from GWSamplegen.noise_utils import combine_seg_list, construct_noise_PSD, get_valid_noise_times
import json

#script to fetch noise data from GWOSC and save it to disk. Since this requires an internet connection, 
#you will have to run it on a node with internet access if submitting to a compute cluster.
#takes roughly 3 minutes per ifo per day of data, for example, downloading 2 days of H1 and L1 data takes ~12 minutes.


#note: in find.py, n_samples definition on line 28 needs to be: n_samples = int(n_channels * duration * sample_rate)
#some strange bug causes it to think it's a 32bit LIGO gps time

#For reference, these are the frame type, channel and state flag used for GWOSC open data.
frame_type = "HOFT_CLEAN_SUB60HZ_C01"
channel =  "DCS-CALIB_STRAIN_CLEAN_SUB60HZ_C01"
state_flag = "DMT-ANALYSIS_READY:1"



print("starting")


# write_dir = "/fred/oz016/alistair/GWSamplegen/noise/O3_first_week_1024"
write_dir = "/fred/oz016/damon/GWSamplegen/noise/O3_first_week_64"

#get current working directory
cwd = os.getcwd()

sample_rate = 2048
min_duration = 64

#create write_dir if it doesn't exist
if not os.path.exists(write_dir):
    os.mkdir(write_dir)
    

ifos = ["H1","L1"]


ifo_1 = '/fred/oz016/alistair/GWSamplegen/noise/segments/H1_O3a.txt'
ifo_2 = '/fred/oz016/alistair/GWSamplegen/noise/segments/L1_O3a.txt'


#start of O3: 1238166018
#second week of O3 start: 1238770818

# start = 1239375618
# end = start + 60*60*24*7
start = 1238166018
end = start + 60*60*24*7

#1239150592 is some time in O3, a bit after GW190425
#start = 1239150592
#end = start+5e5

segs, h1, l1 = combine_seg_list(ifo_1,ifo_2,start,end, min_duration=min_duration)


print(segs)

#data = find_data(segs, ifos)

#channelname = ":GDS-CALIB_STRAIN"
channelname = ""

#channelname = ":DCS-CALIB_STRAIN_CLEAN-SUB60HZ_C01"

channels = [ifo + channelname for ifo in ifos]


data = find_data(segs, channels)



for segment in data:
    segment.resample(sample_rate)

    #print(segment)
    t0 = int(segment[channels[0]].t0.value)
    length = len(segment[channels[0]])//sample_rate

    arr = np.zeros(shape=(len(ifos),len(segment[channels[0]])))
    prefix = ""

    for i in range(len(ifos)):
        arr[i] = segment[channels[i]]
        prefix += ifos[i][0]

    #at this point, we need to check for NaNs

    if np.any(np.isnan(arr)):
        print("NANS DEETECTED!")
        #split_segment(arr,t0,prefix)

    fname = write_dir + "/" + f"{prefix}-{t0}-{length}.npy"
    np.save(fname,arr)


print("finished downloading noise! now calculating PSDs.")
#after saving, we now need to calculate the PSDs.


_, _ , paths = get_valid_noise_times(write_dir, min_duration)


construct_noise_PSD(paths)





def split_segment(
    segment: np.ndarray,
    start_GPS: int,
    prefix: str
):
    """Segments of LIGO noise sometimes have sections where a detector's strain data is good according to the DQ bits, 
    but the detector is not operating. This function splits segments of noise that have these sections 
    of bad data and saves the good sections.
    
    NOTE: after changing how noise segments are selected, this function is probably no longer needed.
    However, it will be kept just in case."""

    if len(segment.shape) == 1:
        times_to_remove = np.isnan(segment)
    else:
        times_to_remove = np.any(np.isnan(segment), axis = 0)

    slice_idx = np.where(np.diff(times_to_remove) == 1)[0]
    slice_idx = np.add(slice_idx, 1)

    if not np.any(np.isnan(segment[:,0])):
        slice_idx = np.insert(slice_idx,0,0)

    if not np.any(np.isnan(segment[:,-1])):
        slice_idx = np.append(slice_idx,segment.shape[1])


    all_arrays = []

    for i in range(0,len(slice_idx),2):
        #first element is start of good section, element after is end of that good section etc.
        #first element gets rounded up to the nearest second, second element gets rounded down

        new_start = int(sample_rate * np.ceil(slice_idx[i]/sample_rate))
        new_end = int(slice_idx[i+1] - slice_idx[i+1]%sample_rate)
        print(new_start,new_end)

        #duration of this sub-segment
        new_duration = (new_end - new_start)//sample_rate
        if new_duration < min_duration:
            print("sub-segment too short!")
            continue
        
        array_to_save = segment[:,new_start:new_end]

        #start time of this sub-segment
        t0 = new_start//sample_rate + start_GPS

        fname = write_dir + "/" + f"{prefix}-{t0}-{new_duration}.npy"
        np.save(fname,array_to_save)


#save sample_rate, min_duration and ifos to a json file

args = {
    "delta_t": 1.0/sample_rate,
    "min_duration": min_duration,
    "detectors": ifos,
    "start_time": start,
    "end_time": end
}

with open(write_dir + "/args.json", "w") as f:
    json.dump(args, f, sort_keys=False, indent=4)

""" 
for ifo in ifos:
    #data = find_data(segs, ["H1","L1"])
    data = find_data(segs, [ifo])
    
    
    for segment in data:
        segment.resample(2048)
 """
"""
data = find_data(segs, ["L1"])
for segment in data:
    print(segment)

"""

