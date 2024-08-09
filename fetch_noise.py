import os
import h5py
import numpy as np
import json
from gwpy.timeseries import TimeSeries, TimeSeriesDict
from GWSamplegen.mldatafind.find import find_data
from GWSamplegen.noise_utils import combine_seg_list, construct_noise_PSD, get_valid_noise_times

#script to fetch noise data from GWOSC and save it to disk. Since this requires an internet connection, 
#you will have to run it on a node with internet access if submitting to a compute cluster.
#takes roughly 3 minutes per ifo per day of data, for example, downloading 2 days of H1 and L1 data takes ~12 minutes.


#For reference, these are the frame type, channel and state flag used for GWOSC open data.
frame_type = "HOFT_CLEAN_SUB60HZ_C01"
channel =  "DCS-CALIB_STRAIN_CLEAN_SUB60HZ_C01"
state_flag = "DMT-ANALYSIS_READY:1"


print("starting")

week = 14

write_dir = "/fred/oz016/alistair/GWSamplegen/noise/O3_week_" + str(week)
#write_dir = "/fred/oz016/damon/GWSamplegen/noise/O3_fourth_week_64"

def get_O3_week(week):
    """Returns the start and end times of the given week of O3."""
    start = 1238166018 + (week-1)*60*60*24*7
    end = start + 60*60*24*7
    return start, end

start, end = get_O3_week(week)


#get current working directory
cwd = os.getcwd()

sample_rate = 2048
min_duration = 1024

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



#start = 1238166018
#end = start + 60*60*24*7

print(start, end)

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

    fname = write_dir + "/" + f"{prefix}-{t0}-{length}.npy"
    np.save(fname,arr)


print("finished downloading noise! now calculating PSDs.")
#after saving, we now need to calculate the PSDs.


_, _ , paths = get_valid_noise_times(write_dir, min_duration)


construct_noise_PSD(paths)

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