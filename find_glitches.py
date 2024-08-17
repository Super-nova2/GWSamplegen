import configparser
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, List
from GWSamplegen.mldatafind import authenticate
import igwn_auth_utils


import h5py
import numpy as np
from omicron.cli.process import main as omicron_main
#from omicron_utils import omicron_main_wrapper
import os

authenticate.authenticate()


def get_O3_week(week):
    """Returns the start and end times of the given week of O3."""
    start = 1238166018 + (week-1)*60*60*24*7
    end = start + 60*60*24*7
    return start, end

#Specify start and end times if you're running on a week in O3. Otherwise, specify your own start and end times.
start, end = get_O3_week(10)

ifos = ["H1", "L1"]

#GWOSC settings
frame_type = "HOFT_CLEAN_SUB60HZ_C01"
channel =  "DCS-CALIB_STRAIN_CLEAN_SUB60HZ_C01"
state_flag = "DMT-ANALYSIS_READY:1"

#directory to save the glitches
glitchdir='glitches_week10'


def omicron_main_wrapper(
    start: int,
    stop: int,
    run_dir: Path,
    q_min: float,
    q_max: float,
    f_min: float,
    f_max: float,
    sample_rate: float,
    cluster_dt: float,
    chunk_duration: int,
    segment_duration: int,
    overlap: int,
    mismatch_max: float,
    snr_thresh: float,
    frame_type: str,
    channel: str,
    state_flag: str,
    ifo: str,
    log_file: Path,
    verbose: bool,
):

    """Parses args into a format compatible for Pyomicron,
    then launches omicron dag
    """

    # pyomicron expects some arguments passed via
    # a config file. Create that config file

    if not run_dir.exists():
        run_dir.mkdir(parents=True)

    config = configparser.ConfigParser()
    section = "GW"
    config.add_section(section)

    config.set(section, "q-range", f"{q_min} {q_max}")
    config.set(section, "frequency-range", f"{f_min} {f_max}")
    config.set(section, "frametype", f"{ifo}_{frame_type}")
    config.set(section, "channels", f"{ifo}:{channel}")
    config.set(section, "cluster-dt", str(cluster_dt))
    config.set(section, "sample-frequency", str(sample_rate))
    config.set(section, "chunk-duration", str(chunk_duration))
    config.set(section, "segment-duration", str(segment_duration))
    config.set(section, "overlap-duration", str(overlap))
    config.set(section, "mismatch-max", str(mismatch_max))
    config.set(section, "snr-threshold", str(snr_thresh))
    # in an online setting, can also pass state-vector,
    # and bits to check for science mode
    #config.set(section, "state-flag", f"{ifo}:{state_flag}")

    config_file_path = run_dir / f"omicron_{ifo}.ini"

    # write config file
    with open(config_file_path, "w") as configfile:
        config.write(configfile)

    # parse args into format expected by omicron
    omicron_args = [
        section,
        "--log-file",
        str(log_file),
        "--config-file",
        str(config_file_path),
        "--gps",
        f"{start}",
        f"{stop}",
        "--ifo",
        ifo,
        "-C 60",
        "--max-concurrent",
        str(20),
        "-c",
        "request_disk=2GB",
        "-c",
        "request_memory=4096",
        "--output-dir",
        str(run_dir),
        "--skip-gzip",
        "--skip-rm"
    ]
    if verbose:
        omicron_args += ["--verbose"]

    # create and launch omicron dag
    omicron_main(omicron_args)

    # return variables necessary for glitch generation
    return ifo



def find_glitches(ifo):

    done = omicron_main_wrapper(
        start=start,
        stop=end,
        run_dir=Path("./{}/triggers_{}".format(glitchdir,ifo)),
        q_min=3.3166,
        q_max=150,
        f_min=18,
        f_max=1024,
        sample_rate=2048,
        cluster_dt=0.5,
        chunk_duration=124,
        segment_duration=64,
        overlap=4,
        mismatch_max=0.2,
        snr_thresh=6,
        frame_type=frame_type,
        channel=channel,
        state_flag=state_flag,
        ifo=ifo,
        log_file=Path("./log.log"),
        verbose=True
    )

    print("finished finding glitches for {}".format(ifo))

    #create a dictionary of the triggers

    print(Path("./{}/triggers_{}/merge/{}:{}/".format(glitchdir, ifo, ifo, channel)))
    #general path is config_dir/triggers_ifo/merge/ifo:channel
    trigger_dir = Path("./{}/triggers_{}/merge/{}:{}/".format(glitchdir, ifo, ifo, channel))
    print(trigger_dir)
    trigger_files = sorted(list(trigger_dir.glob("*.h5")))

    triggers = {}

    for i in range(len(trigger_files)):
        with h5py.File(trigger_files[i],'r') as f:
            for key in f['triggers'].dtype.names:
                if i == 0:
                    triggers[key] = np.array(f['triggers'][key])    
                else:
                    triggers[key] = np.concatenate((triggers[key],np.array(f['triggers'][key])))

    np.save("./{}/{}_glitches.npy".format(glitchdir, ifo),triggers)


for ifo in ifos:
    find_glitches(ifo)
