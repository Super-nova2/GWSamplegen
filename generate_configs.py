#this file will generate the config files for a project. 

#to note: we can have multiple noise datasets, and these can be used by different 'projects'
#we can also have multiple template banks, which are also not necessarily unique to a project
#we therefore need to store metadata in the noise and template bank directories, and will need to do some form of
#checking to determine if the noise and template bank are compatible with the project


#workflow of this project:
#1. generate param file(s) inc. template param file.
#2. generate template waveforms
#3. generate noise files
#4. for now: generate strain etc and save
#in future: step 4. will be done on the fly



#other stuff to do:
#pyomicron!


#the different params needed:

#intrinsic:
#m1,m2, spin

#extrinsic:
#distance, SNR (mutually exclusive, you can choose which one you want), inclination, 

#noise:
#noise start times, 

#simulation:
#noise length, f_lower, sample rate, 

#TEMPLATE IDS
#SEEDS


import numpy as np
from pycbc.filter import sigma
from pycbc.detector import Detector
from pycbc.waveform import get_td_waveform
from pycbc.psd import interpolate
from utils.waveform_utils import choose_templates
import multiprocessing as mp
import h5py
import json
import os

from bilby.core.prior import (
    ConditionalPowerLaw,
    ConditionalPriorDict,
    Constraint,
    Cosine,
    Gaussian,
    LogNormal,
    PowerLaw,
    PriorDict,
    Sine,
    Uniform,
)

from bilby.gw.prior import UniformComovingVolume, UniformSourceFrame

from typing import TYPE_CHECKING, Optional, Union


#SHIELD: SNR-based Highly Intelligent Early Latencyless Detection network

#start of user-defined params and param ranges. 

#seed for reproducibility

seed = 8810235

#number of CPUS to use
n_cpus = 20

project_dir = "./configs/test"
noise_dir = './noise/test'
template_bank_dir = './template_banks/BNS_lowspin'

#number of samples to generate
n_signal_samples = 1000
n_noise_samples = 1000


approximant = "SpinTaylorT4"
f_lower = 18
delta_t = 1/2048
waveform_length = 1000

waveforms_per_file = 100

detectors = ['H1','L1']

network_snr_threshold = 6
detector_snr_threshold = 4

#number of waveform templates to match with each waveform. These templates are taken from a distribution,
#but the first template will be guaranteed to have a high overlap with the waveform.
templates_per_waveform = 10

################################################
#---------------INTRINSIC PARAMS---------------#
################################################


#alpha value to be used for power law priors
powerlaw_alpha = -3.0

#Prior functions to use for component masses.
#For an astrophysical BBH distribution, use Powerlaw
#For a BNS distribution, use Uniform (TODO implement Gaussian)
mass1prior = Uniform
mass2prior = Uniform

mass1_min = 1.0
mass1_max = 2.6

mass2_min = 1.0
mass2_max = 2.6

#prior functions to use for spins. 
spin1zprior = Uniform
spin2zprior = Uniform

spin1z_min = 0.0
spin1z_max = 0.0

spin2z_min = 0.0
spin2z_max = 0.0


################################################
#---------------EXTRINSIC PARAMS---------------#
################################################


#prior functions to use for right ascension and declination.
#RA is transformed from 0 <= x <= 1 to 0 <= x <= 2pi
#DEC is transformed from 0 <= x <= 1 to -pi/2 <= x <= pi/2

ra_prior = Uniform #RA should always be uniform
dec_prior = Cosine

ra_min = 0.0
ra_max = 1.0

dec_min = 0.0
dec_max = 1.0

#prior function for distance. Should be either Uniform, UniformSourceFrame or UniformComovingVolume.

d_prior = UniformComovingVolume

d_min = 10.0
d_max = 100

#prior function for inclination. Should be Sine.

inc_prior = Sine

inc_min = 0.0
inc_max = 1.0

#prior function for polarization. Should be Uniform.
#Polarization is transformed from 0 <= x <= 1 to 0 <= x <= 2pi

pol_prior = Uniform

pol_min = 0.0
pol_max = 1.0



#check that the noise directory exists


#if not os.path.exists(noise_dir):
#    raise ValueError("Noise directory does not exist. Generate a directory of noise to use with this dataset.")

#check that these parameters are compatible with those from the noise directory
#with open(noise_dir + '/args.json') as f:
#    noise_args = json.load(f)
#    for ifo in detectors:
#        if ifo not in noise_args['detectors']:
#            raise ValueError("""Noise directory does not contain all the specified detectors.
#                             Check noise directory and config file.""")
    
#    if noise_args['delta_t'] != delta_t:
#        raise ValueError("""Noise delta_t does not match specified delta_t.
#                             Check noise directory and config file.""")

#check that the template bank directory exists

if not os.path.exists(template_bank_dir):
    raise ValueError("Template bank directory does not exist. Generate a template bank to use with this dataset.")

#check that these parameters are compatible with those from the template bank directory

with open(template_bank_dir + '/args.json') as f:
    template_bank_args = json.load(f)
    if template_bank_args['approximant'] != approximant:

        print("""Warning: template bank approximant is {} but specified approximant is {}"""\
                         .format(template_bank_args['approximant'], approximant))

    if template_bank_args['f_lower'] != f_lower:
        print("""Warning: template bank f_lower is {} but specified f_lower is {}"""\
                         .format(template_bank_args['f_lower'], f_lower))
    
    if template_bank_args['delta_t'] != delta_t:
        raise ValueError("""Fatal Error: template bank delta_t is {} but specified delta_t is {}"""\
                         .format(template_bank_args['delta_t'], delta_t))


template_bank_params = np.load(template_bank_dir + '/params.npy')




def constructPrior(
    prior: Union[Uniform, Cosine, UniformComovingVolume,PowerLaw,UniformSourceFrame], 
    min: float, 
    max: float,
    **kwargs
) -> PriorDict:
    #generic constructor for bilby priors. 
    
    if prior == PowerLaw:
        kwargs['alpha'] = powerlaw_alpha

    if max <= min:
        return max
    else:
        return prior(minimum = min, maximum = max, **kwargs)



#set a seed to ensure reproducibility
np.random.seed(seed)


prior = PriorDict()

prior['mass1'] = constructPrior(mass1prior, mass1_min, mass1_max)
prior['mass2'] = constructPrior(mass2prior, mass2_min, mass2_max)
prior['spin1z'] = constructPrior(spin1zprior, spin1z_min, spin1z_max)
prior['spin2z'] = constructPrior(spin2zprior, spin2z_min, spin2z_max)

prior['ra'] = constructPrior(Uniform, ra_min * 2 * np.pi, ra_max * 2 * np.pi, boundary = 'periodic')
prior['dec'] = constructPrior(dec_prior, np.pi * ra_min - np.pi/2, np.pi * ra_max - np.pi/2)

prior['d'] = constructPrior(d_prior, d_min, d_max, name = 'luminosity_distance')
prior['i'] = constructPrior(inc_prior, inc_min * np.pi, inc_max * np.pi)
prior['pol'] = constructPrior(pol_prior, pol_min * np.pi *2, pol_max * np.pi *2)



from utils.noise_utils import get_valid_noise_times
gps, _, _ = get_valid_noise_times(noise_dir,1000)
gps = np.random.permutation(gps)


#load PSD from noise_dir

psd = np.load(noise_dir + "/psd.npy")


from pycbc.types import FrequencySeries

psds = {}
psds["H1"] = FrequencySeries(psd[1], delta_f = 1.0/psd.shape[0], dtype = np.complex128)
psds["L1"] = FrequencySeries(psd[2], delta_f = 1.0/psd.shape[0], dtype = np.complex128)


all_detectors = {'H1': Detector('H1'), 'L1': Detector('L1'), 'V1': Detector('V1'), 'K1': Detector('K1')}

def get_projected_waveform_mp(args):
    
    hp, hc = get_td_waveform(mass1 = args['mass1'], mass2 = args['mass2'], 
                             spin1z = args['spin1z'], spin2z = args['spin2z'],
                             inclination = args['i'], distance = args['d'],
                             approximant = approximant, f_lower = f_lower, delta_t = delta_t)
    
    snrs = {}
    waveforms = np.empty(shape=(len(detectors), len(hp)))

    for detector in detectors:
        f_plus, f_cross = all_detectors[detector].antenna_pattern(
            right_ascension=args['ra'], declination=args['dec'],
            polarization=args['pol'],
            t_gps=args['gps'])
        
        delta_t_h1 = all_detectors[detector].time_delay_from_detector(
            other_detector=all_detectors['H1'],
            right_ascension=args['ra'],
            declination=args['dec'],
            t_gps=args['gps'])
        

        #print(len(f_plus),len(f_cross),len(hp),len(hc))
        
        detector_signal = f_plus * hp + f_cross * hc

        snr = sigma(htilde=detector_signal,
                    psd=interpolate(psds[detector], delta_f=detector_signal.delta_f),
                    low_frequency_cutoff=f_lower)
        
        snrs[detector] = snr

        detector_index = detectors.index(detector)
        waveforms[detector_index] = detector_signal

        #waveforms[detector] = detector_signal
        #print(snrs)

    return waveforms, snrs

good_waveforms = []
good_params = []

generated_samples = 0

iteration = 0


while generated_samples < n_signal_samples:

    #generate waveforms_per_file samples at a time, to avoid memory issues.

    p = prior.sample(waveforms_per_file)

    #adding the gps times to the parameters
    p['gps'] = gps[:waveforms_per_file]
    gps = gps[waveforms_per_file:]
    
    params = [{key: p[key][i] for key in p.keys()} for i in range(len(p['mass1']))]

    
    #generate the waveforms

    with mp.Pool(processes = n_cpus) as pool:
        mp_waveforms = pool.map(get_projected_waveform_mp, params)
        #mp_waveforms is a list of lists, where each list is [waveform, snrs]

        waveforms, snrs = zip(*mp_waveforms)
    
    #save only the waveforms with network SNR above threshold.
    #save in h5py files with associated parameters.

    for i in range(len(waveforms)):

        network_snr = np.sqrt(sum([snrs[i][detector]**2 for detector in snrs[i]]))

        if network_snr > network_snr_threshold and all([snr > detector_snr_threshold for snr in snrs[i].values()]):
            #this sample is suitable, get it ready for saving
            good_waveforms.append(waveforms[i])

            #add the detector SNRs and network SNR as keys in params[i]
            params[i]['network_snr'] = network_snr
            for detector in detectors:
                params[i][detector + '_snr'] = snrs[i][detector]

            #ensure that mass2 < mass1
            if params[i]['mass2'] > params[i]['mass1']:
                params[i]['mass1'], params[i]['mass2'] = params[i]['mass2'], params[i]['mass1']

            #choose template waveform(s) for this sample, and add them to params[i]
            params[i]['template_waveforms'] = choose_templates(template_bank_params, params[i], templates_per_waveform)

            good_params.append(params[i])
        else:
            print("discarding waveform with SNR " + str(network_snr))

    if iteration == 0 and len(good_waveforms)/waveforms_per_file < 0.5:
        print("WARNING: check your priors!" + str(len(good_waveforms)/waveforms_per_file) + 
              " of the samples meet the SNR threshold.")
    
    #now that we only have the good waveforms, we can save them to file.
    #we don't necessarily have waveforms_per_file samples in good_waveforms, so we need to check that.

    if len(good_waveforms) > waveforms_per_file:
        fname = project_dir+str(iteration*waveforms_per_file)+".npz"

        print(fname)

        temp = good_waveforms[:waveforms_per_file]
        good_waveforms = good_waveforms[waveforms_per_file:]
        np.savez(fname, *temp)
        
        #f = h5py.File(fname,'w')
        #for i in range(waveforms_per_file):
        #    #create group for each waveform
        #    x = f.create_group(str(i))
        #    for detector in detectors:
        #        x.create_dataset(detector, data = good_waveforms[i][detector])

        #f.close()

        generated_samples += waveforms_per_file
        iteration +=1


#save the parameters to a file
#convert from a list of dictionaries to a dictionary of lists

good_params_dict = {key: [good_params[i][key] for i in range(len(good_params))] for key in good_params[0].keys()}

np.save(project_dir+"params.npy", good_params_dict)