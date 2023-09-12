#this file will generate the config files for a project. 

#NEW PLAN: each 'config' directory will instead be a `project` directory containing:
#1. an args file, which defines the waveform args to use, as well as the noise and template bank directories
#2. directories for  associated SNR series + signal/noise files, and a params file containing all that directory's parameters
# This way we can generate multiple files with the same parameters without having to create an entirely new config each time.
# There should also be some params that can be changed on a per-file basis, such as seconds of EW


#TODO params to add:
#SEEDS
#waveform length?

#Other params that probably should be added in a DIFFERENT file:
#seconds before + after merger to slice SNR timeseries
#seconds of early warning
#duration of noise to fetch?

import argparse
import numpy as np
from pycbc.filter import sigma
from pycbc.detector import Detector
from pycbc.waveform import get_td_waveform
from pycbc.types import FrequencySeries
from pycbc.psd import interpolate
from utils.waveform_utils import choose_templates, load_pybc_templates, choose_templates_new
from utils.glitch_utils import get_glitchy_times, get_glitchy_gps_time
from utils.noise_utils import generate_time_slides, get_valid_noise_times, load_psd
import multiprocessing as mp
import json
import os

import time

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


#import args from a config file if it exists

parser = argparse.ArgumentParser()
parser.add_argument('--config-file', type=str, default=None)
args = parser.parse_args()
config_file = args.config_file


#start of user-defined params and param ranges. 

#seed for reproducibility
seed = 88102

#number of CPUS to use
n_cpus = 20

project_dir = "./configs/gaussian_test"
noise_dir = './noise/test'
template_bank = "PyCBC_98_aligned_spin"

#noise_type should be either Gaussian or Real. If Gaussian, it will use the PSD saved from the noise directory.
noise_type = "Gaussian"

#number of samples to generate
n_signal_samples = 10000
n_noise_samples = 0

#n_samples = 10000
#noise_frac = 0.5
glitch_frac = 0.0

approximant = "SpinTaylorT4"
f_lower = 18
delta_t = 1/2048

#If possible, make waveform_length a power of 2. This reduces error between pycbc and tensorflow in the SNR calculation.
waveform_length = 1024

seconds_before = 1
seconds_after = 1

detectors = ['H1','L1']

network_snr_threshold = 0
detector_snr_threshold = 4

#number of waveform templates to match with each waveform. These templates are taken from a distribution,
#but the first template chosen will be guaranteed to have a high overlap with the waveform.
templates_per_waveform = 1

template_approximant = "TaylorF2"
#when we select templates to match with the waveform, we select from a distribution of templates, but have to ensure
#that the template has at least some overlap with the waveform. for BBH signals you can use a width up to 0.05-0.1,
#but for BNS signals you should use a width of 0.02 or less as they are more sensitive to chirp mass.
template_selection_width = 0.01

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

mass1_min = 1.4#1.0
mass1_max = 1.4#2.6

mass2_min = 1.4
mass2_max = 1.4

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

#d_prior = UniformSourceFrame
d_prior = UniformComovingVolume

d_min = 10.0
d_max = 100.0

#if you want to force samples to have a certain effective distance, set d_eff_scaling to True and set the prior as desired.
#This overrides the network_snr_threshold and detector_snr_threshold parameters.
#If d_eff_scaling is set to False, the d_eff prior will be ignored and the d_eff will be based on the sample's extrinsic params.

d_eff_scaling = False
d_eff_target = "L1"

#d_eff_prior = UniformSourceFrame
d_eff_prior = UniformComovingVolume

d_eff_min = 250.0
d_eff_max = 250.0

#prior function for inclination. Should be Sine.
#should be 0 <= inc_min <= inc_max <= 1.

inc_prior = Sine

inc_min = 0.0
inc_max = 1.0

#prior function for polarization. Should be Uniform.
#Polarization is transformed from 0 <= x <= 1 to 0 <= x <= 2pi
#should be 0 <= pol_min <= pol_max <= 1.

pol_prior = Uniform

pol_min = 0.0
pol_max = 1.0

#if there is a config file specified, overwrite the above params with those from the config file
if config_file:
    print("loading args from a config file")
    with open(config_file) as json_file:
        config = json.load(json_file)

        seed = config['seed']
        n_signal_samples = config['n_signal_samples']
        n_noise_samples = config['n_noise_samples']
        glitch_frac = config['glitch_frac']
        project_dir = config['project_dir']
        noise_dir = config['noise_dir']
        noise_type = config['noise_type']
        templates_per_waveform = config['templates_per_waveform']
        approximant = config['approximant']
        f_lower = config['f_lower']
        delta_t = config['delta_t']
        waveform_length = config['duration']
        seconds_before = config['seconds_before']
        seconds_after = config['seconds_after']
        detectors = config['detectors']
        network_snr_threshold = config['network_snr_threshold']
        detector_snr_threshold = config['detector_snr_threshold']
        powerlaw_alpha = config['powerlaw_alpha']
        mass1prior = eval(config['mass1prior'])
        mass2prior = eval(config['mass2prior'])
        mass1_min = config['mass1_min']
        mass1_max = config['mass1_max']
        mass2_min = config['mass2_min']
        mass2_max = config['mass2_max']
        spin1zprior = eval(config['spin1zprior'])
        spin2zprior = eval(config['spin2zprior'])
        spin1z_min = config['spin1z_min']
        spin1z_max = config['spin1z_max']
        spin2z_min = config['spin2z_min']
        spin2z_max = config['spin2z_max']
        ra_prior = eval(config['ra_prior'])
        dec_prior = eval(config['dec_prior'])
        ra_min = config['ra_min']
        ra_max = config['ra_max']
        dec_min = config['dec_min']
        dec_max = config['dec_max']
        d_prior = eval(config['d_prior'])
        d_min = config['d_min']
        d_max = config['d_max']
        d_eff_scaling = config['d_eff_scaling']
        d_eff_target = config['d_eff_target']
        d_eff_min = config['d_eff_min']
        d_eff_max = config['d_eff_max']
        inc_prior = eval(config['inc_prior'])
        inc_min = config['inc_min']
        inc_max = config['inc_max']
        pol_prior = eval(config['pol_prior'])
        pol_min = config['pol_min']
        pol_max = config['pol_max']
        
    for key, value in config.items():
        print(key, value)

waveforms_per_batch = n_signal_samples//10

if not os.path.exists(noise_dir):
    raise ValueError("Noise directory does not exist. Generate a directory of noise to use with this dataset.")

#check that these parameters are compatible with those from the noise directory
with open(noise_dir + '/args.json') as f:
    noise_args = json.load(f)
    for ifo in detectors:
        if ifo not in noise_args['detectors']:
            raise ValueError("""Noise directory does not contain all the specified detectors.
                             Check noise directory and config file.""")
   
    if noise_args['delta_t'] != delta_t:
        raise ValueError("""Noise delta_t does not match specified delta_t.
                             Check noise directory and config file.""")




if not os.path.exists(project_dir):
    os.mkdir(project_dir)

#ADDING TEMPLATE BANK GENERATION HERE

"""
tmass1_min = 1
tmass1_max = 3

tmass2_min = 1
tmass2_max = 3

q_min = 0.0
q_max = 1.0

#rather than selecting for spins, we scale the template spins by a factor. set to 0 for no spins, 1 for full spins
spin_scale = 1

#templates are stored in the form: chirp mass, mass 1, mass 2, spin 1z, spin 2z
templates = np.load("template_banks/GSTLal_templates.npy")

#select only templates that are within the specified range

templates = templates[(templates[:,1] >= tmass1_min) & (templates[:,1] <= tmass1_max) & (templates[:,2] >= tmass2_min) & 
    (templates[:,2] <= tmass2_max) & (templates[:,2]/templates[:,1] >= q_min) & (templates[:,2]/templates[:,1] <= q_max)]

templates[:,3] *= spin_scale
templates[:,4] *= spin_scale

#sort the templates by chirp mass
templates = templates[templates[:,0].argsort()]

#save the template waveform params and waveform generation args for future reference
np.save(project_dir+"/template_params.npy",templates)


#with open(main_dir+bank_dir+"/template_args.json", 'w') as fp:
#    json.dump(args, fp, sort_keys=False, indent=4)

print("Number of templates: ", len(templates))	


template_bank_params = np.load(project_dir+"/template_params.npy")
"""

template_bank_params, metricParams, aXis = load_pybc_templates(template_bank)

np.save(project_dir+"/template_params.npy",template_bank_params)

print("Number of templates: ", len(template_bank_params))	

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

if d_eff_scaling:
    prior['d_eff'] = constructPrior(d_eff_prior, d_eff_min, d_eff_max, name = 'luminosity_distance')


valid_times, _, _ = get_valid_noise_times(noise_dir,waveform_length)

print(len(valid_times), "GPS times available")


#load PSD 


psds = load_psd(noise_dir, waveform_length, detectors, f_lower, int(1/delta_t))

all_detectors = {'H1': Detector('H1'), 'L1': Detector('L1'), 'V1': Detector('V1'), 'K1': Detector('K1')}

def get_snr(args):
    
    hp, hc = get_td_waveform(mass1 = args['mass1'], mass2 = args['mass2'], 
                             spin1z = args['spin1z'], spin2z = args['spin2z'],
                             inclination = args['i'], distance = args['d'],
                             approximant = approximant, f_lower = f_lower, delta_t = delta_t)
    
    snrs = {}

    for detector in detectors:
        f_plus, f_cross = all_detectors[detector].antenna_pattern(
            right_ascension=args['ra'], declination=args['dec'],
            polarization=args['pol'],
            t_gps=args['gps'][0])
        
        detector_signal = f_plus * hp + f_cross * hc

        snr = sigma(htilde=detector_signal,
                    psd=interpolate(psds[detector], delta_f=detector_signal.delta_f),
                    low_frequency_cutoff=f_lower)
        
        snrs[detector] = snr

    return snrs

good_params = []

generated_samples = 0
iteration = 0

wavetime = 0

max_waveform = 200
SNR_thresh = 6

if noise_type == "Real":

    glitchless_times = {}
    glitchy_times = {}
    glitchy_freqs = {}

    for ifo in detectors:
        glitchy_times[ifo] = []
        glitchless_times[ifo] = []
        glitchy_freqs[ifo] = []

    for ifo in detectors:
        
        glitchy, glitchless, freq = get_glitchy_times("noise/test/{}_glitches.npy".format(ifo),
                                        waveform_length, valid_times, max_waveform, SNR_thresh, f_lower, seconds_before, seconds_after)
        
        glitchless_times[ifo] = glitchless
        glitchy_times[ifo] = glitchy
        glitchy_freqs[ifo] = freq


    #create timeslide generators
    min_separation = 3

    glitchless_generator = generate_time_slides([glitchless_times[ifo] for ifo in detectors], min_separation)
    one_glitch_generator = {}

    for ifo in detectors:
        time_list = [glitchy_times[i] if i == ifo else glitchless_times[i] for i in detectors]
        one_glitch_generator[ifo] = generate_time_slides(time_list, min_separation)



while generated_samples < n_signal_samples:
    #generate waveforms_per_file samples at a time, to avoid memory issues.

    p = prior.sample(waveforms_per_batch)

    #adding non-sampled args to the parameters
    p['gps'] = []

    #if we're using real noise, deal with glitches and timeslides
    
    for detector in detectors:
        p[detector + '_glitch'] = np.zeros(waveforms_per_batch, dtype = bool)

    if noise_type == "Real":
        for i in range(waveforms_per_batch):
            if np.random.uniform(0,1) < glitch_frac:
                glitchy_ifo = np.random.choice(detectors)
                p[glitchy_ifo + '_glitch'][i] = True
                glitch_time = list(next(one_glitch_generator[glitchy_ifo]))
                glitch_idx = np.where(glitch_time[detectors.index(glitchy_ifo)] == glitchy_times[glitchy_ifo])[0][0]
                #TODO: shift glitch based on template masses, rather than true masses?
                glitch_time[detectors.index(glitchy_ifo)] = get_glitchy_gps_time(valid_times, p['mass1'][i], p['mass2'][i], 
                                                                                glitch_time[detectors.index(glitchy_ifo)], glitchy_freqs[glitchy_ifo][glitch_idx])
                p['gps'].append(glitch_time)            
                #gps.append(next(one_glitch_generator[glitchy_ifo]))
            else:
                p['gps'].append(list(next(glitchless_generator)))
    
    #otherwise, just choose some random GPS times from valid_times
    #TODO: adapt code to work with Gaussian noise without requiring a directory of real noise
    else:
        p['gps'] = np.random.choice(valid_times, size = (waveforms_per_batch, len(detectors)))
    

    p['injection'] = np.ones(waveforms_per_batch, dtype = bool)

    if d_eff_scaling:
        #adjust distance to reach the desired effective distance
        d_effs = Detector(d_eff_target).effective_distance(p['d'], p['ra'], p['dec'], p['pol'], np.array(p['gps'])[:,0], p['i'])
        p['d'] = p['d']/(d_effs / p['d_eff'])

    else:
        #TODO: add effective distance for each detector
        p['d_eff'] = np.zeros(waveforms_per_batch)

    
    #turn dict of lists into a list of dicts (for multiprocessing)
    params = [{key: p[key][i] for key in p.keys()} for i in range(len(p['mass1']))]

    #get the SNRs of the samples
    start = time.time()
    with mp.Pool(processes = n_cpus) as pool:

        #snrs is a list of dicts, where each dict is {detector: snr}
        snrs = pool.map(get_snr, params)
        #mp_waveforms is a list of lists, where each list is [waveform, snrs]
        
    wavetime += time.time() - start
    
    #save only the waveforms with network SNR and detector SNRs above threshold.

    for i in range(len(snrs)):

        network_snr = np.sqrt(sum([snrs[i][detector]**2 for detector in snrs[i]]))

        if (network_snr > network_snr_threshold and all([snr > detector_snr_threshold for snr in snrs[i].values()])) or d_eff_scaling:
            #this sample is suitable, get it ready for saving

            #add the detector SNRs and network SNR as keys in params[i]
            params[i]['network_snr'] = network_snr
            for detector in detectors:
                params[i][detector + '_snr'] = snrs[i][detector]

            #ensure that mass2 < mass1
            if params[i]['mass2'] > params[i]['mass1']:
                params[i]['mass1'], params[i]['mass2'] = params[i]['mass2'], params[i]['mass1']

            #choose template waveform(s) for this sample, and add them to params[i]
            #params[i]['template_waveforms'] = choose_templates(template_bank_params, params[i], 
            #                                                   templates_per_waveform, template_selection_width)
            params[i]['template_waveforms'] = choose_templates_new(template_bank_params, metricParams, 
                                                                   templates_per_waveform, params[i]['mass1'], params[i]['mass2'], 
                                                                   params[i]['spin1z'], params[i]['spin2z'], aXis = aXis)

            good_params.append(params[i])

    generated_samples = len(good_params)
    if generated_samples <= waveforms_per_batch:
        if generated_samples/waveforms_per_batch < 0.5:
            print("WARNING: check your distance prior and SNR threshold! Only {}% of the samples meet the SNR threshold."\
                  .format(round(generated_samples/waveforms_per_batch*100)))
        else:
            print("SNR threshold looks good, {}% of samples meet the threshold.".format(round(generated_samples/waveforms_per_batch*100)))
    print(len(good_params))

print('done samples with injections')

#save the injection parameters to a file
#convert from a list of dictionaries to a dictionary of lists
if n_signal_samples > 0:
    good_params_dict = {key: np.array([good_params[i][key] for i in range(len(good_params))][:n_signal_samples]) for key in good_params[0].keys()}




#generate noise samples. most of the parameters aren't used, but the masses are used to choose the templates.

if n_noise_samples > 0:
    noise_p = prior.sample(n_noise_samples)
    noise_p['gps'] = []
    noise_p['injection'] = np.zeros(n_noise_samples, dtype = bool)
    noise_p['template_waveforms'] = np.random.randint(0, len(template_bank_params), size=(n_noise_samples,templates_per_waveform))
    templates = []
    noise_p['d_eff'] = np.zeros(n_noise_samples)

    #TODO: get the actual max SNR for the noise segment maybe?
    noise_p['network_snr'] = np.zeros(n_noise_samples)
    for detector in detectors:
        noise_p[detector + '_snr'] = np.zeros(n_noise_samples)
        noise_p[detector + '_glitch'] = np.zeros(n_noise_samples, dtype = bool)

    for i in range(n_noise_samples):

        if noise_p['mass2'][i] > noise_p['mass1'][i]:
            noise_p['mass1'][i], noise_p['mass2'][i] = noise_p['mass2'][i], noise_p['mass1'][i]


        if noise_type == "Real":
            if np.random.uniform(0,1) < glitch_frac:
                glitchy_ifo = np.random.choice(detectors)
                noise_p[glitchy_ifo + '_glitch'][i] = True
                glitch_time = list(next(one_glitch_generator[glitchy_ifo]))
                glitch_idx = np.where(glitch_time[detectors.index(glitchy_ifo)] == glitchy_times[glitchy_ifo])[0][0]
                glitch_time[detectors.index(glitchy_ifo)] = get_glitchy_gps_time(valid_times, noise_p['mass1'][i], noise_p['mass2'][i], 
                                                                                    glitch_time[detectors.index(glitchy_ifo)], glitchy_freqs[glitchy_ifo][glitch_idx])
                noise_p['gps'].append(glitch_time)            
                #noise_p['gps'].append(next(one_glitch_generator[glitchy_ifo]))
            else:
                noise_p['gps'].append(list(next(glitchless_generator)))
        
        else:
            noise_p['gps'] = np.random.choice(valid_times, size = (n_noise_samples, len(detectors)))

        #params = {key: noise_p[key][i] for key in noise_p.keys()}
        #templates.append(choose_templates(template_bank_params, params, templates_per_waveform, template_selection_width))

    #noise_p['template_waveforms'] = np.array(templates)

    if n_signal_samples > 0:
        for key in good_params_dict.keys():
            good_params_dict[key] = np.append(good_params_dict[key], noise_p[key], axis = 0)
    else:
        good_params_dict = noise_p

#np.save(project_dir+"/"+"noise_params.npy", noise_p)
np.save(project_dir+"/"+"params.npy", good_params_dict)

#save the arguments used to generate the parameters to a file

args = {"seed": seed,
            "n_signal_samples": n_signal_samples,
            "n_noise_samples": n_noise_samples,
            "glitch_frac": glitch_frac,
            "project_dir": project_dir,
            "noise_dir": noise_dir,
            "noise_type": noise_type,
            "templates_per_waveform": templates_per_waveform,
            "approximant": approximant,
            "f_lower": f_lower,
            "delta_t": delta_t,
            "duration": waveform_length,
            "seconds_before": seconds_before,
            "seconds_after": seconds_after,
            "detectors": detectors,
            "network_snr_threshold": network_snr_threshold,
            "detector_snr_threshold": detector_snr_threshold,
            "powerlaw_alpha": powerlaw_alpha,
            "mass1prior": mass1prior.__name__,
            "mass2prior": mass2prior.__name__,
            "mass1_min": mass1_min,
            "mass1_max": mass1_max,
            "mass2_min": mass2_min,
            "mass2_max": mass2_max,
            "spin1zprior": spin1zprior.__name__,
            "spin2zprior": spin2zprior.__name__,
            "spin1z_min": spin1z_min,
            "spin1z_max": spin1z_max,
            "spin2z_min": spin2z_min,
            "spin2z_max": spin2z_max,
            "ra_prior": ra_prior.__name__,
            "dec_prior": dec_prior.__name__,
            "ra_min": ra_min,
            "ra_max": ra_max,
            "dec_min": dec_min,
            "dec_max": dec_max,
            "d_prior": d_prior.__name__,
            "d_min": d_min,
            "d_max": d_max,
            "d_eff_scaling": d_eff_scaling,
            "d_eff_target": d_eff_target,
            "d_eff_prior": d_eff_prior.__name__,
            "d_eff_min": d_eff_min,
            "d_eff_max": d_eff_max,
            "inc_prior": inc_prior.__name__,
            "inc_min": inc_min,
            "inc_max": inc_max,
            "pol_prior": pol_prior.__name__,
            "pol_min": pol_min,
            "pol_max": pol_max}

#save args
with open(project_dir+"/"+"args.json", 'w') as f:
    json.dump(args, f, sort_keys=False, indent=4)

print("finished generating waveforms. time taken: " + str(wavetime/60) + " minutes")

import matplotlib.pyplot as plt

if n_signal_samples > 0:
    for detector in detectors:
        plt.hist(good_params_dict[detector+"_snr"][:n_signal_samples], bins = 100, alpha = 0.5)

    plt.xlabel("Injected SNR")
    #plt.xlim(0,30)
    plt.savefig(project_dir+"/injected_SNR.png")
