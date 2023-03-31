

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
#get PSD of segments, save with noise
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

msun = r"$M_{\odot}$"
mpc = "Mpc"
rad = "rad"

#from astropy.cosmology import Cosmology


#import bilby



def uniform_extrinsic() -> PriorDict:
    prior = PriorDict()
    prior["dec"] = Cosine()
    prior["ra"] = Uniform(0, 2 * np.pi)
    prior["theta_jn"] = Sine()
    prior["phase"] = Uniform(0, 2 * np.pi)
    return prior

def nonspin_bns() -> PriorDict:
    prior = uniform_extrinsic()
    prior["mass_1"] = Uniform(5, 100, unit=msun)
    prior["mass_2"] = Uniform(5, 100, unit=msun)
    prior["mass_ratio"] = Constraint(0, 1)
    prior["redshift"] = UniformSourceFrame(
        0, 0.5, name="redshift", cosmology=None
    )
    prior["psi"] = 0
    prior["a_1"] = 0
    prior["a_2"] = 0
    prior["tilt_1"] = 0
    prior["tilt_2"] = 0
    prior["phi_12"] = 0
    prior["phi_jl"] = 0

    detector_frame_prior = True
    return prior, detector_frame_prior


#alpha value to be used for power law priors
powerlaw_alpha = -3.0

#prior functions to use for component masses.
mass1prior = Uniform
mass2prior = Uniform

mass1_min = 1.0
mass1_max = 3.0

mass2_min = 1.0
mass2_max = 3.0

#prior functions to use for spins. 
spin1zprior = Uniform
spin2zprior = Uniform

spin1z_min = 0.0
spin1z_max = 0.0

spin2z_min = 0.0
spin2z_max = 0.0

#prior functions to use for right ascension and declination.
#RA and DEC take ranges 0 <= x <= 1, and are transformed to their proper ranges.

ra_prior = Uniform #RA should always be uniform
dec_prior = Cosine

ra_min = 0.0
ra_max = 1.0

dec_min = 0.0
dec_max = 1.0

#prior function for inclination. Should always be Cosine.

i_prior = Cosine

i_min = 0.0
i_max = 1.0


#prior function for distance. Should be either Uniform or UniformSourceFrame.

d_prior = UniformSourceFrame

d_min = 0.0
d_max = 1000





def constructPrior(
    prior: Union[Uniform, Cosine, UniformComovingVolume,PowerLaw], 
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


#intrinsic prior



intrinsicprior = PriorDict()

intrinsicprior['mass1'] = constructPrior(mass1prior, mass1_min, mass1_max)
intrinsicprior['mass2'] = constructPrior(mass2prior, mass2_min, mass2_max)
intrinsicprior['spin1z'] = constructPrior(spin1zprior, spin1z_min, spin1z_max)
intrinsicprior['spin2z'] = constructPrior(spin2zprior, spin2z_min, spin2z_max)


#extrinsic prior

extrinsicprior = PriorDict()

extrinsicprior['ra'] = constructPrior(ra_prior, ra_min * 2 * np.pi, ra_max * 2 * np.pi, boundary = 'periodic')
extrinsicprior['dec'] = constructPrior(dec_prior, np.pi * ra_min - np.pi/2, np.pi * ra_max - np.pi/2)

extrinsicprior['d'] = constructPrior(d_prior, d_min, d_max, name = 'luminosity_distance')
extrinsicprior['inc'] = constructPrior(i_prior, i_min * np.pi, i_max * np.pi)


