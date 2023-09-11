#utilities for generating, saving and loading waveforms. 
#Also deals with loading template banks and picking templates from the banks.

import numpy as np
import scipy.stats as st
import json
from pycbc.tmpltbank.option_utils import metricParameters
from pycbc.tmpltbank.coord_utils import get_point_distance

import h5py

def chirp_mass(m1,m2):
    return ((m1 * m2)**0.6)/ (m1 + m2)**0.2

def t_at_f(m1,m2,f):
	top = 5 * ((3e8)**5) * (((m1+m2)*2e30)**(1/3))
	bottom = (f**(8/3))*256*(np.pi**(8/3)) * ((6.67e-11)**(5/3)) *m1*m2 * 2e30 * 2e30
	
	return (top/bottom)

def f_at_t(m1,m2,t):
    top = 5 * ((3e8)**5) * (((m1+m2)*2e30)**(1/3))
    bottom = t*256*(np.pi**(8/3)) * ((6.67e-11)**(5/3)) *m1*m2 * 2e30 * 2e30
    
    return (top/bottom)**(3/8)



def load_pybc_templates(bank_name, template_dir = "template_banks", pnOrder = "threePointFivePN", f_lower = 30, f_upper = 1024, deltaF = 0.01):
    """Load a PyCBC template bank file, as well as the metric used to generate it.

    Parameters
    ----------
    template_dir : str
        Directory containing the template bank file.
    bank_name : str
        Name of the template bank file and the file containing the associated metricParams.
    pnOrder : str
        Post-Newtonian order used to generate the template bank.
    f_lower : float
        Lower frequency cutoff of the template bank. TODO: might not actually be used.
    f_upper : float
        Upper frequency cutoff of the template bank. Typically 1024 Hz.
    deltaF : float
        Frequency resolution of the template bank. Typically 0.01 Hz. 
        Note this delta F is not necessarily the same as the delta F used in the SNR time series.
    """

    templates = np.loadtxt(template_dir + "/" + bank_name + ".txt", delimiter=",")
    templates = templates[np.argsort(templates[:,0])]

    f = h5py.File(template_dir + "/" + bank_name + "_intermediate.hdf", "r")

    metricParams = metricParameters(pnOrder=pnOrder, fLow=f_lower, fUpper=f_upper, deltaF=deltaF)

    metricParams.evals = {metricParams.fUpper: f["metric_evals"][()]}
    metricParams.evecs = {metricParams.fUpper: f["metric_evecs"][()]}
    metricParams.evecsCV = {metricParams.fUpper: f["cov_evecs"][()]}
    
    return templates, metricParams



def choose_templates_new(templates, metricParams, n_templates, mass1, mass2, spin1z = 0, spin2z = 0, limit = 100):
    """ Choose a set of templates from a PyCBC template bank using the template's metric.

    Parameters
    ---------- 
    templates: array_like
        A list of templates to choose from. columns should be [chirp mass, mass1, mass2, spin1z, spin2z]
    metricParams: pycbc.tmpltbank.metricParameters that were generated using the same metric as the template bank
    n_templates: int
        Number of templates to choose.
    limit: int
        Maximum template index (sorted by distance) to consider. Templates are sleected randomly up to this limit."""
    mismatches = get_point_distance(templates[:,1:5].T,[mass1,mass2,spin1z,spin2z],metricParams, list(metricParams.evecsCV.keys())[0])[0]

    #get the template indexes sorted by distance
    mismatches = np.argsort(mismatches)
    #np.argsort(mismatches)[::skip][:n_templates]

    #always return the best template first
    ret = [mismatches[0]]
    #append a random selection of the rest
    ret.extend(np.random.choice(mismatches[1:limit], size = n_templates - 1, replace = False))
    return ret




def errfunc(mass1,mass2,m1true,m2true):
    #function for choosing a template which will produce a good match between the template and true waveform

    return np.abs(mass2/mass1 - m2true/m1true) + 1000*np.abs(chirp_mass(mass1,mass2) -chirp_mass(m1true,m2true))


def choose_templates(template_bank_params, waveform_params, templates_per_waveform, template_selection_width):

    mass1,mass2 = waveform_params['mass1'],waveform_params['mass2']
    cm = chirp_mass(mass1,mass2)

    t_mass1,t_mass2 = template_bank_params[:,1], template_bank_params[:,2]

    #given a loaded array of template params and a waveform param, return the closest template.

    best_template =  np.argsort(errfunc(mass1,mass2,t_mass1,t_mass2))[0]

    #selecting a template range
    low_idx = np.searchsorted(template_bank_params[:,0],cm*(1-template_selection_width/2))
    high_idx = np.searchsorted(template_bank_params[:,0],cm*(1+template_selection_width/2))

    #choosing some suboptimal templates from a normal distribution, and 1 optimal template.

    x = np.arange(low_idx, high_idx)

    #print(waveform_params['mass1'],waveform_params['mass2'],np.min(x),np.max(x),best_template)

    # Sigma values for the two sides of the split distribution
    # Here, the 2 refers to the number of standard deviations either side of the peak

    sigma_low = int((best_template - low_idx) / 2)
    sigma_high = int((high_idx - best_template) / 2)

    diff = best_template - low_idx
    # Calculate separate PDFs for below and above the best template

    pdf1 = st.truncnorm.pdf(x, (low_idx - best_template) / sigma_low, (high_idx - best_template) / sigma_low, loc=best_template, scale=sigma_low)
    pdf2 = st.truncnorm.pdf(x, (low_idx - best_template) / sigma_high, (high_idx - best_template) / sigma_high, loc=best_template, scale=sigma_high)

    # Rescale each pdf to 50% and concatenate them together

    scale1 = 0.5 / pdf1[:diff].sum()
    scale2 = 0.5 / pdf2[diff:].sum()
    pdf3 = np.concatenate((pdf1[:diff]*scale1, scale2*pdf2[diff:]))

    # This is where we sample the number of tempaltes we want
    #if there are nans, we are near the edge of our template bank.
    #we instead use the pdf from only one side.

    if scale1 == np.inf or np.isnan(scale1):
        print("Left PDF is nan, using right PDF")
        if len(x) < templates_per_waveform:
            x = np.arange(len(template_bank_params)-templates_per_waveform,len(template_bank_params)-1)
        else:
            x = np.random.choice(x[diff:], size=templates_per_waveform-1, p=pdf2[diff:]*scale2*2, replace=False)
    elif scale2 == np.inf or np.isnan(scale2):
        print("Right PDF is nan, using left PDF")
        x = np.random.choice(x[:diff], size=templates_per_waveform-1, p=pdf1[:diff]*scale1*2, replace=False)
    else:
        if len(x) < templates_per_waveform:
            x = np.arange(len(template_bank_params)-templates_per_waveform,len(template_bank_params)-1)
        else:
            pdf3 = np.concatenate((pdf1[:diff]*scale1, scale2*pdf2[diff:]))
            x = np.random.choice(x, size=templates_per_waveform-1, p=pdf3, replace=False)
    #x = np.random.choice(x, size=templates_per_waveform-1, p=pdf3, replace=False)

    x = np.sort(x)
    x = np.insert(x,0,best_template)

    #making sure the templates are all unique
    for i in range(1,len(x)-1):
        if x[i] >= x[i+1]:
            x[i+1] = x[i] + 1
    
    #making sure the templates are all within the template bank
    if x[-1] >= len(template_bank_params):
        x -= x[-1] - len(template_bank_params) -1

    return x


def load_waveform(directory, index):
    #read json for waveforms_per_file

    with open(directory + '/args.json') as f:
        waveforms_per_file = json.load(f)['waveforms_per_file']

    file_index = waveforms_per_file * (index // waveforms_per_file)
    waveform_index = index % waveforms_per_file

    with np.load(directory + str(file_index) + '.npz',mmap_mode='r') as data:
        x = data['arr_'+str(waveform_index)]
        
    return x 