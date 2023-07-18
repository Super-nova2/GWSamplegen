#utilities for generating, saving and loading waveforms

import numpy as np
import scipy.stats as st
import json

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