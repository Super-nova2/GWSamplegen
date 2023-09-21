"""
GWSamplegen 
====

GWSamplegen is a package for generating training datasets of gravitational wave signal to noise ratio (SNR) time series 
for training machine learning models. Its main functionalities are:
	
  1. `fetch_noise`: fetching real noise data from the gravitational wave open science center (GWOSC)
  2. `find_glitches`: finding glitches in the data using PyOmicron (currently only works on LIGO data grid clusters) 
  3. `generate_configs`: generating parameter files for injecting signals into noise
  4. `asyncSNR`: injecting signals from the parameter files into noise and computing SNR time series for each.
"""