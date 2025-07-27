# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
GWSamplegen is a Python package for generating gravitational wave (GW) signal datasets for machine learning applications. It creates SNR (Signal-to-Noise Ratio) time series by injecting compact binary merger signals into real LIGO noise or Gaussian noise.

## Core Architecture

### Main Components
- **GWSamplegen/**: Core library modules
  - `noise_utils.py`: Noise loading, PSD calculation, time slide generation
  - `waveform_utils.py`: Template bank loading and selection using PyCBC
  - `glitch_utils.py`: Glitch detection and handling
  - `snr_utils_np.py`: NumPy-based matched filtering
  - `mldatafind/`: Data fetching utilities

### Key Workflows

#### 1. Data Generation Pipeline
```
fetch_noise.py → find_glitches.py → generate_configs.sh → SNR_np.sh
```

#### 2. Configuration Flow
- Use `args.json` as template for dataset parameters
- `generate_configs.py` creates parameter files
- `asyncSNR_np.py` generates final SNR datasets

## Development Commands

### Installation
```bash
pip install .  # Install package and dependencies
```

### Data Generation
```bash
# Step 1: Download LIGO noise
python fetch_noise.py

# Step 2: Generate parameter configs
python share/generate_configs.py --config-file=share/args.json

# Step 3: Generate SNR datasets (cluster)
./share/generate_configs.sh
./share/SNR_np.sh
```

### Testing Single Components
```bash
# Test noise loading
python -c "from GWSamplegen.noise_utils import load_noise; print('OK')"

# Test template loading
python -c "from GWSamplegen.waveform_utils import load_pycbc_templates; print('OK')"

# Test parameter generation
python share/generate_configs.py --config-file=share/args.json
```

## Key Parameters & Files

### Configuration Files
- `share/args.json`: Main configuration template for BNS datasets
- `GWSamplegen/segments/`: GPS segment files for different observing runs
- `template_banks/`: PyCBC template banks for waveform matching

### Critical Parameters
- `delta_t`: Time resolution (1/2048 = 0.00048828125 for 2048Hz)
- `duration`: Analysis window length (typically 1024s)
- `f_lower`: Lower frequency cutoff (typically 18-30Hz)
- `detectors`: List of interferometers (['H1', 'L1'])

## Memory & Performance

### SLURM Job Requirements
- **generate_configs.sh**: 20 CPUs, 20GB RAM, 5 hours
- **SNR_np.sh**: 10 CPUs, 110GB RAM, 1 GPU, 6 hours

### Memory Optimization
- `samples_per_batch=10` (limited by 2GB tensor constraint)
- `mp_batch=10` (parallel processing batches)
- Use memmap for large datasets to avoid memory issues

## Data Structure

### Noise Files
- Format: `{prefix}-{GPS_start}-{duration}.npy`
- Contains: Multi-detector strain data (shape: [n_detectors, n_samples])
- PSD: Saved as `psd.npy` in same directory

### Parameter Files
- `params.npy`: Dictionary with keys:
  - `mass1`, `mass2`, `spin1z`, `spin2z`: Intrinsic parameters
  - `ra`, `dec`, `d`, `i`, `pol`: Extrinsic parameters
  - `gps`: Injection GPS times
  - `template_waveforms`: Template indices for matching

### Output Files
- `SNR_abs.npy`: Absolute SNR time series (shape: [n_detectors, n_samples, n_times])

## Common Issues & Solutions

### GPS Time Conflicts
- Use `load_gps_blacklist()` to avoid real GW events
- Check segment overlap with `combine_seg_list()`

### Memory Errors
- Reduce `samples_per_batch` if OOM errors occur
- Check available RAM matches SLURM specifications

### Template Selection
- BNS: Use `template_selection_width=0.01-0.02`
- BBH: Can use `template_selection_width=0.05-0.1`

## Dependencies
- PyCBC: Gravitational wave analysis
- Bilby: Bayesian inference and priors
- GWpy: Data handling
- Astropy: Cosmology calculations

## Testing Checklist
- [ ] Verify noise directory exists and has correct format
- [ ] Check template bank compatibility with parameter ranges
- [ ] Validate GPS times don't conflict with real events
- [ ] Ensure sufficient memory for batch processing
- [ ] Test single-threaded execution before scaling