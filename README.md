# GWSamplegen

A library of tools and scripts for generating compact binary merger samples in real LIGO noise. Specialised for creating SNR time series datasets for use in deep learning.

The general process for creating datasets is as follows:

1. Download LIGO noise using `fetch_noise.py`. This script fetches open LIGO data and saves it in a specified directory. Currently, it only saves data when both the LIGO Hanford and Livingston detectors are active.
2. Identify glitches in the data using `find_glitches.py`. This is to specify a fraction of samples that will contain a glitch. The resulting .npy files should be placed in the corresponding noise directory.
3. (optional) Generate a template bank using pycbc_geom_aligned_bank. Still need to add the code for doing this correctly. Otherwise, use the provided BNS bank, or the all source type GstLAL bank.
4. Make an arguments file for specifying the parameter distributions of the dataset. The provided `args.json` file will generate the dataset used in our BNS detection paper.
5. Run  `generate_configs.sh`, then `SNR_np.sh` to generate the parameter file, then the SNR time series dataset.

## Installation

First, ensure you have a python virtual environment (version 3.10 or above). Next, clone this repository with:

```
git clone https://github.com/alistair-mcleod/GWSamplegen.git
```
then `cd` into GWSamplegen and run

```
pip install .
```

## Usage notes

While generating parameter files is relatively fast, creating the SNR time series datasets would take a couple of days without access to a compute cluster (ideally above 100 cores). The job submission scripts are currently written with Slurm syntax, and you may need to rewrite them if you're not using a Slurm-based cluster.


## Job Submission Scripts

This is a collection of templates of SLURM job submission files so that there is no issues with multiple users in future commits.

### `SNR.sh` - Run SNR generation job. Change the job array size as 

```
#!/bin/bash
#SBATCH --job-name=SNR
#SBATCH --output=./logs/%x_%a.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=06:00:00
#SBATCH --mem=110gb
#SBATCH --gres=gpu:1
#SBATCH --array=0-9

source <your bashrc or virtual environment>

cd "/path/to/GWSamplegen"

echo starting job

python asyncSNR.py --index=$SLURM_ARRAY_TASK_ID --totaljobs=$SLURM_ARRAY_TASK_COUNT --config-file=configs/your_config_file/args.json
```

### `generate_configs.sh` - Generate config file needed to generate dataset

```
#!/bin/bash
#SBATCH --job-name=generate_configs
#SBATCH --output=./logs/%x.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=05:00:00
#SBATCH --mem=20gb

#module load gcc/10.3.0
#module load python/3.9.5
#module load cudnn/8.4.1.50-cuda-11.7.0

source <your bashrc or virtual environment>

cd "/path/to/GWSamplegen"

python generate_configs.py --config-file=args.json
```