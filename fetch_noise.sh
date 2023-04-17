#!/bin/bash
#SBATCH --job-name=fetch_noise
#SBATCH --output=fetch_noise.log
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:10:00
#SBATCH --mem-per-cpu=10gb

source /fred/oz016/alistair/nt_env/bin/activate

cd /fred/oz016/alistair/GWSamplegen

echo starting job

python fetch_noise.py

