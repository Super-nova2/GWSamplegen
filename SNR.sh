#!/bin/bash
#SBATCH --job-name=SNR_gen
#SBATCH --output=generate_SNR.log
#SBATCH --cpus-per-task=20
#SBATCH --time=20:00:00
#SBATCH --mem=200gb
#SBATCH --gres=gpu:1

module load gcc/10.3.0
module load python/3.9.5
module load cudnn/8.4.1.50-cuda-11.7.0

source /fred/oz016/alistair/nt_env/bin/activate

cd "/fred/oz016/alistair/GWSamplegen"

#python SNR_series.py
python generate_configs.py

echo done

python asyncSNR.py