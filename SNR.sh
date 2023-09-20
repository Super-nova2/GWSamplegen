#!/bin/bash
#SBATCH --job-name=SNR2_fix
#SBATCH --output=./logs/%x_%a.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=03:30:00
#SBATCH --mem=30gb
#SBATCH --gres=gpu:1
#SBATCH --array=0-9

#module load gcc/10.3.0
#module load python/3.9.5
#module load cudnn/8.4.1.50-cuda-11.7.0

#source /fred/oz016/alistair/nt_env/bin/activate

source /home/amcleod/.bashrc

cd "/fred/oz016/alistair/GWSamplegen"

#python SNR_series.py
#python generate_configs.py

echo done

#$SLURM_ARRAY_TASK_COUNT
python asyncSNR.py --index=$SLURM_ARRAY_TASK_ID --totaljobs=10 --config-file=configs/gaussian_test_600mpc/args.json