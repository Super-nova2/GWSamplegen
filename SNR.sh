#!/bin/bash
#SBATCH --job-name=SNR5
#SBATCH --output=SNR5.log
#SBATCH --cpus-per-task=10
#SBATCH --time=01:30:00
#SBATCH --mem=100gb
#SBATCH --gres=gpu:1
#SBATCH --array=0-14

module load gcc/10.3.0
module load python/3.9.5
module load cudnn/8.4.1.50-cuda-11.7.0

source /fred/oz016/alistair/nt_env/bin/activate

cd "/fred/oz016/alistair/GWSamplegen"

#python SNR_series.py
#python generate_configs.py

echo done

python asyncSNR.py --index=$SLURM_ARRAY_TASK_ID --totaljobs=$SLURM_ARRAY_TASK_COUNT --config-file=configs/train_glitch/args.json