#!/bin/bash
#SBATCH --job-name=SNRtest
#SBATCH --output=./logs/%x_%a.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=20:00:00
#SBATCH --mem=150gb
#SBATCH --array=0-9

source /home/amcleod/.bashrc

cd "/fred/oz016/alistair/GWSamplegen"

echo starting job

#$SLURM_ARRAY_TASK_COUNT
python asyncSNR_np.py --index=$SLURM_ARRAY_TASK_ID --totaljobs=$SLURM_ARRAY_TASK_COUNT --config-file=configs/real_dsens_200_mpc/args.json