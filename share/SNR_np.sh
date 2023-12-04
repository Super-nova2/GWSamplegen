#!/bin/bash
#SBATCH --job-name=SNR
#SBATCH --output=./logs/%x_%a.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=10:00:00
#SBATCH --mem=160gb
#SBATCH --array=0-9

source /home/amcleod/.bashrc

cd "/fred/oz016/alistair/GWSamplegen"

echo starting job

#$SLURM_ARRAY_TASK_COUNT
python asyncSNR_np.py --index=$SLURM_ARRAY_TASK_ID --totaljobs=$SLURM_ARRAY_TASK_COUNT --config-file=args.json