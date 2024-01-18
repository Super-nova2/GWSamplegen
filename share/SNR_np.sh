#!/bin/bash
#SBATCH --job-name=SNR
#SBATCH --output=./logs/%x_%a.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=2:00:00
#SBATCH --mem=50gb
#SBATCH --array=0
##SBATCH --array=0-9

# source /home/amcleod/.bashrc

ml gcc/11.3.0 openmpi/4.1.4 python/3.10.4 cudnn/8.4.1.50-cuda-11.7.0 git/2.36.0
source /fred/oz016/damon/envs/nt_310/bin/activate

# cd "/fred/oz016/alistair/GWSamplegen"
cd "/fred/oz016/damon/GWSamplegen"

echo starting job

#$SLURM_ARRAY_TASK_COUNT
python asyncSNR_np.py --index=$SLURM_ARRAY_TASK_ID --totaljobs=$SLURM_ARRAY_TASK_COUNT --config-file=configs/test50/args.json
