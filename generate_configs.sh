#!/bin/bash
#SBATCH --job-name=generate_configs-10k-4000MPc-USF
#SBATCH --output=./logs/%x.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=72:00:00
#SBATCH --mem=50gb

#module load gcc/10.3.0
#module load python/3.9.5
#module load cudnn/8.4.1.50-cuda-11.7.0

# source /home/amcleod/.bashrc

ml gcc/11.3.0 openmpi/4.1.4 python/3.10.4 cudnn/8.4.1.50-cuda-11.7.0 git/2.36.0
source /fred/oz016/damon/envs/nt_310/bin/activate

# cd "/fred/oz016/alistair/GWSamplegen"
cd "/fred/oz016/damon/GWSamplegen"

python generate_configs.py --config-file=configs/test10k/args.json
#--config-file=configs/test1/args.json
