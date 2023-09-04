#!/bin/bash
#SBATCH --job-name=generate_configs
#SBATCH --output=./logs/%x.log
#SBATCH --cpus-per-task=20
#SBATCH --time=05:00:00
#SBATCH --mem=20gb

#module load gcc/10.3.0
#module load python/3.9.5
#module load cudnn/8.4.1.50-cuda-11.7.0

#source /fred/oz016/alistair/nt_env/bin/activate

source /home/amcleod/.bashrc

cd "/fred/oz016/alistair/GWSamplegen"

python generate_configs.py --config-file=args.json
#--config-file=configs/test1/args.json
