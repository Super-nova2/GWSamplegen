# GWSamplegen

Generate compact binary merger samples in real LIGO noise

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


### `bank.sh` - Generate template bank waveforms (DEPRECATED)

<details><summary>Bank submission file</summary>
```
#!/bin/bash
#SBATCH --job-name=bank_gen
#SBATCH --output=generate_bank.log
#SBATCH --cpus-per-task=20
#SBATCH --time=04:00:00
#SBATCH --mem=100gb

module load gcc/10.3.0
module load python/3.9.5
module load cudnn/8.4.1.50-cuda-11.7.0

source /fred/oz016/alistair/nt_env/bin/activate
#source /fred/oz016/damon/nt_310/bin/activate

cd "/fred/oz016/alistair/GWSamplegen"

python generate_bank.py
```
</details>
