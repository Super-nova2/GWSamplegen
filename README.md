# GWSamplegen

Generate compact binary merger samples in real LIGO noise

## Job Submission Scripts

This is a collection of templates of SLURM job submission files so that there is no issues with multiple users in future commits.

### `SNR.sh` - Run data generation job

```
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
```

### `fetch_noise.sh` - Download noise data from GWOSC

```
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
```

### `generate_configs.sh` - Generate config file needed to generate dataset

```
#!/bin/bash
#SBATCH --job-name=generate_configs
#SBATCH --output=generate_configs.log
#SBATCH --cpus-per-task=20
#SBATCH --time=04:00:00
#SBATCH --mem=20gb

module load gcc/10.3.0
module load python/3.9.5
module load cudnn/8.4.1.50-cuda-11.7.0

source /fred/oz016/alistair/nt_env/bin/activate

cd "/fred/oz016/alistair/GWSamplegen"

python generate_configs.py
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
