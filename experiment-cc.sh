#!/bin/bash
#SBATCH --account=partial-obs # Account name
#SBATCH --job-name=vae-pu-cc # Job name
#SBATCH --mail-type=FAIL # Mail events # (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=adam.wawrzenczyk@ipipan.waw.pl # Where to send mail
#SBATCH --nodes=1 # Run all processes on a single node
#SBATCH --ntasks=1 # Run a single task
#SBATCH --gpus=1 # Run on single GPU
#SBATCH --mem=16gb # Memory requirement
#SBATCH --time=02:00:00 # Time limit days-hrs:min:sec
#SBATCH --partition=experimental
#SBATCH --output=log-%j.log # Standard output and error log

pwd; hostname; date

mkdir -p "log/$dataset/$training_mode/$label_frequency/$start_idx"

eval "$(../miniforge3/bin/conda shell.bash hook)"
conda activate vae-pu-env

echo "$dataset" "$training_mode" $label_frequency $start_idx
echo $SLURM_JOB_ID: "$dataset" "$training_mode" $label_frequency $start_idx >> all-experiments.log

python -u ./main.py --dataset "$dataset" --training_mode "$training_mode" --c $label_frequency --start_idx $start_idx --num_experiments 1  --label_shift_pi None 0.7 0.5 0.3 0.1 0.9 --case_control

