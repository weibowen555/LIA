#!/bin/bash
#SBATCH --partition=contrib-gpuq                    # need to set 'gpuq' or 'contrib-gpuq'  partition
#SBATCH --qos=gpu                           # need to select 'gpu' QOS or other relvant QOS
#SBATCH --job-name=ablation               # Job name
#SBATCH --output=/scratch/%u/neu_rec/%x-%N-%j.out   # Output file
#SBATCH --error=/scratch/%u/neu_rec/%x-%N-%j.err    # Error file
#SBATCH --nodes=1                           # Single node
#SBATCH --ntasks=1                          # Single task
#SBATCH --gres=gpu:A100.80gb:1              # Single GPU
#SBATCH --mem=80gb                          # memory per node
#SBATCH --export=ALL
#SBATCH --time=1-00:00:00                   # set to 1day; please choose carefully

set echo
umask 0027

# to see ID and state of GPUs assigned
nvidia-smi

## Load the necessary modules
module load gnu10
module load python/3.10.1-5r

source ~/pytorch-env-py3.10/bin/activate

## Execute your script (single GPU)
python3 main_1.py
