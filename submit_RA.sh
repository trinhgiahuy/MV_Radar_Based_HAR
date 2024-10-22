#!/bin/bash
#SBATCH --job-name=RA
#SBATCH --partition=dualcard
##SBATCH --partition=midcard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=12G
#SBATCH --time=48:00:00
#SBATCH --output=RA_%j.out

#If you are using your own custom venv, replace mine with yours. Otherwise, stick to this default. It has torch, transformers, accelerate and a bunch of others. I'm happy to add more common libraries
# source /local/transformers/bin/activate
source /mnt/slurm_nfs/$USER/vit_env/bin/activate

#Trust. If you're using anything from huggingface, leave these lines it. These don't affect your job at all anyway, so really...just leave it in.
#export TRANSFORMERS_CACHE=/local/cache
export HF_HOME=/local/cache
# export HUGGINGFACE_TOKEN='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'
#export SENTENCE_TRANSFORMERS_HOME=/local/cache

python3 main.py

# Deactivate the virtual environment
deactivate