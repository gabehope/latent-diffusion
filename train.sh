#!/bin/bash

#SBATCH --job-name=pcvae
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32000
#SBATCH --time=96:00:00
#SBATCH --gres gpu:1
#SBATCH --nodelist bird  

cd /home/hopej/PC-VAE-2025/latent-diffusion
/home/hopej/miniconda3/bin/python main.py \
    --base configs/latent-diffusion-vae/latent-diffusion-vae-f8.yaml \
    -t \
    --gpus 0, \
    --logdir /scratch/gabe/logs/pcvae