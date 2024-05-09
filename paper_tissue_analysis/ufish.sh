#!/bin/bash

#SBATCH -N 1

#nombre threads max sur GPU 48

#SBATCH -n 2

#SBATCH --cpus-per-task=2

#SBATCH -J ufish_cpu

#SBATCH --output="ufish_cpu.out"
#SBATCH --partition=cbio-cpu

#SBATCH --mem-per-cpu=90000

#  comment SBATCH --gres=gpu:1

## exclude node 6
#SBATCH --exclude=node006

module load cuda/11.3

python U_fish_spots_detection.py \
--path_to_round_folder "/cluster/CBIO/data1/data3/tdefard/autofish/data/2023-10-06_LUSTRA" \
--path_to_save_res "/cluster/CBIO/data1/data3/tdefard/autofish/data/2023-10-06_LUSTRA/detection_ufish" \


echo "ufish_gpu.sh done from slurm"