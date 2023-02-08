#!/bin/bash

#SBATCH -N 1

#nombre threads max sur GPU 48

#SBATCH -n 1


#SBATCH -J impair


#SBATCH --output="impair.out"
#SBATCH --mem 35000    # Memory per node in MB (0 allocates all the memory)

#SBATCH --ntasks=1              # Number of processes to run (default is 1)
#SBATCH --cpus-per-task=6      # CPU cores per process (default 1)

#SBATCH -p cbio-cpu


cd /cluster/CBIO/data1/data3/tdefard/autofish
conda activate rna_topo3_bigfish6

python main.py \
--rounds_folder /cluster/CBIO/data1/data3/tdefard/T7/2023-01-19-PAPER-20-rounds/round_impair \
--detect_beads 0 \
--compute_transform 0 \
--plot_beads 0 \
--detect_spots 1 \
--plot_spots 0 \
--pairing 0 \






