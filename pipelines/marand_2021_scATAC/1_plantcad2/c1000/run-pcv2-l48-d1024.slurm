#!/bin/bash

#SBATCH -J pcv2-l48-1024-scATAC
#SBATCH -o logs/%x-%j.log
#SBATCH -e logs/%x-%j.log
#SBATCH -p gh
#SBATCH -N 4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH -t 48:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jz963@cornell.edu

srun ./run-pcv2-l48-d1024.sh "$@"