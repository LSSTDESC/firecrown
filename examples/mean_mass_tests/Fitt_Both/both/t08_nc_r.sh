#!/usr/bin/bash
# !/bin/sh

# SLURM options:

# SBATCH --job-name=t08_r1    # Job name
# SBATCH --partition=htc               # Partition choice
# SBATCH --ntasks=3                    # Run a single task (by default tasks == CPU)
# SBATCH --mem=8G                    # Memory in MB per default
# SBATCH --time 20:00:00             # 7 days by default on htc partition
# SBATCH --mail-user=eduardojsbarroso@gmail.com
# SBATCH --mail-type=END,FAIL 

# Print the task and run range
cosmosis number_counts_rich.ini
