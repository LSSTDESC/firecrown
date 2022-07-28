#!/bin/bash 

#SBATCH -p regular
#SBATCH -L SCRATCH
#SBATCH -C haswell
#SBATCH -J Firecrown
#SBATCH --nodes=1
#SBATCH --tasks-per-node=61
#SBATCH --time=48:00:00
#SBATCH -o firecrown.log
#SBATCH -A m1727
#SBATCH --error=firecrown-%j.err
#SBATCH --mail-user=ayanmitra375@gmail.com


conda deactivate
module unload gsl/2.5
module unload cray-fftw/3.3.8.4
source $CFS/des/zuntz/cosmosis-global/setup-cosmosis-nersc
conda activate /global/u1/a/ayanmitr/soft/cosmosis
export FIRECROWN_DIR=$HOME/soft//firecrown
export FIRECROWN_EXAMPLES_DIR=$HOME/soft/firecrown/examples
export CSL_DIR=$HOME/soft/cosmosis-standard-library


export OMP_NUM_THREADS=8

module unload gcc
module unload PrgEnv-gnu
module load PrgEnv-gnu
module load cmake
module unload craype-hugepages2M




srun -n 61  cosmosis  sn_srd_planck_bao.ini --mpi
# cosmosis-postprocess sn_srd_planck_bao.ini
