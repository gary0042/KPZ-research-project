#!/bin/bash
#SBATCH -A m3152                 # NERSC account
#SBATCH -q regular               # Queue (partition); use 'debug' for short tests
#SBATCH -C cpu                   # CPU-only node
#SBATCH -N 1                     # Number of nodes
#SBATCH --ntasks-per-node=1      # One MPI task
#SBATCH --cpus-per-task=1        # 60 CPUs for that task
#SBATCH -t 00:00:25              # Walltime (hh:mm:ss)
#SBATCH -J hello_slurm           # Job name
#SBATCH -o logs/%x-%j.out        # STDOUT
#SBATCH -e logs/%x-%j.err        # STDERR

#SBATCH --mail-type=ALL # Email on start, end, fail
#SBATCH --mail-user=gary_han@berkeley.edu

#-------------------------------------------------------------------------------
# QoL: fail fast on errors, echo commands
#-------------------------------------------------------------------------------
set -euo pipefail
echo "Starting job ${SLURM_JOB_NAME} (ID: ${SLURM_JOB_ID}) on host $(hostname)"
echo "Running in directory: ${PWD}"
echo "SLURM_NODELIST: ${SLURM_NODELIST}"
echo "SLURM_NTASKS:   ${SLURM_NTASKS}"
echo "CPUS per task:  ${SLURM_CPUS_PER_TASK}"

date

#-------------------------------------------------------------------------------
# Environment setup
#-------------------------------------------------------------------------------

module load conda

# (Often needed on NERSC to make 'conda activate' work in batch)
if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
fi

conda activate py311-stoch_surf_growth

echo "Using Python: $(which python)"
python --version

#-------------------------------------------------------------------------------
# Run the Python script
#-------------------------------------------------------------------------------

# QoL: create logs dir if not present
mkdir -p logs

# Use srun so resources are correctly bound by Slurm
srun -n 1 -c ${SLURM_CPUS_PER_TASK} /global/homes/g/ghan36/slurm_batch_scripts/slurm_test.py

echo "Job completed at:"
date