#!/bin/bash

#SBATCH --ntasks 1

#SBATCH --cpus-per-task 1

#SBATCH --mem-per-cpu 6G

#SBATCH --time 0-1:0:0

#SBATCH --array=0-3%50

#SBATCH --output /cluster/work/demler/project_active/fmarijanovic/savefiles/psBQP-keldysh/slurm_logs/1788248238/output/output_%a_slurmjob_%j
#SBATCH --error /cluster/work/demler/project_active/fmarijanovic/savefiles/psBQP-keldysh/slurm_logs/1788248238/error/error_%a_slurmjob_%j

#SBATCH --job-name psBQP-keldysh

#SBATCH --mail-user fmarijanovic@phys.ethz.ch
#SBATCH --mail-type NONE

module load stack/2024-06 python/3.11.6 

srun python /cluster/work/demler/tools_library_source_code/development_3/demler_tools/cluster_backend/cluster_run_backend.py main_simulation /cluster/work/demler/project_active/fmarijanovic/savefiles/psBQP-keldysh/simulation_results/1788248238/cluster_files/slurm_control_0 $SLURM_ARRAY_TASK_ID 

exit 0
