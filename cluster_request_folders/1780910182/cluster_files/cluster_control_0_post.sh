#!/bin/bash

#SBATCH --ntasks 1

#SBATCH --cpus-per-task 1

#SBATCH --mem-per-cpu 128M

#SBATCH --time 0-0:2:0

#SBATCH --output /cluster/work/demler/project_active/fmarijanovic/savefiles/psBQP-keldysh/slurm_logs/1780910182/output/output_slurmjob_%j
#SBATCH --error /cluster/work/demler/project_active/fmarijanovic/savefiles/psBQP-keldysh/slurm_logs/1780910182/error/error_slurmjob_%j

#SBATCH --job-name autopost_psBQP-keldysh

#SBATCH --mail-user fmarijanovic@phys.ethz.ch
#SBATCH --mail-type NONE

module load stack/2024-06 python/3.11.6

srun python /cluster/work/demler/tools_library_source_code/development_3/demler_tools/cluster_backend/cluster_run_backend.py auto_postprocessing /cluster/work/demler/project_active/fmarijanovic/savefiles/psBQP-keldysh/simulation_results/1780910182/cluster_files/slurm_control_0

exit 0
