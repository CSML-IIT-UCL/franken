#!/bin/bash
#SBATCH --account=IscrB_ProAmmo
#SBATCH --time 24:00:00             # format: HH:MM:SS
#SBATCH --nodes=1                   # node
#SBATCH --ntasks-per-node=1         # tasks out of 32
#SBATCH --gres=gpu:1                # gpus per node out of 4
#SBATCH --cpus-per-task=10
#SBATCH -p boost_usr_prod
##SBATCH --qos=boost_qos_dbg 
############################

. ~/.bashrc

micromamba activate franken-dev
cd /leonardo/home/userexternal/gmeanti0/franken

export PYTHONUNBUFFERED=1
PYTHONPATH='.' python tests/test_transfer_perf/test_water.py --db-path=tests/test_transfer_perf/results_2703.pkl
