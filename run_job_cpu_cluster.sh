#!/bin/bash
#SBATCH -A sharpe                # group to which you belong
#SBATCH -N 1                        # number of nodes
#SBATCH -n 1                        # number of cores
#SBATCH --mem 8G	                # memory pool for all cores
#SBATCH -t 0-16:00                  # runtime limit (D-HH:MM)
#SBATCH -o slurm.%N.%j.out          # STDOUT
#SBATCH -e slurm.%N.%j.err          # STDERR
#SBATCH --mail-type=END,FAIL        # notifications for job done & fail
#SBATCH --mail-user=miquel.marin@embl.es # send-to address

## gpu=1080Ti  gpu=K20   gpu=P100

echo "START JOB"
T0=$(date +%s)

module load CUDA/11.0.2-GCC-9.3.0
echo "MODULES LOADED"

SOURCE_FOLDER=/home/marin/crypt_patterning_simulation

mkdir $TMPDIR/yalla
mkdir $TMPDIR/crypt_patterning_simulation

cp -r /home/marin/yalla/include $TMPDIR/yalla/

cp $SOURCE_FOLDER/crypt_patterning.cu $TMPDIR/crypt_patterning_simulation
cp $SOURCE_FOLDER/batch_run_process_crypt_patterning.py $TMPDIR/crypt_patterning_simulation
cp $SOURCE_FOLDER/compute_crypt_histogram.py $TMPDIR/crypt_patterning_simulation


echo "COPIED STUFF TO NODE"
cd $TMPDIR/crypt_patterning_simulation

echo "START"
echo "visible devices" $CUDA_VISIBLE_DEVICES

PYTHON=/home/marin/.conda/envs/py3env/bin/python

$PYTHON batch_run_process_crypt_patterning.py

echo "Simulations finished"

#cp $TMPDIR/spheroid_fusion_code/paropt_cluster_logs/* $LOGS_FOLDER
# echo "COPIED STUFF BACK TO SERVER"

echo "DONE!"
echo "time elapsed = $((($(date +%s)-$T0)/60))"
