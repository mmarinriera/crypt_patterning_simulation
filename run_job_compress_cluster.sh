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


gzip --verbose -9 /g/sharpe-hd/marin/crypt_patterning_spatial_scales_28-10-21/*1.vtk

#cp $TMPDIR/spheroid_fusion_code/paropt_cluster_logs/* $LOGS_FOLDER
# echo "COPIED STUFF BACK TO SERVER"

echo "DONE!"
echo "time elapsed = $((($(date +%s)-$T0)/60))"
