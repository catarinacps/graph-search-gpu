#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --chdir=.
#SBATCH --partition=draco
#SBATCH --nodes=1
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hcpsilva@inf.ufrgs.br

set -euxo pipefail

HOST=$(hostname)

# machine:
MACHINE=${HOST}_${SLURM_CPUS_ON_NODE}

# parameters:
# the experiment ID, defined in the lab-book
EXP_ID=gpu_aa_tp2
# the code directory
CODE_DIR=$1
# the experiment directory
EXP_DIR=$CODE_DIR/labbook/gpu

# experiment name (which is the ID and the machine and its core count)
EXP_NAME=${EXP_ID}_${MACHINE}

# go to the scratch dir
cd $SCRATCH

# and clean everything
rm -rf *

# prepare our directory
mkdir $EXP_NAME
pushd $EXP_NAME

# set out chosen cuda path version
CUDA_INSTALLATION=/usr/local/cuda-10.1

# update env vars
LD_LIBRARY_PATH+=:${CUDA_INSTALLATION}/lib64
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH

PATH+=:${CUDA_INSTALLATION}/bin
export PATH=$PATH

# copy the code folder
cp -r $CODE_DIR code
mkdir results
results_csv=$(readlink -f results/${EXP_NAME}.csv)
results_dir=$(readlink -f results)
pushd code

# build so we run faster
make CUDA_OPT=NVIDIA CUDA_PATH=$CUDA_INSTALLATION

# init the csv results file
echo "id,method,instance,block_size,time" > $results_csv

# math solver
while read -r id method instance block_size; do
    csv_line=${id},${method},${instance},${block_size}

    echo
    echo "--> Running with params: $id $method $instance $block_size"

    log_file=$results_dir/${id}_${method}_${instance}_${block_size}.log

    ./build/gsg cuda -b $block_size $method data/$instance > $log_file

    time_obs=$(grep '^time' $log_file | awk '{print $2}')

    echo ${csv_line},${time_obs} >> $results_csv
done < $EXP_DIR/runs.plan

popd

# pack everything and send to the exp dir
tar czf $EXP_DIR/data/$EXP_NAME.tar.gz *

popd
rm -rf $SCRATCH/*
