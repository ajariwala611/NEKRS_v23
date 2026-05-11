#!/bin/bash

# ---- Set hardware resources ----
GPU=0                    # Only one GPU per node
CPU=0                    # NUMA node 0 (all CPUs are here)
NIC=mlx5_0               # Mellanox ConnectX-7

APP="ibrun nekrs --setup NACA.par"

# ---- Detect MPI rank ----
if [ ! -z ${OMPI_COMM_WORLD_RANK} ]; then
    RANK=${OMPI_COMM_WORLD_RANK}
    LOCAL_RANK=${OMPI_COMM_WORLD_LOCAL_RANK}
elif [ ! -z ${MV2_COMM_WORLD_RANK} ]; then
    RANK=${MV2_COMM_WORLD_RANK}
    LOCAL_RANK=${MV2_COMM_WORLD_LOCAL_RANK}
else
    RANK=${SLURM_PROCID}
    LOCAL_RANK=${SLURM_LOCALID}
fi

# ---- Environment variables ----
export CUDA_VISIBLE_DEVICES=$GPU
export UCX_NET_DEVICES=$NIC:1
export UCX_TLS="cuda,sm,rc"
export UCX_RNDV_SCHEME="auto"
export UCX_RNDV_THRESH=intra:auto,inter:auto
#export CUDA_VISIBLE_DEVICES="$GPU"
#export UCX_NET_DEVICES="all"
#export UCX_TLS="all"
export UCX_IB_GPU_DIRECT_RDMA="yes"
export UCX_PROTO_ENABLE="y"
export UCX_LOG_LEVEL="info"
export UCX_MEMTYPE_CACHE="n"

export OMPI_MCA_pml="ucx"
export OMPI_MCA_btl="^vader,tcp,openib,smcuda"
export OMPI_MCA_osc="ucx"
#export NEKRS_GPU_MPI=1
if [[ $UCX_TLS == *"cuda"* ]]; then
  export NEKRS_GPU_MPI=1
fi

ulimit -s unlimited 2>/dev/null
COMMAND="numactl --cpunodebind=$CPU --membind=$CPU $APP"
echo "RANK=$RANK, LOCAL_RANK=$LOCAL_RANK, GPU=$GPU, NIC=$NIC, CPU=$CPU, NEKRS_GPU_MPI=$NEKRS_GPU_MPI"
$COMMAND
