#!/usr/bin/env bash
export MXNET_GPU_MEM_POOL_RESERVE=50
# export MXNET_CPU_WORKER_NTHREADS=24
# export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
# export MXNET_ENGINE_TYPE=ThreadedEnginePerDevice

python face_det.py --is-test 1 --gpu 3
