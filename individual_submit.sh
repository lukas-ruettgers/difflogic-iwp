#!/bin/bash
INTERACTIVE=true

SIF="container-devel-imagenet" # container-devel-imagenet, container-devonly, container
eval_freq=10
ext_eval_freq=10
num_iterations=0
arch="lgn" #lgn, cnn
depth_scale=20
iwp="--iwp"
init="--weights_init ri"

# additional="--wandb_verbose eval --dataset c100-real-input"
additional="--wandb_verbose eval --log_verbose timing --n_timing_measurements 20"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

if [ "$INTERACTIVE" = true ]; then
    apptainer exec \
        --nv \
        --bind /itet-stor/$USER/net_scratch/projects:/itet-stor/$USER/net_scratch/projects \
        --bind /scratch \
        --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        /itet-stor/lruettgers/net_scratch/projects/difflogic/singularity/"$SIF".sif \
        python /itet-stor/lruettgers/net_scratch/projects/difflogic-iwp/main.py \
        -a "$arch" --dataset "cifar-100" --encoding '3-thresholds' \
        -ef "$eval_freq" -eef "$ext_eval_freq" -ni "$num_iterations" \
        --depth_scale "$depth_scale" $iwp $init $resinit $additional
else
    echo "Submitting: sif=$SIF | ef= $eval_freq | eef=$ext_eval_freq | ni=$num_iterations | arch=$arch | depth=$depth_scale | $iwp | $init | $additional "
    sbatch template_job.sh "$SIF" "$eval_freq" "$ext_eval_freq" "$num_iterations" "$arch" "$depth_scale" "$iwp" "$init" "$additional"
fi
