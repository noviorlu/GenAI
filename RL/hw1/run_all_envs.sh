#!/bin/bash
export PYTHONPATH=.
export WANDB_ANONYMOUS=allow

# List of environments
ENVS=("Ant" "HalfCheetah" "Hopper" "Walker2d")

for ENV in "${ENVS[@]}"; do
    ENV_NAME="${ENV}-v4"
    
    echo "========================================"
    echo "Starting Vanilla BC for ${ENV} (1 iteration, with video)..."
    echo "========================================"
    conda run -n cs224r python cs224r/scripts/run_hw1.py \
        --expert_policy_file cs224r/policies/experts/${ENV}.pkl \
        --env_name ${ENV_NAME} \
        --exp_name bc_${ENV}_full \
        --n_iter 1 \
        --expert_data cs224r/expert_data/expert_data_${ENV_NAME}.pkl \
        --video_log_freq 1
    
    echo "========================================"
    echo "Starting DAgger for ${ENV} (50 iterations, with video every iteration)..."
    echo "========================================"
    conda run -n cs224r python cs224r/scripts/run_hw1.py \
        --expert_policy_file cs224r/policies/experts/${ENV}.pkl \
        --env_name ${ENV_NAME} \
        --exp_name dagger_${ENV}_full \
        --n_iter 50 \
        --do_dagger \
        --expert_data cs224r/expert_data/expert_data_${ENV_NAME}.pkl \
        --video_log_freq 1
done

echo "🎉 All environments have finished training!"
