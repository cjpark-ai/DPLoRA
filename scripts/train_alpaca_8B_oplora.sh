#!/bin/bash
# Get the project root directory (parent of scripts directory)
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH=$PYTHONPATH:${PROJECT_DIR}

# Load environment variables from .env file
if [ -f "${PROJECT_DIR}/.env" ]; then
    source "${PROJECT_DIR}/.env"
    echo "Environment variables loaded from .env"
else
    echo "Warning: .env file not found at ${PROJECT_DIR}/.env"
fi

# Ensure reproducibility at system level
export PYTHONHASHSEED=42
# Critical: Enable expandable memory segments to prevent OOM errors
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Define variables
MODEL_NAME="meta-llama/Meta-Llama-3-8B"
TASK_NAME="alpaca"
LORA_R_VALUES="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16"
LORA_BUDGET=20500000

batch_size=1
learning_rate=3.4e-5
num_epochs=7
weight_decay=0.03
max_grad_norm=1.287
max_seq_length=512
warmup_ratio=0.073
gradient_accumulation_steps=16
lora_dropout=0.179

SEEDS=(42 2025 777)

# Loop through each seed and run the experiment
for SEED in "${SEEDS[@]}"; do
    echo "========================================"
    echo "Starting training with seed ${SEED}"
    echo "========================================"
    
    # Generate single timestamp for entire experiment
    EXPERIMENT_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    
    # Create unique output directory for this seed (use absolute path)
    OUTPUT_DIR="${PROJECT_DIR}/output/${TASK_NAME}_8B_oplora_seed${SEED}_${EXPERIMENT_TIMESTAMP}"
    
    # Create output directory
    mkdir -p "$OUTPUT_DIR"

    # Run the training script with current seed (exact same as trial_0)
    python3 -m training.train_alpaca \
        --model_name_or_path "$MODEL_NAME" \
        --task_name "$TASK_NAME" \
        --output_dir "$OUTPUT_DIR" \
        --lora_r_values "$LORA_R_VALUES" \
        --lora_budget "$LORA_BUDGET" \
        --per_device_train_batch_size $batch_size \
        --per_device_eval_batch_size 40 \
        --eval_split_ratio 0.01 \
        --max_eval_samples 200 \
        --gradient_accumulation_steps $gradient_accumulation_steps \
        --learning_rate $learning_rate \
        --num_train_epochs $num_epochs \
        --weight_decay $weight_decay \
        --warmup_ratio $warmup_ratio \
        --max_seq_length $max_seq_length \
        --lr_scheduler_type cosine \
        --max_grad_norm $max_grad_norm \
        --lora_dropout $lora_dropout \
        --seed "$SEED" \
        --overwrite_output_dir \
        --use_initial_rank_allocation \
        --use_memory_efficient_importance \
        --importance_chunk_size 1 \
        2>&1 | tee "$OUTPUT_DIR/full_terminal_output.log"


    
    TRAIN_EXIT_CODE=$?

    # Save final reproducibility verification
    echo "Execution completed: $(date)" >> "$OUTPUT_DIR/reproducibility_info.txt"
    echo "Exit code: $TRAIN_EXIT_CODE" >> "$OUTPUT_DIR/reproducibility_info.txt"
    echo "Used seed: ${SEED}" >> "$OUTPUT_DIR/reproducibility_info.txt"
    
    echo "Completed training with seed ${SEED}"
    echo "========================================"
done

echo "All training runs completed successfully!"
