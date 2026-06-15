#!/bin/bash
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Ensure reproducibility at system level
export PYTHONHASHSEED=42

# Define variables
MODEL_NAME="roberta-base"
TASK_NAME="qnli"
LORA_R_VALUES="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16"
LORA_BUDGET=1263000

batch_size=32
learning_rate=4.8e-4
num_epochs=49
weight_decay=0.283
max_grad_norm=0.450
max_seq_length=128
warmup_ratio=0.022
pruning_target_reduction=0.8
pruning_steps=10
importance_ema_decay=0.976
momentum_penalty_weight=0.915
stable_layer_bonus=0.05  

RECOVERY_STEPS=100  # Recovery steps after pruning
EXTENDED_RECOVERY_STEPS=200

SEEDS=(42 2025 777)

# Loop through each seed and run the experiment
for SEED in "${SEEDS[@]}"; do
    echo "========================================"
    echo "Starting training with seed ${SEED}"
    echo "========================================"
    
    # Create unique output directory for this seed
    OUTPUT_DIR="output/${TASK_NAME}_dp${pruning_target_reduction}_seed${SEED}_$(date +%Y%m%d_%H%M%S)"
    
    # Create output directory
    mkdir -p $OUTPUT_DIR

    # Run the training script with current seed
    script -q -c "python3 -m training.train_glue \
        --model_name_or_path $MODEL_NAME \
        --task_name $TASK_NAME \
        --output_dir $OUTPUT_DIR \
        --lora_r_values $LORA_R_VALUES \
        --lora_budget $LORA_BUDGET \
        --per_device_train_batch_size $batch_size \
        --per_device_eval_batch_size $((batch_size * 4)) \
        --learning_rate $learning_rate \
        --num_train_epochs $num_epochs \
        --weight_decay $weight_decay \
        --warmup_ratio $warmup_ratio \
        --max_seq_length $max_seq_length \
        --lr_scheduler_type linear \
        --max_grad_norm $max_grad_norm \
        --seed $SEED \
        --overwrite_output_dir \
        --use_initial_rank_allocation \
        --apply_pruning \
        --pruning_target_reduction $pruning_target_reduction \
        --pruning_steps $pruning_steps \
        --importance_ema_decay $importance_ema_decay \
        --momentum_penalty_weight $momentum_penalty_weight \
        --stable_layer_bonus $stable_layer_bonus \
        --recovery_steps $RECOVERY_STEPS \
        --extended_recovery_steps $EXTENDED_RECOVERY_STEPS" $OUTPUT_DIR/full_terminal_output.log

    # Save final reproducibility verification
    echo "Execution completed: $(date)" >> $OUTPUT_DIR/reproducibility_info.txt
    echo "Exit code: $?" >> $OUTPUT_DIR/reproducibility_info.txt
    echo "Used seed: ${SEED}" >> $OUTPUT_DIR/reproducibility_info.txt
    
    echo "Completed training with seed ${SEED}"
    echo "========================================"
done

echo "All training runs completed successfully!"
