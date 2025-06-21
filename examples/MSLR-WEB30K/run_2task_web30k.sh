#!/bin/bash

# Number of available GPUs (change this as needed)
NUM_GPUS=1

# List of multi-objective optimization methods
MOO_METHODS=(
    # "stl"
    # "ls"
    # "uw"
    # "scaleinvls"
    # "rlw"
    # "dwa"
    # "mgda"
    # "pcgrad"
    # "graddrop"
    # "log_mgda"
    # "cagrad" 
    # "log_cagrad"
    # "imtl"
    # "log_imtl"
    # "nashmtl"
    # "famo"
    # "sdmgrad"
    # "wc"
    # "soft_wc"
    # "epo"
    # "wc_mgda"
    "ec"
)

# Dataset options
DATASETS=("50bps")

# Reduction options
REDUCTIONS=("mean")

# Common parameters
CONFIG_FILE_PATH="scripts/local_config_web.json"
# RUN_ID="test_run1"
OUTPUT_DIR="allrank/run"
TASK_INDICES="0,131"
TASK_WEIGHTS="0,10"

# Function to run a batch of jobs
run_batch() {
    local batch_start_idx=$1
    local batch_end_idx=$2
    local dataset_name=$3
    local reduction_method=$4
    
    # Launch jobs
    for ((i=batch_start_idx; i<batch_end_idx && i<${#MOO_METHODS[@]}; i++)); do
        gpu_id=$((i % NUM_GPUS))
        moo_method=${MOO_METHODS[i]}
        echo "Starting job with method $moo_method on GPU $gpu_id (dataset: $dataset_name, reduction: $reduction_method)"
        
        CUDA_VISIBLE_DEVICES=$gpu_id python main_ntasks.py \
            --config-file-path $CONFIG_FILE_PATH \
            --output-dir $OUTPUT_DIR \
            --task-indices $TASK_INDICES \
            --task-weights $TASK_WEIGHTS \
            --moo-method $moo_method \
            --dataset-name $dataset_name \
            --reduction-method $reduction_method &
    done
    
    # Wait for all background jobs to complete
    wait
    echo "Batch complete for dataset $dataset_name with reduction $reduction_method"
}

# Calculate total number of batches needed
TOTAL_METHODS=${#MOO_METHODS[@]}
BATCH_SIZE=$NUM_GPUS

echo "=== Starting all job combinations ==="
echo "Datasets: ${DATASETS[*]}"
echo "Reductions: ${REDUCTIONS[*]}"
echo "Methods: ${#MOO_METHODS[@]} different methods"
echo "GPUs: $NUM_GPUS"
echo "==============================="

# Loop through all datasets
for dataset in "${DATASETS[@]}"; do
    # Loop through all reductions
    for reduction in "${REDUCTIONS[@]}"; do
        echo "===== Starting jobs for dataset $dataset with reduction $reduction ====="
        
        # Run batches for current dataset and reduction combination
        for ((batch=0; batch*BATCH_SIZE<TOTAL_METHODS; batch++)); do
            start_idx=$((batch * BATCH_SIZE))
            end_idx=$(((batch + 1) * BATCH_SIZE))
            
            echo "Running batch $((batch + 1)) (methods ${start_idx}-$((end_idx-1)))"
            run_batch $start_idx $end_idx $dataset $reduction
        done
        
        echo "===== Completed all jobs for dataset $dataset with reduction $reduction ====="
    done
done

echo "All job combinations completed successfully!"








#!/bin/bash

# CUDA_VISIBLE_DEVICES=0 python main_2tasks.py --config-file-path scripts/local_config_web.json --run-id test_run1 --output-dir allrank/test_run --task-indices 0,135 --task-weights 0,12 --moo-method ls --dataset 50bps --reduction mean
# CUDA_VISIBLE_DEVICES=1 python main_2tasks.py --config-file-name scripts/local_config_web.json --run-id test_run1 --job-dir allrank/test_run --task_number_in_indices_all 0,131 --weight_selection 0,12 --moo_method uw&
# CUDA_VISIBLE_DEVICES=2 python main_2tasks.py --config-file-name scripts/local_config_web.json --run-id test_run1 --job-dir allrank/test_run --task_number_in_indices_all 0,131 --weight_selection 0,12 --moo_method scaleinvls&
# CUDA_VISIBLE_DEVICES=3 python main_2tasks.py --config-file-name scripts/local_config_web.json --run-id test_run1 --job-dir allrank/test_run --task_number_in_indices_all 0,131 --weight_selection 0,12 --moo_method rlw&
# CUDA_VISIBLE_DEVICES=4 python main_2tasks.py --config-file-name scripts/local_config_web.json --run-id test_run1 --job-dir allrank/test_run --task_number_in_indices_all 0,131 --weight_selection 0,12 --moo_method dwa&
# CUDA_VISIBLE_DEVICES=5 python main_2tasks.py --config-file-name scripts/local_config_web.json --run-id test_run1 --job-dir allrank/test_run --task_number_in_indices_all 0,131 --weight_selection 0,12 --moo_method mgda&
# CUDA_VISIBLE_DEVICES=6 python main_2tasks.py --config-file-name scripts/local_config_web.json --run-id test_run1 --job-dir allrank/test_run --task_number_in_indices_all 0,131 --weight_selection 0,12  --moo_method pcgrad&
# CUDA_VISIBLE_DEVICES=7 python main_2tasks.py --config-file-name scripts/local_config_web.json --run-id test_run1 --job-dir allrank/test_run --task_number_in_indices_all 0,131 --weight_selection 0,12 --moo_method graddrop&

# wait 

# CUDA_VISIBLE_DEVICES=0 python main_2tasks.py --config-file-name scripts/local_config_web.json --run-id test_run1 --job-dir allrank/test_run --task_number_in_indices_all 0,131 --weight_selection 0,12  --moo_method log_mgda&
# CUDA_VISIBLE_DEVICES=1 python main_2tasks.py --config-file-name scripts/local_config_web.json --run-id test_run1 --job-dir allrank/test_run --task_number_in_indices_all 0,131 --weight_selection 0,12 --moo_method cagrad&
# CUDA_VISIBLE_DEVICES=2 python main_2tasks.py --config-file-name scripts/local_config_web.json --run-id test_run1 --job-dir allrank/test_run --task_number_in_indices_all 0,131 --weight_selection 0,12  --moo_method log_cagrad&
# CUDA_VISIBLE_DEVICES=3 python main_2tasks.py --config-file-name scripts/local_config_web.json --run-id test_run1 --job-dir allrank/test_run --task_number_in_indices_all 0,131 --weight_selection 0,12 --moo_method imtl&
# CUDA_VISIBLE_DEVICES=4 python main_2tasks.py --config-file-name scripts/local_config_web.json --run-id test_run1 --job-dir allrank/test_run --task_number_in_indices_all 0,131 --weight_selection 0,12  --moo_method log_imtl&
# CUDA_VISIBLE_DEVICES=5 python main_2tasks.py --config-file-name scripts/local_config_web.json --run-id test_run1 --job-dir allrank/test_run --task_number_in_indices_all 0,131 --weight_selection 0,12 --moo_method nashmtl&
# CUDA_VISIBLE_DEVICES=6 python main_2tasks.py --config-file-name scripts/local_config_web.json --run-id test_run1 --job-dir allrank/test_run --task_number_in_indices_all 0,131 --weight_selection 0,12  --moo_method famo&
# CUDA_VISIBLE_DEVICES=7 python main_2tasks.py --config-file-name scripts/local_config_web.json --run-id test_run1 --job-dir allrank/test_run --task_number_in_indices_all 0,131 --weight_selection 0,12  --moo_method wc&

# wait 

# CUDA_VISIBLE_DEVICES=0 python main_2tasks.py --config-file-name scripts/local_config_web.json --run-id test_run1 --job-dir allrank/test_run --task_number_in_indices_all 0,131 --weight_selection 0,12  --moo_method epo&
# CUDA_VISIBLE_DEVICES=1 python main_2tasks.py --config-file-name scripts/local_config_web.json --run-id test_run1 --job-dir allrank/test_run --task_number_in_indices_all 0,131 --weight_selection 0,12  --moo_method wc_mgda&
# CUDA_VISIBLE_DEVICES=2 python main_2tasks.py --config-file-name scripts/local_config_web.json --run-id test_run1 --job-dir allrank/test_run --task_number_in_indices_all 0,131 --weight_selection 0,12 --moo_method ec&

# wait
