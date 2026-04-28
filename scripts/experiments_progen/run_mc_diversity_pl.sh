#!/bin/bash

# Capture Git Metadata, tying experiment to specific code version
VERSION_ID=$(git describe --abbrev=0) 2>/dev/null || VERSION_ID=$(git rev-parse --short HEAD 2>/dev/null || echo "no_git")

# Get this script's directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Setup base output directory
BASE_LOG_DIR="$PROJECT_DIR/logs/diversity/progen3/PL/git_${VERSION_ID}"
CONFIG_FILE="$SCRIPT_DIR/diversity_pl_config.csv"

# Move to project directory
cd "$PROJECT_DIR" || exit

# 3. Fixed Parameters
block_size=2
num_steps=80
num_mcmc_steps=10
num_sequences=1024
seed=42
save_interval=5
ckpt_path=$PROJECT_DIR/checkpoints/progen3/PL/checkpoint-132

# 4. Read CSV and Loop
# We use tail to skip the header (first line)
job_number=0
tail -n +2 "$CONFIG_FILE" | while IFS=',' read -r current_window current_beta
do
    # Get starting time and job number
    start_time=$(date +%s)
    job_number=$((job_number + 1))

    # Skip empty lines if any
    [ -z "$current_window" ] && continue

    # 5. Improved Versioning String
    # Converts 1.8 to 1p8; maintains transparency
    current_beta_str=$(echo "$current_beta" | tr '.' 'p')
    
    # Path: git_HASH/K{window}_Be{beta}
    output_dir="${BASE_LOG_DIR}/K${current_window}_Be${current_beta_str}"

    mkdir -p "$output_dir"
    potentials_config="$output_dir/config.yaml"

    # 6. Generate YAML Config
    cat <<EOL > "$potentials_config"
# Metadata
git_hash: $GIT_HASH

# Varying parameters (Parsed from CSV)
resampling_window: $current_window
beta: $current_beta

# Fixed parameters
block_size: $block_size
num_steps: $num_steps
num_mcmc_steps: $num_mcmc_steps 
num_sequences: $num_sequences
seed: $seed
save_interval: $save_interval
ckpt_path: $ckpt_path
EOL

    echo "----------------------------------------------------------"
    echo "Processing: Window=$current_window | Beta=$current_beta"
    echo "----------------------------------------------------------"

    # 7. Execution
    pixi run progen-run-mc \
        --output-dir "$output_dir" \
        --save-interval "$save_interval" \
        --ckpt "$ckpt_path" \
        --num-sequences "$num_sequences" \
        --beta "${current_beta}" \
        --block-size "$block_size" \
        --num-mcmc-steps "$num_mcmc_steps" \
        --num-steps "$num_steps" \
        --resampling-window "$current_window" \
        --seed "$seed" \
        --potentials-config "$potentials_config" \
        --min-length 110 \
        --disable-progress-bar

    # Get ending time
    end_time=$(date +%s)
    elapsed=$((end_time - start_time))
    echo "Job $job_number | Window=$current_window | Beta=$current_beta | Time=$elapsed seconds."

done

echo "All configurations processed. Git Tag: $VERSION_ID"
