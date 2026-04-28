#!/bin/bash

# Capture Git Metadata, tying experiment to specific code version
VERSION_ID=$(git describe --abbrev=0) 2>/dev/null || VERSION_ID=$(git rev-parse --short HEAD 2>/dev/null || echo "no_git")

# Get this script's directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Setup base output directory
BASE_LOG_DIR="$PROJECT_DIR/logs/thermostability/progen3/CM/git_${VERSION_ID}"
mkdir -p "$BASE_LOG_DIR"

# Move to project directory
cd "$PROJECT_DIR" || exit

# Set fixed parameters
block_size=2
num_steps=60
num_mcmc_steps=10
num_sequences=1024
resampling_window=5
seed=42
save_interval=5
ckpt_path=$PROJECT_DIR/checkpoints/progen3/CM/checkpoint-170

# Define parameters grid
weights=(0 5 10 15 20)
betas=(1.0 1.8 2.6)

# Nested loops to iterate through the grid
job_number=0
for current_weight in "${weights[@]}"; do
    for current_beta in "${betas[@]}"; do

        # Get starting time and job number
        start_time=$(date +%s)
        job_number=$((job_number + 1))
        
        # Create output directory name, removing '.' from float values
        current_beta_str=$(echo "$current_beta" | tr '.' 'p')
        output_dir="${BASE_LOG_DIR}/Be${current_beta_str}_Lth${current_weight}"

        # Create YAML config file in output directory
        mkdir -p "$output_dir"
        potentials_config="$output_dir/config.yaml"

        cat <<EOL > "$potentials_config"
# Varying parameters
potentials:
  - type: thermostability
    weight: $current_weight
beta: $current_beta

# Fixed parameters
block_size: $block_size
num_steps: $num_steps
num_mcmc_steps: $num_mcmc_steps
num_sequences: $num_sequences
resampling_window: $resampling_window
seed: $seed
save_interval: $save_interval
ckpt_path: $ckpt_path
EOL

        echo "----------------------------------------------------------"
        echo "Launching: weight=$current_weight | beta=$current_beta"
        echo "Output: $output_dir"
        echo "----------------------------------------------------------"

        # Execute the command
        # Note: If running on a single GPU, these will run sequentially.
        pixi run progen-run-mc \
            --output-dir "$output_dir" \
            --save-interval "$save_interval" \
            --ckpt "$ckpt_path" \
            --num-sequences "$num_sequences" \
            --beta "${current_beta}" \
            --block-size "$block_size" \
            --num-mcmc-steps "$num_mcmc_steps" \
            --num-steps "$num_steps" \
            --resampling-window "$resampling_window" \
            --seed "$seed" \
            --potentials-config "$potentials_config" \
            --disable-progress-bar

        # Get ending time
        end_time=$(date +%s)
        elapsed=$((end_time - start_time))
        echo "Job $job_number | Weight=$current_weight | Beta=$current_beta | Time=$elapsed seconds."

    done
done

echo "All jobs completed. Git Tag: $VERSION_ID"
