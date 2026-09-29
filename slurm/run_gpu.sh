#!/bin/bash
################################################################################
# SLURM Resource Directives — MUST be before any executable lines
################################################################################
#SBATCH -A <your_project_id>          # e.g. a Bianca sens project
#SBATCH -p node
#SBATCH -n 16
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH --gpus-per-node=1              # maximum 2x A100 40GB                   
#SBATCH --time=72:00:00                
#SBATCH --output=results/slurm_%j.log  # stdout
#SBATCH --error=results/slurm_%j.err   # stderr
################################################################################
# Minimal & Reliable Python Environment Setup
################################################################################
# Python path (MODIFY THIS to match your environment)
PYTHON_BIN="${PYTHON_BIN:-python}"      # or the full path to your environment's python
# Script settings
# GPU step to run, passed as the first argument (default: tabpfn_train.py):
#   sbatch slurm/run_gpu.sh tabpfn_ratio_grid_search.py
#   sbatch slurm/run_gpu.sh tabpfn_train.py
#   sbatch slurm/run_gpu.sh tabpfn_xai.py        # ~80 GPU-hours (SHAP)
PYTHON_SCRIPT="${1:-tabpfn_train.py}"
OUTPUT_DIR="results"
OUTPUT_FILE="$OUTPUT_DIR/output_${SLURM_JOB_ID}.txt"
################################################################################
# Validation & Execution
################################################################################
echo "=========================================="
echo "Environment Check"
echo "=========================================="

# Verify Python exists
if ! command -v "$PYTHON_BIN" > /dev/null 2>&1; then
    echo "ERROR: Python not found at $PYTHON_BIN"
    exit 1
fi
echo "✓ Python: $($PYTHON_BIN --version 2>&1)"

# Verify key packages
# $PYTHON_BIN -c "import seaborn, pandas, numpy, tabpfn" 2>&1
# if [ $? -ne 0 ]; then
#     echo "ERROR: Failed to import required packages"
#     exit 1
# fi
# echo "✓ Required packages available"

# # --- GPU / CUDA check (critical for TabPFN) ---
# echo ""
# echo "=========================================="
# echo "GPU Check"
# echo "=========================================="
# if command -v nvidia-smi &> /dev/null; then
#     nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
# else
#     echo "WARNING: nvidia-smi not found"
# fi

# $PYTHON_BIN -c "
# import torch
# if not torch.cuda.is_available():
#     raise RuntimeError('CUDA is not available — TabPFN requires a GPU.')
# print(f'✓ CUDA {torch.version.cuda} | GPU: {torch.cuda.get_device_name(0)} | '
#       f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
# " 2>&1
# if [ $? -ne 0 ]; then
#     echo "ERROR: GPU/CUDA check failed. Check #SBATCH --gres=gpu directive."
#     exit 1
# fi

echo ""
echo "=========================================="

# Create output directory
mkdir -p $OUTPUT_DIR

# Environment variables
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export CUDA_VISIBLE_DEVICES=0          # Pin to the first allocated GPU

# Run the script
echo "Running: $PYTHON_SCRIPT"
echo "Output:  $OUTPUT_FILE"
echo ""
$PYTHON_BIN $PYTHON_SCRIPT 2>&1 | tee $OUTPUT_FILE

# Report status
EXIT_CODE=$?
echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Completed successfully"
else
    echo "✗ Failed (exit code: $EXIT_CODE)"
    exit $EXIT_CODE
fi