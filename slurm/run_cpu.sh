#!/bin/bash
################################################################################
# SLURM Resource Directives
################################################################################
#SBATCH -A <your_project_id>          # e.g. a Bianca sens project
#SBATCH -p node
#SBATCH -n 16
#SBATCH -N 1
#SBATCH --time=48:00:00               
#SBATCH --output=results/slurm_%j.log
#SBATCH --error=results/slurm_%j.err
################################################################################
# Environment Setup
################################################################################
PYTHON_BIN="${PYTHON_BIN:-python}"      # or the full path to your environment's python
OUTPUT_DIR="results"

# Full CPU pipeline, in order. Run from the repository root:
#   sbatch slurm/run_cpu.sh
# Sensitivity analysis: export WAVE_INTERVAL_DAYS=18 SERIAL_INTERVAL_DAYS=3
# before submitting (read by data_preprocessing.py and feature_extraction.py).
PYTHON_SCRIPTS=(
    "data_preprocessing.py"
    "feature_extraction.py"
    "feature_aggregation.py"
    "baseline_logistic_regression.py"
    "baseline_random_forest.py"
    "baseline_xgboost.py"
    "baseline_lightgbm.py"
    "baseline_catboost.py"
)

################################################################################
# Validation
################################################################################
echo "=========================================="
echo "Environment Check"
echo "=========================================="

if ! command -v "$PYTHON_BIN" > /dev/null 2>&1; then
    echo "ERROR: Python not found at $PYTHON_BIN"
    exit 1
fi
echo "✓ Python: $($PYTHON_BIN --version 2>&1)"
echo ""

mkdir -p $OUTPUT_DIR

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export CUDA_VISIBLE_DEVICES=0

################################################################################
################################################################################
echo "=========================================="
echo "Starting Pipeline"
echo "Total scripts: ${#PYTHON_SCRIPTS[@]}"
echo "=========================================="
echo ""

PIPELINE_START=$(date +%s)

for i in "${!PYTHON_SCRIPTS[@]}"; do
    SCRIPT="${PYTHON_SCRIPTS[$i]}"
    SCRIPT_NUM=$((i + 1))
    OUTPUT_FILE="$OUTPUT_DIR/output_${SLURM_JOB_ID}_step${SCRIPT_NUM}_$(basename $SCRIPT .py).txt"
    
    echo "=========================================="
    echo "Step $SCRIPT_NUM/${#PYTHON_SCRIPTS[@]}: $SCRIPT"
    echo "=========================================="
    echo "Started at: $(date)"
    echo "Output file: $OUTPUT_FILE"
    echo ""
    
    SCRIPT_START=$(date +%s)
    
    $PYTHON_BIN $SCRIPT 2>&1 | tee $OUTPUT_FILE
    EXIT_CODE=$?
    
    SCRIPT_END=$(date +%s)
    SCRIPT_DURATION=$((SCRIPT_END - SCRIPT_START))
    
    echo ""
    echo "Finished at: $(date)"
    echo "Duration: ${SCRIPT_DURATION}s"
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo "✓ Step $SCRIPT_NUM completed successfully"
    else
        echo "✗ Step $SCRIPT_NUM failed (exit code: $EXIT_CODE)"
        echo "Pipeline aborted at step $SCRIPT_NUM"
        exit $EXIT_CODE
    fi
    
    echo ""
done

PIPELINE_END=$(date +%s)
PIPELINE_DURATION=$((PIPELINE_END - PIPELINE_START))

echo "=========================================="
echo "Pipeline Summary"
echo "=========================================="
echo "✓ All ${#PYTHON_SCRIPTS[@]} scripts completed successfully"
echo "Total duration: ${PIPELINE_DURATION}s ($(($PIPELINE_DURATION / 60))m $(($PIPELINE_DURATION % 60))s)"
echo "=========================================="