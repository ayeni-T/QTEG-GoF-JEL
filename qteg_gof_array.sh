#!/bin/bash
#SBATCH --job-name=QTEG_GoF
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=qCPU120
#SBATCH --array=0-71
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=48:00:00
#SBATCH --output=/path/to/your/project/logs/qteg_gof_%A_%a.out
#SBATCH --error=/path/to/your/project/logs/qteg_gof_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=YOUR_EMAIL@institution.edu

mkdir -p /path/to/your/project/logs
mkdir -p /path/to/your/project/results

module load miniconda3/25.5.1
eval "$(conda shell.bash hook)"
conda activate /path/to/your/conda/env

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# Block layout (72 total):
#   0-11:   size study  (sc_idx = block//4, n_idx = block%4)
#   12-71:  power study (3 scenarios x 5 alternatives x 4 sample sizes)
#           pb = block-12
#           sc_idx  = pb // 20
#           alt_idx = (pb % 20) // 4
#           n_idx   = pb % 4
#
# Tests: JEL + BC-JEL (chi2_1) + KS + AD + CvM (B=300 bootstrap each)
# Heaviest block: n=200 ~ 17.5h wall time (within 48h limit)

echo "============================================"
echo "Job:     QTEG GoF -- Full Power (3 scenarios)"
echo "Block:   ${SLURM_ARRAY_TASK_ID}"
echo "Job ID:  ${SLURM_JOB_ID}"
echo "Node:    $(hostname)"
echo "Python:  $(which python)"
echo "Started: $(date)"
echo "CPUs:    ${SLURM_CPUS_PER_TASK}"
echo "============================================"

if [ "${SLURM_ARRAY_TASK_ID}" -lt 12 ]; then
    SC_IDX=$(( SLURM_ARRAY_TASK_ID / 4 ))
    N_IDX=$((  SLURM_ARRAY_TASK_ID % 4 ))
    echo "Study:   SIZE (scenario ${SC_IDX}, n_idx ${N_IDX})"
else
    PB=$(( SLURM_ARRAY_TASK_ID - 12 ))
    SC_IDX=$(( PB / 20 ))
    REM=$(( PB % 20 ))
    ALT_IDX=$(( REM / 4 ))
    N_IDX=$((   REM % 4 ))
    echo "Study:   POWER (scenario ${SC_IDX}, alt_idx ${ALT_IDX}, n_idx ${N_IDX})"
fi
echo "============================================"

python /path/to/your/project/QTEG_GoF_Arctic.py \
    --block ${SLURM_ARRAY_TASK_ID} \
    --n_sim 5000 \
    --B 300

EXIT_CODE=$?
echo "Block ${SLURM_ARRAY_TASK_ID} finished at $(date) | Exit: ${EXIT_CODE}"
exit ${EXIT_CODE}
