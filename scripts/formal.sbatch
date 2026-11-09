#!/bin/bash
# Configure account, partition and available GPU constraint at your site.
#SBATCH --gres=gpu:1
#SBATCH --job-name=oat-formal
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
set -euo pipefail
cd "${OAT_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the project root}}"
export UV_PYTHON=3.11 UV_NO_SYNC=1 UV_LINK_MODE=symlink
export HF_HOME="$PWD/cache/huggingface" UV_CACHE_DIR="$PWD/cache/uv"
export LIBERO_CONFIG_PATH="$PWD/cache/libero"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TOKENIZERS_PARALLELISM=false HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$PWD/vendor:$PWD/praxis-vla/src:$PWD/scripts${PYTHONPATH:+:$PYTHONPATH}"
exec praxis-vla/.venv/bin/python -u scripts/pipeline.py "$1"
