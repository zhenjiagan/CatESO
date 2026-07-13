#!/bin/bash
# =============================================================================
# One-shot, SELF-CONTAINED environment setup for CatESO-FOLD.
#
#     bash scripts/setup_env.sh
# It:
#   1) creates the `cateso` conda env (Python 3.7 / CUDA 11.3 / PyTorch 1.12 …)
#   2) installs the full pip stack (DGL + CatESO app deps + ESMFold/OpenFold)
#
# =============================================================================
# NB: no `set -u` — conda's binutils activate.d script references $HOST unbound,
# which nounset would turn into a fatal error during `conda activate`.
set -eo pipefail
export HOST="${HOST:-$(hostname)}"

ENV_NAME="${CATESO_ENV_NAME:-cateso}"

# --- keep conda/pip off $HOME (inode quota) ---------------------------------
export CONDA_ENVS_DIRS="${CONDA_ENVS_DIRS:-/scratch/$USER/conda_envs}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-/scratch/$USER/.conda/pkgs}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/scratch/$USER/.cache/pip}"
mkdir -p "$CONDA_ENVS_DIRS" "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR"

# --- embedded specs written to a scratch temp dir ----------------------------
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

cat > "$WORK/environment.yml" <<'YAML'
name: cateso
channels:
  - conda-forge
  - bioconda
  - pytorch
dependencies:
  - conda-forge::python=3.7
  - conda-forge::setuptools=59.5.0
  - conda-forge::pip
  - conda-forge::openmm=7.5.1
  - conda-forge::pdbfixer
  - conda-forge::cudatoolkit==11.3.*
  - conda-forge::cudatoolkit-dev==11.3.*
  - conda-forge::einops==0.6.1
  - conda-forge::fairscale
  - conda-forge::mkl=2024.0.0
  - conda-forge::omegaconf
  - conda-forge::hydra-core
  - conda-forge::pandas
  - conda-forge::pytest
  - bioconda::hmmer==3.3.2
  - bioconda::hhsuite==3.3.0
  - bioconda::kalign2==2.04
  - pytorch::pytorch=1.12.*
  - ehmoussi::gxx_linux-64
YAML

cat > "$WORK/requirements.txt" <<'PIP'
# DGL (CUDA 11.3): wheels live on the DGL index, not PyPI
-f https://data.dgl.ai/wheels/repo.html
dgl-cu113

# CatESO application deps
dgllife==0.2.8
matplotlib==3.5.3
prettytable==3.7.0
safetensors==0.4.1
timm
yacs
rdkit
transformers

# ESMFold / OpenFold stack
biopython==1.79
deepspeed==0.5.9
dm-tree==0.1.6
ml-collections==0.1.0
numpy==1.21.2
PyYAML==5.4.1
requests==2.26.0
scipy==1.7.1
tqdm==4.62.2
typing-extensions==3.10.0.2
pytorch_lightning==1.5.10
wandb==0.12.21
git+https://github.com/NVIDIA/dllogger.git
git+https://github.com/aqlaboratory/openfold.git@4b41059694619831a7db195b7e0988fc4ff3a307
git+https://github.com/facebookresearch/esm.git
fair-esm[esmfold]
biotite
PIP

# --- conda -------------------------------------------------------------------
# `module` is undefined in non-login shells (e.g. sbatch #!/bin/bash), so source
# the Lmod/Environment-Modules init before loading anaconda.
source /etc/profile.d/lmod.sh 2>/dev/null || source /etc/profile.d/modules.sh 2>/dev/null || true
module load anaconda3/2025.06
source "$(conda info --base)/etc/profile.d/conda.sh"

ENV_PREFIX="$CONDA_ENVS_DIRS/$ENV_NAME"

echo "[1/2] Creating conda env '$ENV_NAME' at $ENV_PREFIX ..."
if [ -d "$ENV_PREFIX" ]; then
    echo "      env already exists -> updating"
    conda env update -n "$ENV_NAME" -f "$WORK/environment.yml" --prune
else
    conda env create -n "$ENV_NAME" -f "$WORK/environment.yml"
fi

conda activate "$ENV_NAME"

echo "[2/2] Installing pip dependencies ..."
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
python -m pip install -r "$WORK/requirements.txt"