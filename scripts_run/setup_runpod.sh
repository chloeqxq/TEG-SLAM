#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${ENV_NAME:-wildgs-slam}"
MICROMAMBA_BIN="${MICROMAMBA_BIN:-$HOME/.local/bin/micromamba}"
MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/micromamba}"

cd "$REPO_ROOT"

missing_system_packages=()
for pkg in curl wget unzip; do
  if ! command -v "$pkg" >/dev/null 2>&1; then
    missing_system_packages+=("$pkg")
  fi
done

if [ "${#missing_system_packages[@]}" -gt 0 ]; then
  if command -v apt-get >/dev/null 2>&1; then
    apt_prefix=()
    if [ "$(id -u)" -ne 0 ]; then
      if command -v sudo >/dev/null 2>&1; then
        apt_prefix=(sudo)
      else
        echo "Missing system packages: ${missing_system_packages[*]}"
        echo "Install them first, then rerun:"
        echo "  apt-get update && apt-get install -y ${missing_system_packages[*]}"
        exit 1
      fi
    fi

    "${apt_prefix[@]}" apt-get update
    "${apt_prefix[@]}" apt-get install -y "${missing_system_packages[@]}"
  else
    echo "Missing system packages: ${missing_system_packages[*]}"
    echo "Please install them before rerunning the setup script."
    exit 1
  fi
fi

if [ "${SKIP_GPU_CHECK:-0}" != "1" ] && command -v nvidia-smi >/dev/null 2>&1; then
  GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1 || true)"
  case "$GPU_NAME" in
    *5090*|*5080*|*5070*|*5060*|*Blackwell*)
      echo "Detected unsupported GPU for the pinned WildGS-SLAM stack: $GPU_NAME"
      echo "PyTorch 2.1 + cu118 does not support Blackwell GPUs yet."
      echo "Use a RunPod GPU like 4090, A5000, A6000, A100, or H100 for this setup."
      exit 1
      ;;
  esac
fi

if [ ! -x "$MICROMAMBA_BIN" ]; then
  mkdir -p "$(dirname "$MICROMAMBA_BIN")"
  curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
    | tar -xvj -C "$(dirname "$MICROMAMBA_BIN")" --strip-components=1 bin/micromamba
fi

export MAMBA_ROOT_PREFIX
eval "$("$MICROMAMBA_BIN" shell hook -s bash)"

if ! "$MICROMAMBA_BIN" run -n "$ENV_NAME" python -V >/dev/null 2>&1; then
  micromamba create -y -n "$ENV_NAME" -c conda-forge python=3.10 gcc_linux-64=11 gxx_linux-64=11
  micromamba install -y -n "$ENV_NAME" -c "nvidia/label/cuda-11.8.0" cuda-toolkit
fi

# Some conda-forge activation hooks assume unset toolchain variables can be
# expanded safely, which breaks under this script's global `set -u`.
set +u
micromamba activate "$ENV_NAME"
set -u

export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export MAX_JOBS="${MAX_JOBS:-4}"

git submodule update --init --recursive thirdparty/lietorch thirdparty/diff-gaussian-rasterization-w-pose thirdparty/simple-knn
python scripts_run/patch_cuda_arch.py

python -m pip install wheel setuptools==78.1.1 ninja numpy==1.26.3
python -m pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
python -m pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-2.1.0+cu118.html
python -m pip install -U xformers==0.0.22.post7+cu118 --index-url https://download.pytorch.org/whl/cu118

python -m pip install --no-build-isolation -e thirdparty/lietorch/
python -m pip install --no-build-isolation -e thirdparty/diff-gaussian-rasterization-w-pose/
python -m pip install --no-build-isolation thirdparty/simple-knn/
python -m pip install --no-build-isolation -e .

python - <<'PY'
import importlib
import torch

for module_name in (
    "lietorch",
    "diff_gaussian_rasterization",
    "simple_knn._C",
    "droid_backends",
):
    importlib.import_module(module_name)

print("Core CUDA extensions import correctly.")
PY

python -m pip install pyyaml colorama gdown
python -m pip install -r requirements.txt
python -m pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1.0/index.html

mkdir -p pretrained
python -m gdown --fuzzy "https://drive.google.com/file/d/1PpqVt1H4maBa_GbPJp4NwxRsd9jk-elh/view?usp=sharing" -O pretrained/droid.pth

python - <<'PY'
import torch

torch.hub.load("yvanyin/metric3d", "metric3d_vit_large", pretrain=True, trust_repo=True)
torch.hub.load("ywyue/FiT3D", "dinov2_reg_small_fine", trust_repo=True)

print("Torch hub caches are ready.")
PY

bash scripts_downloading/download_demo_data.sh

echo
echo "Setup finished."
echo "Activate env:"
echo "  micromamba activate $ENV_NAME"
echo
echo "Headless demo command:"
echo "  python run.py ./configs/Dynamic/Wild_SLAM_Mocap/crowd_demo_headless.yaml"
echo
echo "Paper-fidelity single-sequence command:"
echo "  bash scripts_downloading/download_wild_slam_mocap_scene1.sh"
echo "  bash scripts_downloading/download_wild_slam_mocap_scene2.sh"
echo "  bash scripts_run/run_wild_slam_mocap_paper_sequence.sh crowd"
echo
echo "Paper-fidelity full MoCap sweep:"
echo "  bash scripts_run/run_wild_slam_mocap_paper_all.sh --skip-existing"
