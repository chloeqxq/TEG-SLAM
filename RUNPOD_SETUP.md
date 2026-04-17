## WildGS-SLAM RunPod Setup

This repo can be reproduced on a fresh RunPod machine, but the setup is more fragile than the upstream README suggests.

### Key Notes

- Use a GPU supported by `PyTorch 2.1 + cu118`, such as `4090`, `A5000`, `A6000`, `A100`, or `H100`.
- Do not use `RTX 5090` or other Blackwell GPUs with this pinned stack. They fail with `no kernel image is available`.
- Build the local CUDA extensions with a `CUDA 11.8` toolkit. Using the system `nvcc 12.x` with `torch==2.1.0+cu118` causes a compile-time mismatch.
- The first real run needs extra model downloads through `torch.hub` unless you prefetch them.
- The official `crowd_demo.yaml` enables GUI, which is a bad default for headless RunPod machines.

### One-Command Path

From the repo root:

```bash
bash scripts_run/setup_runpod.sh
```

If you only want to force the setup flow on an unsupported local GPU for debugging, you can override the guard:

```bash
SKIP_GPU_CHECK=1 bash scripts_run/setup_runpod.sh
```

Then:

```bash
micromamba activate wildgs-slam
python run.py ./configs/Dynamic/Wild_SLAM_Mocap/crowd_demo_headless.yaml
```

For paper-fidelity evaluation on the Wild-SLAM MoCap dataset, first download the full dynamic MoCap split:

```bash
bash scripts_downloading/download_wild_slam_mocap_scene1.sh
bash scripts_downloading/download_wild_slam_mocap_scene2.sh
```

Then run:

```bash
bash scripts_run/run_wild_slam_mocap_paper_sequence.sh crowd
bash scripts_run/run_wild_slam_mocap_paper_all.sh --skip-existing
```

Those helpers write isolated results to `./output/Wild_SLAM_Mocap_paper/` and save the batch summary to `./output/Wild_SLAM_Mocap_paper_eval.csv`.

### Manual Commands

If you do not want to use the helper script, these are the core commands:

```bash
git clone --recursive https://github.com/GradientSpaces/WildGS-SLAM.git
cd WildGS-SLAM

apt-get update
apt-get install -y curl wget unzip

mkdir -p "$HOME/.local/bin"
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
  | tar -xvj -C "$HOME/.local/bin" --strip-components=1 bin/micromamba

export MAMBA_ROOT_PREFIX="$HOME/micromamba"
eval "$("$HOME/.local/bin/micromamba" shell hook -s bash)"

micromamba create -y -n wildgs-slam -c conda-forge python=3.10 gcc_linux-64=11 gxx_linux-64=11
micromamba install -y -n wildgs-slam -c "nvidia/label/cuda-11.8.0" cuda-toolkit
set +u
micromamba activate wildgs-slam
set -u

export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export MAX_JOBS=4

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
python run.py ./configs/Dynamic/Wild_SLAM_Mocap/crowd_demo_headless.yaml

# Paper-fidelity single-sequence comparison
bash scripts_downloading/download_wild_slam_mocap_scene1.sh
bash scripts_downloading/download_wild_slam_mocap_scene2.sh
bash scripts_run/run_wild_slam_mocap_paper_sequence.sh crowd

# Paper-fidelity full MoCap sweep (resume-safe)
bash scripts_run/run_wild_slam_mocap_paper_all.sh --skip-existing
```

### Why The Extra Steps Exist

- `setup_runpod.sh` now auto-installs `curl`, `wget`, and `unzip` on Debian/Ubuntu machines, because fresh GPU containers often miss `unzip`.
- `scripts_run/patch_cuda_arch.py` removes hardcoded old SM targets from the custom CUDA extensions, so the build can follow the actual RunPod GPU instead of being stuck on legacy architectures only.
- `micromamba activate` is wrapped with `set +u` / `set -u`, because some conda-forge activation hooks fail under global nounset mode.
- `thirdparty/simple-knn/` is installed without editable mode. The editable install creates a broken import mapping on this repo layout and `from simple_knn._C import distCUDA2` fails at runtime.
- `pyyaml` and `colorama` are used at runtime but are not explicitly listed in `requirements.txt`, so they are installed separately.
- `crowd_demo_headless.yaml` disables GUI and online plotting, and lowers the output resolution to reduce the chance of out-of-memory errors on remote servers.
