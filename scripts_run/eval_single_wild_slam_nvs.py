import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch
import yaml
from scipy.spatial.transform import Rotation as R
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval_map.utils import (
    check_quality_of_transformation,
    get_render_pipline_params,
    get_temp_viewpoint,
    line_to_T,
)
from src.utils.datasets import get_dataset
from thirdparty.gaussian_splatting.gaussian_renderer import render
from thirdparty.gaussian_splatting.scene.gaussian_model import GaussianModel
from thirdparty.gaussian_splatting.utils.image_utils import psnr
from thirdparty.gaussian_splatting.utils.loss_utils import ssim


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate WildGS-SLAM novel view synthesis for a single MoCap experiment."
    )
    parser.add_argument(
        "--exp-dir",
        type=Path,
        required=True,
        help="Experiment directory containing cfg.yaml, final_gs.ply, and trajectory outputs.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional JSON path. Defaults to <exp-dir>/nvs/final_result.json.",
    )
    parser.add_argument(
        "--full-resol",
        action="store_true",
        help="Render evaluation at full resolution instead of the mapping resolution.",
    )
    return parser.parse_args()


def load_cfg(cfg_path: Path) -> dict:
    with cfg_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def get_resized_color(frame_reader, img: np.ndarray) -> np.ndarray:
    return cv2.resize(img, (frame_reader.W_out_with_edge, frame_reader.H_out_with_edge))


def main() -> None:
    args = parse_args()
    exp_dir = args.exp_dir.resolve()
    cfg = load_cfg(exp_dir / "cfg.yaml")
    device = cfg.get("device", "cuda:0")

    data_folder = cfg["data"]["input_folder"].replace(
        "ROOT_FOLDER_PLACEHOLDER", cfg["data"]["root_folder"]
    )
    cfg["data"]["input_folder"] = data_folder

    frame_reader = get_dataset(cfg, device=device)
    gt_poses = frame_reader.poses

    est_pose_path = exp_dir / "traj" / "est_poses_full.txt"
    metric_path = exp_dir / "traj" / "metrics_full_traj.txt"
    if not est_pose_path.exists():
        raise FileNotFoundError(f"Missing estimated trajectory: {est_pose_path}")
    if not metric_path.exists():
        raise FileNotFoundError(f"Missing trajectory metrics: {metric_path}")

    est_lines = est_pose_path.read_text(encoding="utf-8").splitlines()
    metric_lines = metric_path.read_text(encoding="utf-8").splitlines()
    scale_line = next((line for line in metric_lines if "scale:" in line), None)
    if scale_line is None:
        raise RuntimeError(f"Failed to parse scale from {metric_path}")
    traj_scale = float(scale_line.replace("scale:", "").strip())

    trans, rot = [], []
    w2c_first_inv = np.linalg.inv(frame_reader.w2c_first_pose)
    for idx, line in enumerate(est_lines):
        t_we_c = line_to_T(line)
        t_we_c[:3, 3] *= traj_scale
        t_wg_c = w2c_first_inv @ gt_poses[idx]
        t_we_wg_curr = t_we_c @ np.linalg.inv(t_wg_c)
        rot.append(t_we_wg_curr[:3, :3])
        trans.append(t_we_wg_curr[:3, 3])

    t_we_wg = np.eye(4)
    t_we_wg[:3, :3] = R.from_matrix(np.array(rot)).mean().as_matrix()
    t_we_wg[:3, 3] = np.mean(trans, axis=0)
    check_quality_of_transformation(trans, rot, t_we_wg)

    nvs_folder = Path(data_folder) / "nvs"
    static_poses_gt = [
        line
        for line in (nvs_folder / "groundtruth.txt").read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    ]
    intrins_per_frame = json.loads(
        (nvs_folder / "per_frame_intrinsics.json").read_text(encoding="utf-8")
    )

    gaussians = GaussianModel(0, config=None)
    gaussians.load_ply(str(exp_dir / "final_gs.ply"))
    pipe = get_render_pipline_params(str(exp_dir))

    lpips_metric = LearnedPerceptualImagePatchSimilarity(
        net_type="alex", normalize=True
    ).to(device)
    background = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=device)

    psnr_scores, ssim_scores, lpips_scores = [], [], []
    frame_metrics = []

    for i in range(len(intrins_per_frame)):
        t_wg_c = line_to_T(static_poses_gt[i])
        t_we_c = t_we_wg @ t_wg_c
        t_we_c[:3, 3] /= traj_scale
        t_c_we = torch.tensor(np.linalg.inv(t_we_c), device=device, dtype=torch.float32)

        this_intrinsic = intrins_per_frame[str(i)]
        viewpoint = get_temp_viewpoint(
            this_intrinsic, full_resol=args.full_resol, exp_cfg=cfg
        )
        viewpoint.update_RT(t_c_we[:3, :3], t_c_we[:3, 3])

        with torch.no_grad():
            rendering_pkg = render(viewpoint, gaussians, pipe, background)
            image = torch.clamp(rendering_pkg["render"], 0.0, 1.0)

        input_rgb = cv2.imread(str(nvs_folder / f"rgb/nvs_{i:05d}.png"))
        k = np.eye(3)
        k[0, 0], k[0, 2], k[1, 1], k[1, 2] = (
            this_intrinsic["fx"],
            this_intrinsic["ppx"],
            this_intrinsic["fy"],
            this_intrinsic["ppy"],
        )
        input_rgb = cv2.undistort(input_rgb, k, np.array(this_intrinsic["coeffs"]))
        input_rgb = get_resized_color(frame_reader, input_rgb)

        gt_image = torch.from_numpy(input_rgb).float().permute(2, 0, 1).to(device) / 255.0
        mask = gt_image > 0

        psnr_score = psnr(image[mask].unsqueeze(0), gt_image[mask].unsqueeze(0)).item()
        ssim_score = ssim(image.unsqueeze(0), gt_image.unsqueeze(0)).item()
        lpips_score = lpips_metric(image.unsqueeze(0), gt_image.unsqueeze(0)).item()

        psnr_scores.append(psnr_score)
        ssim_scores.append(ssim_score)
        lpips_scores.append(lpips_score)
        frame_metrics.append(
            {
                "nvs_idx": i,
                "psnr": float(psnr_score),
                "ssim": float(ssim_score),
                "lpips": float(lpips_score),
            }
        )

    output = {
        "scene": exp_dir.name,
        "frame_count": len(frame_metrics),
        "mean_psnr": float(np.mean(psnr_scores)),
        "mean_ssim": float(np.mean(ssim_scores)),
        "mean_lpips": float(np.mean(lpips_scores)),
        "frames": frame_metrics,
    }

    output_json = args.output_json or (exp_dir / "nvs" / "final_result.json")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
