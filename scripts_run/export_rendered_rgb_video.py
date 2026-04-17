import argparse
import re
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from munch import munchify
from tqdm import tqdm

from src.utils.camera_utils import Camera
from src.utils.datasets import get_dataset
from thirdparty.gaussian_splatting.gaussian_renderer import render
from thirdparty.gaussian_splatting.scene.gaussian_model import GaussianModel
from thirdparty.gaussian_splatting.utils.graphics_utils import (
    focal2fov,
    getProjectionMatrix2,
)


FRAME_PATTERN = re.compile(r"video_idx_(\d+)_kf_idx_(\d+)(?:_.+)?\.png$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render raw RGB frames directly from final_gs.ply and stitch them into an MP4. "
            "This bypasses matplotlib plot screenshots."
        )
    )
    parser.add_argument(
        "--exp-dir",
        type=Path,
        required=True,
        help="Experiment output directory that contains cfg.yaml, video.npz and final_gs.ply.",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=None,
        help=(
            "Directory with plots_after_refine PNGs used only to pick which keyframes to render. "
            "Defaults to <exp-dir>/plots_after_refine."
        ),
    )
    parser.add_argument(
        "--frame-dir",
        type=Path,
        default=None,
        help="Directory to save the re-rendered raw RGB frames.",
    )
    parser.add_argument(
        "--video-path",
        type=Path,
        default=None,
        help="Output MP4 path.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=10.0,
        help="Output video FPS. Defaults to 10.0 to match the existing GIF cadence.",
    )
    return parser.parse_args()


def load_cfg(cfg_path: Path) -> dict:
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_plot_frame_specs(plots_dir: Path) -> list[tuple[int, int]]:
    frame_specs: list[tuple[int, int]] = []
    for path in plots_dir.iterdir():
        if not path.is_file() or path.suffix.lower() != ".png":
            continue
        match = FRAME_PATTERN.match(path.name)
        if match is None:
            continue
        frame_specs.append((int(match.group(1)), int(match.group(2))))

    frame_specs.sort(key=lambda item: (item[0], item[1]))
    if not frame_specs:
        raise RuntimeError(f"No video_idx_*_kf_idx_*.png files found in {plots_dir}")
    return frame_specs


def get_render_camera_params(frame_reader, full_resolution: bool, device: str):
    if full_resolution:
        intrinsic_full = frame_reader.get_intrinsic_full_resol()
        fx, fy, cx, cy = [intrinsic_full[i].item() for i in range(4)]
        width, height = frame_reader.W_out_full, frame_reader.H_out_full
    else:
        fx, fy, cx, cy = (
            frame_reader.fx,
            frame_reader.fy,
            frame_reader.cx,
            frame_reader.cy,
        )
        width, height = frame_reader.W_out, frame_reader.H_out

    projection_matrix = getProjectionMatrix2(
        znear=0.01,
        zfar=100.0,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        W=width,
        H=height,
    ).transpose(0, 1).to(device=device)

    return {
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "width": width,
        "height": height,
        "fovx": focal2fov(fx, width),
        "fovy": focal2fov(fy, height),
        "projection_matrix": projection_matrix,
    }


def make_viewpoint(video_idx: int, w2c: torch.Tensor, cam_params: dict, device: str) -> Camera:
    viewpoint = Camera(
        uid=video_idx,
        color=None,
        depth=None,
        gt_T=w2c,
        projection_matrix=cam_params["projection_matrix"],
        fx=cam_params["fx"],
        fy=cam_params["fy"],
        cx=cam_params["cx"],
        cy=cam_params["cy"],
        fovx=cam_params["fovx"],
        fovy=cam_params["fovy"],
        image_height=cam_params["height"],
        image_width=cam_params["width"],
        features=None,
        device=device,
    )
    viewpoint.update_RT(w2c[:3, :3], w2c[:3, 3])
    return viewpoint


def write_video(frame_paths: list[Path], video_path: Path, fps: float) -> None:
    if not frame_paths:
        raise RuntimeError("No frames were rendered, cannot create video.")

    first_frame = cv2.imread(str(frame_paths[0]), cv2.IMREAD_COLOR)
    if first_frame is None:
        raise RuntimeError(f"Failed to read first rendered frame: {frame_paths[0]}")

    height, width = first_frame.shape[:2]
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for: {video_path}")

    try:
        for frame_path in frame_paths:
            frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if frame is None:
                raise RuntimeError(f"Failed to read rendered frame: {frame_path}")
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LANCZOS4)
            writer.write(frame)
    finally:
        writer.release()


def main() -> None:
    args = parse_args()

    exp_dir = args.exp_dir.resolve()
    if not exp_dir.exists():
        raise FileNotFoundError(f"exp_dir does not exist: {exp_dir}")

    plots_dir = (args.plots_dir or (exp_dir / "plots_after_refine")).resolve()
    if not plots_dir.exists():
        raise FileNotFoundError(f"plots_dir does not exist: {plots_dir}")

    iteration_tag = plots_dir.name[6:] if plots_dir.name.startswith("plots_") else plots_dir.name
    frame_dir = (args.frame_dir or (exp_dir / f"rendered_rgb_{iteration_tag}")).resolve()
    video_path = (args.video_path or (exp_dir / f"rendered_rgb_{iteration_tag}.mp4")).resolve()
    frame_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_cfg(exp_dir / "cfg.yaml")
    device = cfg.get("device", "cuda:0")
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"Config requests CUDA device '{device}', but CUDA is not available in this environment."
        )

    frame_specs = parse_plot_frame_specs(plots_dir)
    offline_video = np.load(exp_dir / "video.npz")
    poses = offline_video["poses"]
    timestamps = offline_video["timestamps"]

    if len(poses) == 0:
        raise RuntimeError(f"No poses found in {exp_dir / 'video.npz'}")

    frame_reader = get_dataset(cfg, device=device)
    full_resolution = bool(cfg["mapping"]["full_resolution"])
    cam_params = get_render_camera_params(frame_reader, full_resolution, device)

    use_spherical_harmonics = cfg["mapping"]["Training"]["spherical_harmonics"]
    sh_degree = 3 if use_spherical_harmonics else 0
    gaussians = GaussianModel(sh_degree, config=cfg)
    gaussians.load_ply(str(exp_dir / "final_gs.ply"))

    pipeline_params = munchify(cfg["mapping"]["pipeline_params"])
    background = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=device)

    rendered_frame_paths: list[Path] = []
    mismatch_count = 0

    for video_idx, parsed_frame_idx in tqdm(frame_specs, desc="Rendering raw RGB frames"):
        if video_idx < 0 or video_idx >= len(poses):
            raise IndexError(
                f"video_idx {video_idx} from {plots_dir} is out of range for {exp_dir / 'video.npz'}"
            )

        c2w = torch.from_numpy(poses[video_idx]).float().to(device)
        w2c = torch.linalg.inv(c2w)
        timestamp = int(timestamps[video_idx])
        if timestamp != parsed_frame_idx:
            mismatch_count += 1

        viewpoint = make_viewpoint(video_idx, w2c, cam_params, device)
        with torch.no_grad():
            render_pkg = render(viewpoint, gaussians, pipeline_params, background)
            image = torch.clamp(render_pkg["render"], 0.0, 1.0)

        rendered_rgb = (image.permute(1, 2, 0).detach().cpu().numpy() * 255.0).astype(np.uint8)
        frame_path = frame_dir / f"video_idx_{video_idx}_kf_idx_{timestamp}.png"
        ok = cv2.imwrite(str(frame_path), cv2.cvtColor(rendered_rgb, cv2.COLOR_RGB2BGR))
        if not ok:
            raise RuntimeError(f"Failed to write rendered frame: {frame_path}")
        rendered_frame_paths.append(frame_path)

    write_video(rendered_frame_paths, video_path, fps=args.fps)

    print(f"saved_frames_dir={frame_dir}")
    print(f"saved_video_path={video_path}")
    print(f"frame_count={len(rendered_frame_paths)}")
    print(f"fps={args.fps}")
    if mismatch_count > 0:
        print(f"warning_timestamp_mismatches={mismatch_count}")


if __name__ == "__main__":
    main()
