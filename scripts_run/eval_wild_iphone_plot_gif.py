import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image, ImageSequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a WildGS-SLAM composite output GIF by cropping the GT/rendered RGB panels "
            "and computing full-image and background-only metrics."
        )
    )
    parser.add_argument("--gif-path", type=Path, required=True, help="Composite GIF path.")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional JSON output path. Defaults to <gif-path>.metrics.json.",
    )
    parser.add_argument(
        "--tail-count",
        type=int,
        default=26,
        help="Number of final frames to treat as the tail region. Defaults to 26.",
    )
    parser.add_argument(
        "--person-threshold",
        type=float,
        default=0.35,
        help="DeepLab person confidence threshold. Defaults to 0.35.",
    )
    parser.add_argument(
        "--dilation-radius",
        type=int,
        default=1,
        help="Max-pool dilation radius for the person mask. Defaults to 1.",
    )
    return parser.parse_args()


def largest_span(mask_1d: np.ndarray) -> tuple[int, int]:
    best_start = 0
    best_end = 0
    start = None
    for idx, value in enumerate(mask_1d):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            if idx - start > best_end - best_start:
                best_start, best_end = start, idx
            start = None
    if start is not None and len(mask_1d) - start > best_end - best_start:
        best_start, best_end = start, len(mask_1d)
    return best_start, best_end


def infer_panel_box(frame: np.ndarray, row_slice: slice, col_slice: slice) -> tuple[int, int, int, int]:
    region = frame[row_slice, col_slice]
    non_white = np.any(region < 245, axis=2)
    row_score = non_white.mean(axis=1)
    col_score = non_white.mean(axis=0)

    row_mask = row_score > 0.15
    col_mask = col_score > 0.15
    row_start, row_end = largest_span(row_mask)
    col_start, col_end = largest_span(col_mask)
    if row_end <= row_start or col_end <= col_start:
        raise RuntimeError("Failed to infer panel crop from the composite GIF.")

    y0 = row_slice.start + row_start
    y1 = row_slice.start + row_end
    x0 = col_slice.start + col_start
    x1 = col_slice.start + col_end
    return x0, y0, x1, y1


def crop_rgb(frame: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = box
    return frame[y0:y1, x0:x1, :3]


def compute_metrics(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray | None = None) -> dict:
    pred_f = pred.astype(np.float32)
    gt_f = gt.astype(np.float32)

    if mask is None:
        valid = np.ones(pred.shape[:2], dtype=bool)
    else:
        valid = mask.astype(bool)

    valid_ratio = float(valid.mean())
    if valid_ratio <= 0.0:
        return {
            "mae": float("nan"),
            "rmse": float("nan"),
            "psnr": float("nan"),
            "valid_ratio": 0.0,
        }

    valid_3 = np.repeat(valid[:, :, None], 3, axis=2)
    diff = pred_f[valid_3] - gt_f[valid_3]
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff**2)))
    psnr = float(20.0 * math.log10(255.0 / max(rmse, 1e-6)))
    return {
        "mae": mae,
        "rmse": rmse,
        "psnr": psnr,
        "valid_ratio": valid_ratio,
    }


@torch.no_grad()
def build_person_mask(
    image: np.ndarray,
    model,
    preprocess,
    device: str,
    person_threshold: float,
    dilation_radius: int,
) -> np.ndarray:
    input_tensor = preprocess(Image.fromarray(image)).unsqueeze(0).to(device)
    logits = model(input_tensor)["out"]
    logits = F.interpolate(
        logits, size=image.shape[:2], mode="bilinear", align_corners=False
    )
    person_prob = torch.softmax(logits, dim=1)[:, 15:16]
    person_prob = person_prob.squeeze(0).squeeze(0)
    person_prob = torch.clamp(
        (person_prob - person_threshold) / max(1.0 - person_threshold, 1e-6),
        min=0.0,
        max=1.0,
    )
    if dilation_radius > 0:
        k = 2 * dilation_radius + 1
        person_prob = F.max_pool2d(
            person_prob.unsqueeze(0).unsqueeze(0), kernel_size=k, stride=1, padding=dilation_radius
        ).squeeze(0).squeeze(0)
    return (person_prob > 0).detach().cpu().numpy()


def main() -> None:
    args = parse_args()
    gif_path = args.gif_path.resolve()
    if not gif_path.exists():
        raise FileNotFoundError(f"GIF does not exist: {gif_path}")

    image = Image.open(gif_path)
    frames = [np.array(frame.convert("RGB")) for frame in ImageSequence.Iterator(image)]
    if not frames:
        raise RuntimeError(f"No frames found in GIF: {gif_path}")

    h, w = frames[0].shape[:2]
    gt_box = infer_panel_box(frames[0], slice(0, h // 2), slice(0, w // 4))
    rendered_box = infer_panel_box(frames[0], slice(h // 2, h), slice(0, w // 4))

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    weights = torchvision.models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT
    preprocess = weights.transforms()
    model = torchvision.models.segmentation.deeplabv3_resnet50(weights=weights).to(device)
    model.eval()

    frame_metrics = []
    background_psnr = []
    background_mae = []
    all_psnr = []
    all_mae = []

    for frame_idx, frame in enumerate(frames):
        gt = crop_rgb(frame, gt_box)
        rendered = crop_rgb(frame, rendered_box)
        if gt.shape != rendered.shape:
            rendered = np.array(Image.fromarray(rendered).resize((gt.shape[1], gt.shape[0])))

        person_mask = build_person_mask(
            gt,
            model=model,
            preprocess=preprocess,
            device=device,
            person_threshold=args.person_threshold,
            dilation_radius=args.dilation_radius,
        )
        background_mask = ~person_mask

        bg_metrics = compute_metrics(rendered, gt, mask=background_mask)
        all_metrics = compute_metrics(rendered, gt, mask=None)
        frame_metrics.append(
            {
                "frame_idx": frame_idx,
                "background_valid_ratio": bg_metrics["valid_ratio"],
                "background": bg_metrics,
                "all": all_metrics,
                "person_mask_ratio": float(person_mask.mean()),
            }
        )

        background_psnr.append(bg_metrics["psnr"])
        background_mae.append(bg_metrics["mae"])
        all_psnr.append(all_metrics["psnr"])
        all_mae.append(all_metrics["mae"])

    tail_count = min(args.tail_count, len(frame_metrics))
    tail_metrics = frame_metrics[-tail_count:] if tail_count > 0 else []

    def safe_mean(values: list[float]) -> float:
        return float(np.nanmean(np.asarray(values, dtype=np.float32)))

    output = {
        "gif_path": str(gif_path),
        "summary": {
            "frame_count": len(frame_metrics),
            "mean_background_psnr": safe_mean(background_psnr),
            "mean_background_mae": safe_mean(background_mae),
            "mean_all_psnr": safe_mean(all_psnr),
            "mean_all_mae": safe_mean(all_mae),
            "tail_frame_count": len(tail_metrics),
            "tail_mean_background_psnr": safe_mean([item["background"]["psnr"] for item in tail_metrics]),
            "tail_mean_background_mae": safe_mean([item["background"]["mae"] for item in tail_metrics]),
            "tail_mean_all_psnr": safe_mean([item["all"]["psnr"] for item in tail_metrics]),
            "tail_mean_all_mae": safe_mean([item["all"]["mae"] for item in tail_metrics]),
            "mean_person_mask_ratio": safe_mean([item["person_mask_ratio"] for item in frame_metrics]),
        },
        "crop_boxes": {
            "ground_truth_rgb": {"x0": gt_box[0], "y0": gt_box[1], "x1": gt_box[2], "y1": gt_box[3]},
            "rendered_rgb": {
                "x0": rendered_box[0],
                "y0": rendered_box[1],
                "x1": rendered_box[2],
                "y1": rendered_box[3],
            },
        },
        "frames": frame_metrics,
    }

    output_json = args.output_json or gif_path.with_suffix(".metrics.json")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output["summary"], indent=2))


if __name__ == "__main__":
    main()
