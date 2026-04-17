import argparse
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image, ImageDraw
from simple_lama_inpainting import SimpleLama


def parse_video_idx(plot_path: Path) -> int:
    parts = plot_path.stem.split("_")
    if len(parts) < 3:
        raise ValueError(f"Unexpected plot filename: {plot_path.name}")
    return int(parts[2])


def extract_gt_and_render_panels(plot_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w, _ = plot_rgb.shape
    gt_panel = plot_rgb[0 : h // 2, 0 : w // 4, :]
    render_panel = plot_rgb[h // 2 : h, 0 : w // 4, :]
    return gt_panel, render_panel


def draw_compare_title(frame: np.ndarray, title: str) -> np.ndarray:
    frame_pil = Image.fromarray(frame)
    canvas = Image.new("RGB", (frame_pil.width, frame_pil.height + 24), (255, 255, 255))
    canvas.paste(frame_pil, (0, 24))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 4), title, fill=(0, 0, 0))
    return np.array(canvas)


def predict_person_prob(
    image_rgb: np.ndarray,
    seg_model,
    seg_preprocess,
    device: torch.device,
    output_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    image_pil = Image.fromarray(image_rgb)
    with torch.no_grad():
        inp = seg_preprocess(image_pil).unsqueeze(0).to(device)
        logits = seg_model(inp)["out"]
        if output_shape is None:
            output_shape = image_rgb.shape[:2]
        logits = F.interpolate(
            logits,
            size=output_shape,
            mode="bilinear",
            align_corners=False,
        )
        person_prob = torch.softmax(logits, dim=1)[0, 15].detach().cpu().numpy()
    return person_prob.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Demo-only person eraser for wandering rendered panels using DeepLab + LaMa."
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        required=True,
        help="Path to plots_after_refine directory that contains video_idx_* PNG files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Root output dir. Defaults to parent of --plots-dir.",
    )
    parser.add_argument(
        "--start-video-idx",
        type=int,
        default=40,
        help="Only apply inpainting when video_idx >= this threshold.",
    )
    parser.add_argument(
        "--end-video-idx",
        type=int,
        default=None,
        help="Optional upper bound for video_idx to inpaint.",
    )
    parser.add_argument(
        "--person-prob-threshold",
        type=float,
        default=0.30,
        help="DeepLab person probability threshold.",
    )
    parser.add_argument(
        "--mask-dilation",
        type=int,
        default=5,
        help="Mask dilation kernel size (odd number recommended).",
    )
    parser.add_argument(
        "--min-mask-pixels",
        type=int,
        default=20,
        help="Skip inpainting when person mask has fewer pixels than this.",
    )
    parser.add_argument(
        "--gif-duration",
        type=float,
        default=0.16,
        help="Duration per frame in output GIFs.",
    )
    parser.add_argument(
        "--temporal-mask-window",
        type=int,
        default=0,
        help="Average person probabilities over +/- this many neighboring frames before thresholding.",
    )
    args = parser.parse_args()

    plots_dir = args.plots_dir
    if not plots_dir.exists():
        raise FileNotFoundError(f"plots_dir does not exist: {plots_dir}")

    output_root = args.output_root if args.output_root is not None else plots_dir.parent
    end_tag = "end" if args.end_video_idx is None else str(args.end_video_idx)
    run_tag = f"v{args.start_video_idx}_to_{end_tag}"
    render_dir = output_root / f"plots_after_refine_lama_cond_render_{run_tag}"
    compare_dir = output_root / f"plots_after_refine_lama_cond_compare_{run_tag}"
    render_dir.mkdir(parents=True, exist_ok=True)
    compare_dir.mkdir(parents=True, exist_ok=True)

    plot_files = sorted(
        p for p in plots_dir.iterdir() if p.name.startswith("video_idx_") and p.suffix == ".png"
    )
    if not plot_files:
        raise RuntimeError(f"No video_idx_*.png found in {plots_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seg_weights = torchvision.models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT
    seg_model = torchvision.models.segmentation.deeplabv3_resnet50(weights=seg_weights).to(device)
    seg_model.eval()
    seg_preprocess = seg_weights.transforms()
    lama = SimpleLama()

    render_frames = []
    compare_frames = []
    entries = []

    kernel_size = max(1, int(args.mask_dilation))
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    for i, plot_path in enumerate(plot_files, start=1):
        video_idx = parse_video_idx(plot_path)
        plot_rgb = np.array(Image.open(plot_path).convert("RGB"))
        gt_panel, render_panel = extract_gt_and_render_panels(plot_rgb)

        render_h, render_w = render_panel.shape[:2]
        person_prob = predict_person_prob(
            gt_panel,
            seg_model=seg_model,
            seg_preprocess=seg_preprocess,
            device=device,
            output_shape=(render_h, render_w),
        )
        entries.append(
            {
                "plot_path": plot_path,
                "video_idx": video_idx,
                "render_panel": render_panel,
                "person_prob": person_prob,
            }
        )

        if i % 10 == 0 or i == len(plot_files):
            print(f"prepared {i}/{len(plot_files)}")

    processed_frames = 0
    total_mask_pixels = 0
    total_person_pixels_before = 0
    total_person_pixels_after = 0

    for i, entry in enumerate(entries, start=1):
        render_panel = entry["render_panel"]
        render_h, render_w = render_panel.shape[:2]
        neighbor_start = max(0, i - 1 - args.temporal_mask_window)
        neighbor_end = min(len(entries), i + args.temporal_mask_window)
        prob_stack = [entries[j]["person_prob"] for j in range(neighbor_start, neighbor_end)]
        smoothed_prob = np.mean(prob_stack, axis=0)

        person_mask = (smoothed_prob > args.person_prob_threshold).astype(np.uint8)
        person_mask = cv2.dilate(person_mask, kernel, iterations=1)
        person_mask_u8 = (person_mask * 255).astype(np.uint8)

        within_range = entry["video_idx"] >= args.start_video_idx
        if args.end_video_idx is not None:
            within_range = within_range and entry["video_idx"] <= args.end_video_idx

        if within_range and person_mask.sum() > args.min_mask_pixels:
            inpainted = np.array(
                lama(Image.fromarray(render_panel), Image.fromarray(person_mask_u8))
            )
            if inpainted.shape[:2] != (render_h, render_w):
                inpainted = cv2.resize(
                    inpainted, (render_w, render_h), interpolation=cv2.INTER_CUBIC
                )

            before_prob = predict_person_prob(
                render_panel,
                seg_model=seg_model,
                seg_preprocess=seg_preprocess,
                device=device,
                output_shape=(render_h, render_w),
            )
            after_prob = predict_person_prob(
                inpainted,
                seg_model=seg_model,
                seg_preprocess=seg_preprocess,
                device=device,
                output_shape=(render_h, render_w),
            )
            total_person_pixels_before += int(
                (before_prob > args.person_prob_threshold).sum()
            )
            total_person_pixels_after += int(
                (after_prob > args.person_prob_threshold).sum()
            )
            total_mask_pixels += int(person_mask.sum())
            processed_frames += 1
        else:
            inpainted = render_panel.copy()

        stem = entry["plot_path"].stem
        render_out_path = render_dir / f"{stem}_render_lama_cond.png"
        Image.fromarray(inpainted).save(render_out_path)

        compare = np.concatenate([render_panel, inpainted], axis=1)
        compare = draw_compare_title(compare, f"{stem} | original (left) vs lama_cond (right)")
        compare_out_path = compare_dir / f"{stem}_compare.png"
        Image.fromarray(compare).save(compare_out_path)

        render_frames.append(inpainted)
        compare_frames.append(compare)

        if i % 10 == 0 or i == len(entries):
            print(f"inpainted {i}/{len(entries)}")

    render_gif_path = output_root / f"output_render_lama_cond_{run_tag}.gif"
    compare_gif_path = output_root / f"output_render_lama_cond_compare_{run_tag}.gif"
    imageio.mimsave(render_gif_path, render_frames, duration=args.gif_duration)
    imageio.mimsave(compare_gif_path, compare_frames, duration=args.gif_duration)

    report_path = output_root / f"lama_cond_report_{run_tag}.txt"
    if processed_frames > 0:
        avg_mask_pixels = total_mask_pixels / processed_frames
        avg_person_before = total_person_pixels_before / processed_frames
        avg_person_after = total_person_pixels_after / processed_frames
        reduction_ratio = 1.0 - (avg_person_after / max(avg_person_before, 1.0))
    else:
        avg_mask_pixels = 0.0
        avg_person_before = 0.0
        avg_person_after = 0.0
        reduction_ratio = 0.0

    report_lines = [
        f"plots_dir={plots_dir}",
        f"run_tag={run_tag}",
        f"processed_frames={processed_frames}",
        f"start_video_idx={args.start_video_idx}",
        f"end_video_idx={args.end_video_idx}",
        f"person_prob_threshold={args.person_prob_threshold}",
        f"mask_dilation={args.mask_dilation}",
        f"temporal_mask_window={args.temporal_mask_window}",
        f"avg_mask_pixels={avg_mask_pixels:.2f}",
        f"avg_render_person_pixels_before={avg_person_before:.2f}",
        f"avg_render_person_pixels_after={avg_person_after:.2f}",
        f"avg_render_person_pixel_reduction_ratio={reduction_ratio:.4f}",
        f"render_gif_path={render_gif_path}",
        f"compare_gif_path={compare_gif_path}",
    ]
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(f"saved: {render_gif_path}")
    print(f"saved: {compare_gif_path}")
    print(f"saved: {report_path}")


if __name__ == "__main__":
    main()
