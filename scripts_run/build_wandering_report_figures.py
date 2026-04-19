import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image, ImageDraw, ImageFont, ImageSequence


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "output" / "report_v6_full" / "figures"
METRICS_DIR = REPO_ROOT / "output" / "report_metrics"

FRAME_SELECTIONS = [
    ("Representative", 12),
    ("Turn", 66),
    ("Tail", 79),
]

GIFS = {
    "Temporal Fusion": REPO_ROOT
    / "output"
    / "Wild_SLAM_iPhone_temporal_proposal_v2_turn"
    / "iphone_wandering_temporal_proposal_v2_turn"
    / "plots_after_refine"
    / "output.gif",
    "Person-Prior Baseline": REPO_ROOT
    / "output"
    / "Wild_SLAM_iPhone_temporal_proposal_v3_turn"
    / "iphone_wandering_temporal_proposal_v3_turn"
    / "plots_after_refine"
    / "output.gif",
    "Insertion Gating": REPO_ROOT
    / "output"
    / "Wild_SLAM_iPhone_temporal_proposal_v4_turn"
    / "iphone_wandering_temporal_proposal_v4_turn"
    / "plots_after_refine"
    / "output.gif",
    "Opacity Decay": REPO_ROOT
    / "output"
    / "Wild_SLAM_iPhone_temporal_proposal_v5_turn"
    / "iphone_wandering_temporal_proposal_v5_turn"
    / "plots_after_refine"
    / "output.gif",
    "Full Model": REPO_ROOT
    / "output"
    / "Wild_SLAM_iPhone_temporal_proposal_v6_turn"
    / "iphone_wandering_temporal_proposal_v6_turn"
    / "plots_after_refine"
    / "output.gif",
}


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


FONT_HEADER = load_font(28, bold=True)
FONT_SUBHEADER = load_font(24, bold=True)
FONT_LABEL = load_font(22, bold=False)


def load_metrics_box() -> tuple[int, int, int, int]:
    payload = json.loads((METRICS_DIR / "wandering_v6.metrics.json").read_text())
    box = payload["crop_boxes"]["ground_truth_rgb"]
    return (box["x0"], box["y0"], box["x1"], box["y1"])


def load_frame(gif_path: Path, frame_idx: int) -> np.ndarray:
    image = Image.open(gif_path)
    for idx, frame in enumerate(ImageSequence.Iterator(image)):
        if idx == frame_idx:
            return np.array(frame.convert("RGB"))
    raise IndexError(f"Frame {frame_idx} not found in {gif_path}")


def crop_rgb(frame: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = box
    return frame[y0:y1, x0:x1, :3]


@torch.no_grad()
def build_person_mask(
    image: np.ndarray,
    model,
    preprocess,
    device: str,
    person_threshold: float = 0.35,
    dilation_radius: int = 1,
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
            person_prob.unsqueeze(0).unsqueeze(0),
            kernel_size=k,
            stride=1,
            padding=dilation_radius,
        ).squeeze(0).squeeze(0)
    return (person_prob > 0).detach().cpu().numpy()


def mask_to_box(mask: np.ndarray, width: int, height: int) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        cx, cy = width // 2, height // 2
        half_w, half_h = width // 6, height // 4
        return (
            max(0, cx - half_w),
            max(0, cy - half_h),
            min(width, cx + half_w),
            min(height, cy + half_h),
        )
    x0, x1 = xs.min(), xs.max() + 1
    y0, y1 = ys.min(), ys.max() + 1
    pad_x = max(20, int((x1 - x0) * 0.35))
    pad_y = max(20, int((y1 - y0) * 0.35))
    return (
        max(0, x0 - pad_x),
        max(0, y0 - pad_y),
        min(width, x1 + pad_x),
        min(height, y1 + pad_y),
    )


def draw_box(image: np.ndarray, box: tuple[int, int, int, int], color=(220, 30, 30), width: int = 5) -> Image.Image:
    pil_image = Image.fromarray(image.copy())
    draw = ImageDraw.Draw(pil_image)
    for offset in range(width):
        draw.rectangle(
            (
                box[0] - offset,
                box[1] - offset,
                box[2] + offset,
                box[3] + offset,
            ),
            outline=color,
        )
    return pil_image


def resize_image(image: Image.Image, target_width: int) -> Image.Image:
    ratio = target_width / image.width
    target_height = int(image.height * ratio)
    return image.resize((target_width, target_height), Image.Resampling.LANCZOS)


def paste_center(canvas: Image.Image, image: Image.Image, x: int, y: int, cell_w: int, cell_h: int) -> None:
    offset_x = x + (cell_w - image.width) // 2
    offset_y = y + (cell_h - image.height) // 2
    canvas.paste(image, (offset_x, offset_y))


def write_centered(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], text: str, font, fill=(0, 0, 0)) -> None:
    bbox = draw.multiline_textbbox((0, 0), text, font=font, align="center")
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = box[0] + (box[2] - box[0] - text_w) // 2
    y = box[1] + (box[3] - box[1] - text_h) // 2
    draw.multiline_text((x, y), text, font=font, fill=fill, align="center")


def build_main_comparison_figure(
    gt_frames: dict[int, np.ndarray],
    cropped_frames: dict[str, dict[int, np.ndarray]],
    zoom_boxes: dict[int, tuple[int, int, int, int]],
) -> Path:
    headers = [
        "Ground Truth",
        "Person-Prior\nBaseline",
        "Full Model",
        "Zoom:\nBaseline",
        "Zoom:\nFull Model",
    ]
    row_names = [name for name, _ in FRAME_SELECTIONS]
    target_w = 210
    sample = Image.fromarray(gt_frames[FRAME_SELECTIONS[0][1]])
    sample = resize_image(sample, target_w)
    cell_w = target_w + 10
    cell_h = sample.height + 10
    label_w = 190
    header_h = 90
    margin = 28

    width = label_w + len(headers) * cell_w + margin * 2
    height = header_h + len(row_names) * cell_h + margin * 2
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    for col, header in enumerate(headers):
        x0 = margin + label_w + col * cell_w
        write_centered(
            draw,
            (x0, margin, x0 + cell_w, margin + header_h),
            header,
            FONT_SUBHEADER,
        )

    for row, (row_name, frame_idx) in enumerate(FRAME_SELECTIONS):
        y0 = margin + header_h + row * cell_h
        write_centered(
            draw,
            (margin, y0, margin + label_w - 10, y0 + cell_h),
            f"{row_name}\n(frame {frame_idx})",
            FONT_SUBHEADER,
        )

        gt = draw_box(gt_frames[frame_idx], zoom_boxes[frame_idx])
        baseline = draw_box(cropped_frames["Person-Prior Baseline"][frame_idx], zoom_boxes[frame_idx])
        full = draw_box(cropped_frames["Full Model"][frame_idx], zoom_boxes[frame_idx])

        zb = zoom_boxes[frame_idx]
        zoom_baseline = Image.fromarray(cropped_frames["Person-Prior Baseline"][frame_idx][zb[1]:zb[3], zb[0]:zb[2]])
        zoom_full = Image.fromarray(cropped_frames["Full Model"][frame_idx][zb[1]:zb[3], zb[0]:zb[2]])

        images = [
            resize_image(gt, target_w),
            resize_image(baseline, target_w),
            resize_image(full, target_w),
            resize_image(zoom_baseline, target_w),
            resize_image(zoom_full, target_w),
        ]
        for col, image in enumerate(images):
            x0 = margin + label_w + col * cell_w
            paste_center(canvas, image, x0, y0, cell_w, cell_h)

    out_path = OUTPUT_DIR / "wandering_main_comparison.png"
    canvas.save(out_path)
    return out_path


def build_ablation_figure(
    cropped_frames: dict[str, dict[int, np.ndarray]],
    zoom_boxes: dict[int, tuple[int, int, int, int]],
) -> Path:
    headers = [
        "Temporal\nFusion",
        "+ Person\nPrior",
        "+ Insertion\nGating",
        "+ Opacity\nDecay",
        "Full\nModel",
    ]
    labels = [
        "Temporal Fusion",
        "Person-Prior Baseline",
        "Insertion Gating",
        "Opacity Decay",
        "Full Model",
    ]
    target_w = 200
    sample = Image.fromarray(cropped_frames["Temporal Fusion"][FRAME_SELECTIONS[0][1]])
    sample = resize_image(sample, target_w)
    cell_w = target_w + 10
    cell_h = sample.height + 10
    label_w = 190
    header_h = 90
    margin = 28

    width = label_w + len(headers) * cell_w + margin * 2
    height = header_h + len(FRAME_SELECTIONS) * cell_h + margin * 2
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    for col, header in enumerate(headers):
        x0 = margin + label_w + col * cell_w
        write_centered(
            draw,
            (x0, margin, x0 + cell_w, margin + header_h),
            header,
            FONT_SUBHEADER,
        )

    for row, (row_name, frame_idx) in enumerate(FRAME_SELECTIONS):
        y0 = margin + header_h + row * cell_h
        write_centered(
            draw,
            (margin, y0, margin + label_w - 10, y0 + cell_h),
            f"{row_name}\n(frame {frame_idx})",
            FONT_SUBHEADER,
        )

        for col, label in enumerate(labels):
            image = draw_box(cropped_frames[label][frame_idx], zoom_boxes[frame_idx])
            image = resize_image(image, target_w)
            x0 = margin + label_w + col * cell_w
            paste_center(canvas, image, x0, y0, cell_w, cell_h)

    out_path = OUTPUT_DIR / "wandering_ablation_progression.png"
    canvas.save(out_path)
    return out_path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    box = load_metrics_box()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    weights = torchvision.models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT
    preprocess = weights.transforms()
    model = torchvision.models.segmentation.deeplabv3_resnet50(weights=weights).to(device)
    model.eval()

    cropped_frames: dict[str, dict[int, np.ndarray]] = {key: {} for key in GIFS}
    gt_frames: dict[int, np.ndarray] = {}
    zoom_boxes: dict[int, tuple[int, int, int, int]] = {}

    for _, frame_idx in FRAME_SELECTIONS:
        gt_frame = load_frame(GIFS["Full Model"], frame_idx)
        gt = crop_rgb(gt_frame, box)
        gt_frames[frame_idx] = gt

        mask = build_person_mask(gt, model, preprocess, device)
        zoom_boxes[frame_idx] = mask_to_box(mask, gt.shape[1], gt.shape[0])

        for label, gif_path in GIFS.items():
            frame = load_frame(gif_path, frame_idx)
            rendered = crop_rgb(frame, box)
            cropped_frames[label][frame_idx] = rendered

    main_path = build_main_comparison_figure(gt_frames, cropped_frames, zoom_boxes)
    ablation_path = build_ablation_figure(cropped_frames, zoom_boxes)

    manifest = {
        "main_comparison_figure": str(main_path),
        "ablation_figure": str(ablation_path),
        "selected_frames": [{"label": name, "frame_idx": idx} for name, idx in FRAME_SELECTIONS],
        "ablation_labels": [
            "Temporal Fusion",
            "Person-Prior Baseline",
            "Insertion Gating",
            "Opacity Decay",
            "Full Model",
        ],
    }
    (OUTPUT_DIR / "wandering_figures_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
