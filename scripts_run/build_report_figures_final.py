import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "output" / "report_v6_full" / "figures"

WANDERING_BASELINE = (
    REPO_ROOT
    / "output"
    / "Wild_SLAM_iPhone"
    / "iphone_wandering"
    / "rendered_rgb_after_refine"
    / "video_idx_43_kf_idx_387.png"
)
WANDERING_TEG = (
    REPO_ROOT
    / "output"
    / "Wild_SLAM_iPhone_temporal_proposal_v6_turn"
    / "iphone_wandering_temporal_proposal_v6_turn"
    / "rendered_rgb_after_refine"
    / "video_idx_43_kf_idx_387.png"
)

MOCAP_EXAMPLES = [
    (
        "Crowd",
        238,
        REPO_ROOT
        / "output"
        / "Wild_SLAM_Mocap_report_v6"
        / "Crowd_v6_report"
        / "plots_after_refine"
        / "video_idx_30_kf_idx_238.png",
    ),
    (
        "Ball",
        386,
        REPO_ROOT
        / "output"
        / "Wild_SLAM_Mocap_report_v6"
        / "Ball_v6_report"
        / "plots_after_refine"
        / "video_idx_43_kf_idx_386.png",
    ),
    (
        "Racket",
        374,
        REPO_ROOT
        / "output"
        / "Wild_SLAM_Mocap_report_v6"
        / "Racket_v6_report"
        / "plots_after_refine"
        / "video_idx_43_kf_idx_374.png",
    ),
    (
        "Umbrella",
        203,
        REPO_ROOT
        / "output"
        / "Wild_SLAM_Mocap_report_v6"
        / "Umbrella_v6_report"
        / "plots_after_refine"
        / "video_idx_25_kf_idx_203.png",
    ),
]

# All selected MoCap plots share the same layout.
MOCAP_GT_CROP = (10, 95, 476, 356)
MOCAP_RENDER_CROP = (10, 428, 476, 689)


def load_font(size: int, bold: bool = False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


FONT_TITLE = load_font(30, bold=True)
FONT_HEADER = load_font(24, bold=True)
FONT_LABEL = load_font(22, bold=False)


def resize_to_width(image: Image.Image, target_width: int) -> Image.Image:
    ratio = target_width / image.width
    target_height = int(image.height * ratio)
    return image.resize((target_width, target_height), Image.Resampling.LANCZOS)


def paste_center(canvas: Image.Image, image: Image.Image, x: int, y: int, cell_w: int, cell_h: int) -> None:
    canvas.paste(
        image,
        (x + (cell_w - image.width) // 2, y + (cell_h - image.height) // 2),
    )


def write_centered(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    font,
    fill=(0, 0, 0),
) -> None:
    bbox = draw.multiline_textbbox((0, 0), text, font=font, align="center")
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = box[0] + (box[2] - box[0] - text_w) // 2
    y = box[1] + (box[3] - box[1] - text_h) // 2
    draw.multiline_text((x, y), text, font=font, fill=fill, align="center")


def build_wandering_figure() -> Path:
    baseline = Image.open(WANDERING_BASELINE).convert("RGB")
    teg = Image.open(WANDERING_TEG).convert("RGB")

    target_w = 500
    baseline = resize_to_width(baseline, target_w)
    teg = resize_to_width(teg, target_w)

    margin = 28
    header_h = 70
    title_h = 60
    cell_w = target_w + 12
    cell_h = baseline.height + 12

    width = margin * 2 + cell_w * 2
    height = margin * 2 + title_h + header_h + cell_h
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    write_centered(
        draw,
        (margin, margin, width - margin, margin + title_h),
        "iphone_wandering Rendered RGB Comparison",
        FONT_TITLE,
    )

    headers = ["WildGS-SLAM Baseline", "TEG-SLAM (Ours)"]
    for col, header in enumerate(headers):
        x0 = margin + col * cell_w
        write_centered(
            draw,
            (x0, margin + title_h, x0 + cell_w, margin + title_h + header_h),
            header,
            FONT_HEADER,
        )

    y0 = margin + title_h + header_h
    paste_center(canvas, baseline, margin, y0, cell_w, cell_h)
    paste_center(canvas, teg, margin + cell_w, y0, cell_w, cell_h)

    out_path = OUTPUT_DIR / "wandering_rendered_rgb_comparison.png"
    canvas.save(out_path)
    return out_path


def crop_box(image: Image.Image, box: tuple[int, int, int, int]) -> Image.Image:
    return image.crop(box)


def build_mocap_examples_figure() -> Path:
    gt_images = []
    render_images = []
    for _, _, plot_path in MOCAP_EXAMPLES:
        plot = Image.open(plot_path).convert("RGB")
        gt_images.append(crop_box(plot, MOCAP_GT_CROP))
        render_images.append(crop_box(plot, MOCAP_RENDER_CROP))

    target_w = 250
    gt_images = [resize_to_width(img, target_w) for img in gt_images]
    render_images = [resize_to_width(img, target_w) for img in render_images]

    margin = 28
    title_h = 60
    header_h = 72
    row_label_w = 130
    cell_w = target_w + 10
    cell_h = gt_images[0].height + 10

    width = margin * 2 + row_label_w + cell_w * len(MOCAP_EXAMPLES)
    height = margin * 2 + title_h + header_h + cell_h * 2
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    write_centered(
        draw,
        (margin, margin, width - margin, margin + title_h),
        "Successful Dynamic-Object Removal on Non-wandering Sequences",
        FONT_TITLE,
    )

    header_y0 = margin + title_h
    for col, (name, frame_idx, _) in enumerate(MOCAP_EXAMPLES):
        x0 = margin + row_label_w + col * cell_w
        write_centered(
            draw,
            (x0, header_y0, x0 + cell_w, header_y0 + header_h),
            f"{name}\n(frame {frame_idx})",
            FONT_HEADER,
        )

    row1_y = margin + title_h + header_h
    row2_y = row1_y + cell_h
    write_centered(
        draw,
        (margin, row1_y, margin + row_label_w - 8, row1_y + cell_h),
        "Input RGB",
        FONT_HEADER,
    )
    write_centered(
        draw,
        (margin, row2_y, margin + row_label_w - 8, row2_y + cell_h),
        "TEG-SLAM\noutput",
        FONT_HEADER,
    )

    for col in range(len(MOCAP_EXAMPLES)):
        x0 = margin + row_label_w + col * cell_w
        paste_center(canvas, gt_images[col], x0, row1_y, cell_w, cell_h)
        paste_center(canvas, render_images[col], x0, row2_y, cell_w, cell_h)

    out_path = OUTPUT_DIR / "mocap_dynamic_removal_examples.png"
    canvas.save(out_path)
    return out_path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    wandering = build_wandering_figure()
    mocap = build_mocap_examples_figure()

    manifest = {
        "wandering_rendered_rgb_comparison": str(wandering),
        "mocap_dynamic_removal_examples": str(mocap),
        "wandering_frame": {
            "sequence": "iphone_wandering",
            "frame_idx": 387,
            "baseline": str(WANDERING_BASELINE),
            "teg_slam": str(WANDERING_TEG),
        },
        "mocap_examples": [
            {"sequence": name, "frame_idx": frame_idx, "plot_path": str(plot_path)}
            for name, frame_idx, plot_path in MOCAP_EXAMPLES
        ],
    }
    (OUTPUT_DIR / "report_figures_manifest_final.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
