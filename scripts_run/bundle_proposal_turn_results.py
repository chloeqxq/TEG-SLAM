from pathlib import Path
import shutil

from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[1]
BUNDLE_ROOT = REPO_ROOT / "output" / "wandering_proposal_bundle"
VERSIONS = {
    "v1": REPO_ROOT
    / "output"
    / "Wild_SLAM_iPhone_temporal_proposal_v1_turn"
    / "iphone_wandering_temporal_proposal_v1_turn",
    "v2": REPO_ROOT
    / "output"
    / "Wild_SLAM_iPhone_temporal_proposal_v2_turn"
    / "iphone_wandering_temporal_proposal_v2_turn",
    "v3": REPO_ROOT
    / "output"
    / "Wild_SLAM_iPhone_temporal_proposal_v3_turn"
    / "iphone_wandering_temporal_proposal_v3_turn",
}
SELECTED_FRAMES = [
    "video_idx_13_kf_idx_117.png",
    "video_idx_38_kf_idx_342.png",
    "video_idx_50_kf_idx_450.png",
    "video_idx_55_kf_idx_495.png",
    "video_idx_65_kf_idx_575.png",
    "video_idx_80_kf_idx_693.png",
]
CORE_FILES = [
    "cfg.yaml",
    "rendered_rgb_after_refine.mp4",
]


def ensure_exists(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)


def copy_assets():
    BUNDLE_ROOT.mkdir(parents=True, exist_ok=True)
    for version, source_dir in VERSIONS.items():
        ensure_exists(source_dir)
        target_dir = BUNDLE_ROOT / version
        target_dir.mkdir(parents=True, exist_ok=True)

        for name in CORE_FILES:
            source = source_dir / name
            ensure_exists(source)
            shutil.copy2(source, target_dir / source.name)

        gif_path = source_dir / "plots_after_refine" / "output.gif"
        ensure_exists(gif_path)
        shutil.copy2(gif_path, target_dir / gif_path.name)

        frames_dir = target_dir / "selected_frames"
        frames_dir.mkdir(exist_ok=True)
        for frame_name in SELECTED_FRAMES:
            source = source_dir / "plots_after_refine" / frame_name
            ensure_exists(source)
            shutil.copy2(source, frames_dir / frame_name)


def render_compare_grids():
    compare_dir = BUNDLE_ROOT / "compare_frames"
    compare_dir.mkdir(exist_ok=True)
    label_height = 40
    padding = 24
    font = ImageDraw.ImageDraw(Image.new("RGB", (1, 1))).getfont()

    for frame_name in SELECTED_FRAMES:
        opened = []
        for version, source_dir in VERSIONS.items():
            image = Image.open(source_dir / "plots_after_refine" / frame_name).convert("RGB")
            opened.append((version, image))

        widths = [image.width for _, image in opened]
        heights = [image.height for _, image in opened]
        canvas = Image.new(
            "RGB",
            (
                sum(widths) + padding * (len(opened) + 1),
                max(heights) + label_height + padding * 2,
            ),
            color=(255, 255, 255),
        )
        draw = ImageDraw.Draw(canvas)

        x_offset = padding
        for version, image in opened:
            draw.text((x_offset, 8), version.upper(), fill=(0, 0, 0), font=font)
            canvas.paste(image, (x_offset, label_height))
            x_offset += image.width + padding

        output_name = frame_name.replace("_kf_idx_", "_compare_kf_idx_")
        canvas.save(compare_dir / output_name)


def write_readme():
    readme = BUNDLE_ROOT / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# Proposal Turn Experiments",
                "",
                "Bundled outputs for `proposal_v1`, `proposal_v2`, and `proposal_v3`.",
                "",
                "Included per version:",
                "- `cfg.yaml`",
                "- `rendered_rgb_after_refine.mp4`",
                "- `output.gif`",
                "- `selected_frames/` with the key turn-region frames",
                "",
                "Included shared compare views:",
                "- `compare_frames/` with horizontal `v1 | v2 | v3` grids",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    copy_assets()
    render_compare_grids()
    write_readme()
    print(f"Bundled proposal outputs into: {BUNDLE_ROOT}")
