import argparse
import ast
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build v6-only CSV/Markdown summaries for WildGS-SLAM report outputs."
    )
    parser.add_argument(
        "--wandering-json",
        type=Path,
        required=True,
        help="Path to the v6 wandering metrics JSON.",
    )
    parser.add_argument(
        "--mocap",
        action="append",
        default=[],
        help="Sequence-to-exp-dir mapping in the form Crowd=/path/to/exp_dir",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for generated CSV, Markdown, and JSON summaries.",
    )
    return parser.parse_args()


def parse_mapping(entries: list[str]) -> dict[str, Path]:
    mapping = {}
    for entry in entries:
        if "=" not in entry:
            raise ValueError(f"Expected NAME=PATH entry, got: {entry}")
        name, path = entry.split("=", 1)
        mapping[name.strip()] = Path(path).resolve()
    return mapping


def parse_rmse_cm(metrics_path: Path) -> float:
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("{") and "'rmse':" in line:
            stats = ast.literal_eval(line)
            return float(stats["rmse"]) * 1e2
    raise RuntimeError(f"Failed to parse RMSE from {metrics_path}")


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def format_float(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    mocap_paths = parse_mapping(args.mocap)
    wandering_payload = load_json(args.wandering_json.resolve())
    wandering = wandering_payload["summary"]

    wandering_rows = [
        [
            "v6",
            format_float(wandering["mean_background_psnr"], 3),
            format_float(wandering["mean_background_mae"], 3),
            format_float(wandering["tail_mean_background_psnr"], 3),
            format_float(wandering["tail_mean_background_mae"], 3),
            format_float(wandering["mean_all_psnr"], 3),
            format_float(wandering["mean_all_mae"], 3),
            format_float(wandering["tail_mean_all_psnr"], 3),
            format_float(wandering["tail_mean_all_mae"], 3),
        ]
    ]
    write_csv(
        output_dir / "wandering_v6.csv",
        [
            "Method",
            "Mean BG PSNR",
            "Mean BG MAE",
            "Tail BG PSNR",
            "Tail BG MAE",
            "Mean All PSNR",
            "Mean All MAE",
            "Tail All PSNR",
            "Tail All MAE",
        ],
        wandering_rows,
    )

    mocap_rows = []
    summary = {
        "wandering_v6": wandering,
        "mocap_v6": {},
    }
    for sequence, exp_dir in sorted(mocap_paths.items()):
        ate_cm = parse_rmse_cm(exp_dir / "traj" / "metrics_full_traj.txt")
        nvs = load_json(exp_dir / "nvs" / "final_result.json")
        mocap_rows.append(
            [
                sequence,
                format_float(ate_cm, 3),
                format_float(float(nvs["mean_psnr"]), 3),
                format_float(float(nvs["mean_ssim"]), 4),
                format_float(float(nvs["mean_lpips"]), 4),
                str(int(nvs["frame_count"])),
            ]
        )
        summary["mocap_v6"][sequence] = {
            "ate_cm": ate_cm,
            "nvs": {
                "frame_count": int(nvs["frame_count"]),
                "mean_psnr": float(nvs["mean_psnr"]),
                "mean_ssim": float(nvs["mean_ssim"]),
                "mean_lpips": float(nvs["mean_lpips"]),
            },
        }

    write_csv(
        output_dir / "mocap_v6.csv",
        [
            "Sequence",
            "ATE (cm)",
            "PSNR",
            "SSIM",
            "LPIPS",
            "NVS Frames",
        ],
        mocap_rows,
    )

    markdown_lines = [
        "## MoCap v6",
        "",
        "| Sequence | ATE (cm) | PSNR | SSIM | LPIPS | NVS Frames |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in mocap_rows:
        markdown_lines.append(
            f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} | {row[4]} | {row[5]} |"
        )

    markdown_lines += [
        "",
        "## Wandering v6",
        "",
        "| Method | Mean BG PSNR | Mean BG MAE | Tail BG PSNR | Tail BG MAE | Mean All PSNR | Mean All MAE | Tail All PSNR | Tail All MAE |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| {wandering_rows[0][0]} | {wandering_rows[0][1]} | {wandering_rows[0][2]} | {wandering_rows[0][3]} | {wandering_rows[0][4]} | {wandering_rows[0][5]} | {wandering_rows[0][6]} | {wandering_rows[0][7]} | {wandering_rows[0][8]} |",
        "",
    ]
    (output_dir / "report.md").write_text(
        "\n".join(markdown_lines), encoding="utf-8"
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
