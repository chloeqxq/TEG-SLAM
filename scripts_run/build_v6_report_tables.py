import argparse
import ast
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build CSV/Markdown tables for the WildGS-SLAM v6 project report."
    )
    parser.add_argument(
        "--wandering",
        action="append",
        default=[],
        help="Method-to-JSON mapping in the form method=/path/to/wandering_metrics.json",
    )
    parser.add_argument(
        "--mocap-baseline",
        action="append",
        default=[],
        help="Sequence-to-exp-dir mapping in the form Crowd=/path/to/baseline_exp_dir",
    )
    parser.add_argument(
        "--mocap-v6",
        action="append",
        default=[],
        help="Sequence-to-exp-dir mapping in the form Crowd=/path/to/v6_exp_dir",
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


def load_nvs_metrics(exp_dir: Path) -> dict:
    result_path = exp_dir / "nvs" / "final_result.json"
    if not result_path.exists():
        raise FileNotFoundError(result_path)
    return json.loads(result_path.read_text(encoding="utf-8"))


def load_wandering_metrics(json_path: Path) -> dict:
    if not json_path.exists():
        raise FileNotFoundError(json_path)
    return json.loads(json_path.read_text(encoding="utf-8"))


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

    wandering_paths = parse_mapping(args.wandering)
    mocap_baseline = parse_mapping(args.mocap_baseline)
    mocap_v6 = parse_mapping(args.mocap_v6)

    summary = {
        "wandering": {},
        "mocap_ate": {},
        "mocap_nvs": {},
    }

    wandering_rows = []
    for method, json_path in wandering_paths.items():
        payload = load_wandering_metrics(json_path)
        metrics = payload["summary"]
        summary["wandering"][method] = metrics
        wandering_rows.append(
            [
                method,
                format_float(metrics["mean_background_psnr"], 3),
                format_float(metrics["mean_background_mae"], 3),
                format_float(metrics["tail_mean_background_psnr"], 3),
                format_float(metrics["tail_mean_background_mae"], 3),
                format_float(metrics["mean_all_psnr"], 3),
                format_float(metrics["mean_all_mae"], 3),
                format_float(metrics["tail_mean_all_psnr"], 3),
                format_float(metrics["tail_mean_all_mae"], 3),
            ]
        )
    wandering_rows.sort(key=lambda row: row[0])
    write_csv(
        output_dir / "wandering_metrics.csv",
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

    shared_sequences = sorted(set(mocap_baseline) & set(mocap_v6))
    ate_rows = []
    nvs_rows = []
    for sequence in shared_sequences:
        baseline_dir = mocap_baseline[sequence]
        v6_dir = mocap_v6[sequence]

        baseline_ate = parse_rmse_cm(baseline_dir / "traj" / "metrics_full_traj.txt")
        v6_ate = parse_rmse_cm(v6_dir / "traj" / "metrics_full_traj.txt")
        ate_delta = v6_ate - baseline_ate
        ate_winner = "v6" if v6_ate < baseline_ate else "original"

        baseline_nvs = load_nvs_metrics(baseline_dir)
        v6_nvs = load_nvs_metrics(v6_dir)
        psnr_delta = float(v6_nvs["mean_psnr"] - baseline_nvs["mean_psnr"])
        ssim_delta = float(v6_nvs["mean_ssim"] - baseline_nvs["mean_ssim"])
        lpips_delta = float(v6_nvs["mean_lpips"] - baseline_nvs["mean_lpips"])
        nvs_winner = (
            "v6"
            if (psnr_delta > 0 and ssim_delta > 0 and lpips_delta < 0)
            else "mixed"
        )

        summary["mocap_ate"][sequence] = {
            "original_cm": baseline_ate,
            "v6_cm": v6_ate,
            "delta_cm": ate_delta,
        }
        summary["mocap_nvs"][sequence] = {
            "original": {
                "psnr": baseline_nvs["mean_psnr"],
                "ssim": baseline_nvs["mean_ssim"],
                "lpips": baseline_nvs["mean_lpips"],
            },
            "v6": {
                "psnr": v6_nvs["mean_psnr"],
                "ssim": v6_nvs["mean_ssim"],
                "lpips": v6_nvs["mean_lpips"],
            },
            "delta": {
                "psnr": psnr_delta,
                "ssim": ssim_delta,
                "lpips": lpips_delta,
            },
        }

        ate_rows.append(
            [
                sequence,
                format_float(baseline_ate, 2),
                format_float(v6_ate, 2),
                format_float(ate_delta, 2),
                ate_winner,
            ]
        )
        nvs_rows.append(
            [
                sequence,
                format_float(baseline_nvs["mean_psnr"], 3),
                format_float(v6_nvs["mean_psnr"], 3),
                format_float(psnr_delta, 3),
                format_float(baseline_nvs["mean_ssim"], 4),
                format_float(v6_nvs["mean_ssim"], 4),
                format_float(ssim_delta, 4),
                format_float(baseline_nvs["mean_lpips"], 4),
                format_float(v6_nvs["mean_lpips"], 4),
                format_float(lpips_delta, 4),
                nvs_winner,
            ]
        )

    write_csv(
        output_dir / "mocap_ate.csv",
        ["Sequence", "Original ATE (cm)", "v6 ATE (cm)", "Delta (cm)", "Winner"],
        ate_rows,
    )
    write_csv(
        output_dir / "mocap_nvs.csv",
        [
            "Sequence",
            "Original PSNR",
            "v6 PSNR",
            "Delta PSNR",
            "Original SSIM",
            "v6 SSIM",
            "Delta SSIM",
            "Original LPIPS",
            "v6 LPIPS",
            "Delta LPIPS",
            "Winner",
        ],
        nvs_rows,
    )

    markdown_lines = [
        "## MoCap ATE",
        "",
        "| Sequence | Original ATE (cm) | v6 ATE (cm) | Delta (cm) | Winner |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for row in ate_rows:
        markdown_lines.append(
            f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} | {row[4]} |"
        )

    markdown_lines += [
        "",
        "## MoCap NVS",
        "",
        "| Sequence | Original PSNR | v6 PSNR | Delta PSNR | Original SSIM | v6 SSIM | Delta SSIM | Original LPIPS | v6 LPIPS | Delta LPIPS | Winner |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in nvs_rows:
        markdown_lines.append(
            f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} | {row[4]} | {row[5]} | {row[6]} | {row[7]} | {row[8]} | {row[9]} | {row[10]} |"
        )

    markdown_lines += [
        "",
        "## Wandering",
        "",
        "| Method | Mean BG PSNR | Mean BG MAE | Tail BG PSNR | Tail BG MAE | Mean All PSNR | Mean All MAE | Tail All PSNR | Tail All MAE |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in wandering_rows:
        markdown_lines.append(
            f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} | {row[4]} | {row[5]} | {row[6]} | {row[7]} | {row[8]} |"
        )

    (output_dir / "report_tables.md").write_text(
        "\n".join(markdown_lines) + "\n", encoding="utf-8"
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
