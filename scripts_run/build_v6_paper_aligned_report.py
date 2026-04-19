import argparse
import ast
import csv
import json
from pathlib import Path


SEQUENCE_ORDER = [
    "ANYmal1",
    "ANYmal2",
    "Ball",
    "Crowd",
    "Person",
    "Racket",
    "Stones",
    "Table1",
    "Table2",
    "Umbrella",
]

PAPER_ATE_CM = {
    "ANYmal1": 0.2,
    "ANYmal2": 0.3,
    "Ball": 0.2,
    "Crowd": 0.3,
    "Person": 0.8,
    "Racket": 0.4,
    "Stones": 0.3,
    "Table1": 0.6,
    "Table2": 1.3,
    "Umbrella": 0.2,
}

PAPER_NVS = {
    "ANYmal1": {"psnr": 21.85, "ssim": 0.807, "lpips": 0.211},
    "ANYmal2": {"psnr": 21.46, "ssim": 0.832, "lpips": 0.230},
    "Ball": {"psnr": 20.06, "ssim": 0.754, "lpips": 0.191},
    "Crowd": {"psnr": 21.28, "ssim": 0.802, "lpips": 0.176},
    "Person": {"psnr": 20.31, "ssim": 0.801, "lpips": 0.189},
    "Racket": {"psnr": 20.87, "ssim": 0.785, "lpips": 0.186},
    "Stones": {"psnr": 20.52, "ssim": 0.768, "lpips": 0.185},
    "Table1": {"psnr": 20.33, "ssim": 0.788, "lpips": 0.209},
    "Table2": {"psnr": 19.16, "ssim": 0.728, "lpips": 0.303},
    "Umbrella": {"psnr": 20.03, "ssim": 0.766, "lpips": 0.210},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a paper-aligned report package for WildGS-SLAM v6 results."
    )
    parser.add_argument(
        "--wandering-json",
        type=Path,
        required=True,
        help="Path to the wandering artifact metrics JSON.",
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
    mapping: dict[str, Path] = {}
    for entry in entries:
        if "=" not in entry:
            raise ValueError(f"Expected NAME=PATH entry, got: {entry}")
        name, path = entry.split("=", 1)
        seq = name.strip()
        if seq not in SEQUENCE_ORDER:
            raise ValueError(f"Unknown sequence name: {seq}")
        mapping[seq] = Path(path).resolve()
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


def format_float(value: float, digits: int = 3, signed: bool = False) -> str:
    if signed:
        return f"{value:+.{digits}f}"
    return f"{value:.{digits}f}"


def metric_delta(metric_name: str, v6: float, paper: float) -> float:
    return v6 - paper


def metric_is_better(metric_name: str, v6: float, paper: float) -> bool:
    if metric_name in {"ate_cm", "lpips"}:
        return v6 < paper
    return v6 > paper


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def build_claim_line(metric_label: str, delta: float, better_when_lower: bool) -> str:
    improved = delta < 0 if better_when_lower else delta > 0
    direction = "improves" if improved else "regresses"
    return f"- Average {metric_label}: v6 {direction} by {format_float(abs(delta), 3)}."


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    mocap_paths = parse_mapping(args.mocap)
    sequences_present = [seq for seq in SEQUENCE_ORDER if seq in mocap_paths]
    if not sequences_present:
        raise ValueError("At least one --mocap sequence mapping is required.")

    wandering_payload = load_json(args.wandering_json.resolve())
    wandering = wandering_payload["summary"]

    mocap_v6: dict[str, dict] = {}
    for sequence in sequences_present:
        exp_dir = mocap_paths[sequence]
        ate_cm = parse_rmse_cm(exp_dir / "traj" / "metrics_full_traj.txt")
        nvs = load_json(exp_dir / "nvs" / "final_result.json")
        mocap_v6[sequence] = {
            "exp_dir": str(exp_dir),
            "ate_cm": ate_cm,
            "nvs": {
                "frame_count": int(nvs["frame_count"]),
                "mean_psnr": float(nvs["mean_psnr"]),
                "mean_ssim": float(nvs["mean_ssim"]),
                "mean_lpips": float(nvs["mean_lpips"]),
            },
        }

    mocap_v6_rows: list[list[str]] = []
    tracking_rows: list[list[str]] = []
    nvs_rows: list[list[str]] = []
    combined_rows: list[list[str]] = []

    wins = {"ate_cm": 0, "psnr": 0, "ssim": 0, "lpips": 0}
    paper_ate_values: list[float] = []
    v6_ate_values: list[float] = []
    paper_psnr_values: list[float] = []
    v6_psnr_values: list[float] = []
    paper_ssim_values: list[float] = []
    v6_ssim_values: list[float] = []
    paper_lpips_values: list[float] = []
    v6_lpips_values: list[float] = []

    for sequence in sequences_present:
        v6 = mocap_v6[sequence]
        paper_ate = PAPER_ATE_CM[sequence]
        paper_nvs = PAPER_NVS[sequence]
        v6_ate = v6["ate_cm"]
        v6_psnr = v6["nvs"]["mean_psnr"]
        v6_ssim = v6["nvs"]["mean_ssim"]
        v6_lpips = v6["nvs"]["mean_lpips"]

        delta_ate = metric_delta("ate_cm", v6_ate, paper_ate)
        delta_psnr = metric_delta("psnr", v6_psnr, paper_nvs["psnr"])
        delta_ssim = metric_delta("ssim", v6_ssim, paper_nvs["ssim"])
        delta_lpips = metric_delta("lpips", v6_lpips, paper_nvs["lpips"])

        wins["ate_cm"] += int(metric_is_better("ate_cm", v6_ate, paper_ate))
        wins["psnr"] += int(metric_is_better("psnr", v6_psnr, paper_nvs["psnr"]))
        wins["ssim"] += int(metric_is_better("ssim", v6_ssim, paper_nvs["ssim"]))
        wins["lpips"] += int(metric_is_better("lpips", v6_lpips, paper_nvs["lpips"]))

        paper_ate_values.append(paper_ate)
        v6_ate_values.append(v6_ate)
        paper_psnr_values.append(paper_nvs["psnr"])
        v6_psnr_values.append(v6_psnr)
        paper_ssim_values.append(paper_nvs["ssim"])
        v6_ssim_values.append(v6_ssim)
        paper_lpips_values.append(paper_nvs["lpips"])
        v6_lpips_values.append(v6_lpips)

        mocap_v6_rows.append(
            [
                sequence,
                format_float(v6_ate, 3),
                format_float(v6_psnr, 3),
                format_float(v6_ssim, 4),
                format_float(v6_lpips, 4),
                str(v6["nvs"]["frame_count"]),
            ]
        )
        tracking_rows.append(
            [
                sequence,
                format_float(paper_ate, 3),
                format_float(v6_ate, 3),
                format_float(delta_ate, 3, signed=True),
            ]
        )
        nvs_rows.append(
            [
                sequence,
                format_float(paper_nvs["psnr"], 3),
                format_float(v6_psnr, 3),
                format_float(delta_psnr, 3, signed=True),
                format_float(paper_nvs["ssim"], 4),
                format_float(v6_ssim, 4),
                format_float(delta_ssim, 4, signed=True),
                format_float(paper_nvs["lpips"], 4),
                format_float(v6_lpips, 4),
                format_float(delta_lpips, 4, signed=True),
            ]
        )
        combined_rows.append(
            [
                sequence,
                format_float(paper_ate, 3),
                format_float(v6_ate, 3),
                format_float(delta_ate, 3, signed=True),
                format_float(paper_nvs["psnr"], 3),
                format_float(v6_psnr, 3),
                format_float(delta_psnr, 3, signed=True),
                format_float(paper_nvs["ssim"], 4),
                format_float(v6_ssim, 4),
                format_float(delta_ssim, 4, signed=True),
                format_float(paper_nvs["lpips"], 4),
                format_float(v6_lpips, 4),
                format_float(delta_lpips, 4, signed=True),
            ]
        )

    avg_paper_ate = mean(paper_ate_values)
    avg_v6_ate = mean(v6_ate_values)
    avg_paper_psnr = mean(paper_psnr_values)
    avg_v6_psnr = mean(v6_psnr_values)
    avg_paper_ssim = mean(paper_ssim_values)
    avg_v6_ssim = mean(v6_ssim_values)
    avg_paper_lpips = mean(paper_lpips_values)
    avg_v6_lpips = mean(v6_lpips_values)

    avg_rows = [
        [
            "ATE (cm)",
            format_float(avg_paper_ate, 3),
            format_float(avg_v6_ate, 3),
            format_float(avg_v6_ate - avg_paper_ate, 3, signed=True),
            f"{wins['ate_cm']}/{len(sequences_present)}",
        ],
        [
            "PSNR",
            format_float(avg_paper_psnr, 3),
            format_float(avg_v6_psnr, 3),
            format_float(avg_v6_psnr - avg_paper_psnr, 3, signed=True),
            f"{wins['psnr']}/{len(sequences_present)}",
        ],
        [
            "SSIM",
            format_float(avg_paper_ssim, 4),
            format_float(avg_v6_ssim, 4),
            format_float(avg_v6_ssim - avg_paper_ssim, 4, signed=True),
            f"{wins['ssim']}/{len(sequences_present)}",
        ],
        [
            "LPIPS",
            format_float(avg_paper_lpips, 4),
            format_float(avg_v6_lpips, 4),
            format_float(avg_v6_lpips - avg_paper_lpips, 4, signed=True),
            f"{wins['lpips']}/{len(sequences_present)}",
        ],
    ]

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
            str(int(wandering["frame_count"])),
            str(int(wandering["tail_frame_count"])),
        ]
    ]

    write_csv(
        output_dir / "mocap_v6.csv",
        ["Sequence", "ATE (cm)", "PSNR", "SSIM", "LPIPS", "NVS Frames"],
        mocap_v6_rows,
    )
    write_csv(
        output_dir / "paper_aligned_tracking.csv",
        ["Sequence", "Paper ATE (cm)", "v6 ATE (cm)", "Delta (v6 - paper)"],
        tracking_rows,
    )
    write_csv(
        output_dir / "paper_aligned_nvs.csv",
        [
            "Sequence",
            "Paper PSNR",
            "v6 PSNR",
            "Delta PSNR",
            "Paper SSIM",
            "v6 SSIM",
            "Delta SSIM",
            "Paper LPIPS",
            "v6 LPIPS",
            "Delta LPIPS",
        ],
        nvs_rows,
    )
    write_csv(
        output_dir / "paper_aligned_combined.csv",
        [
            "Sequence",
            "Paper ATE (cm)",
            "v6 ATE (cm)",
            "Delta ATE",
            "Paper PSNR",
            "v6 PSNR",
            "Delta PSNR",
            "Paper SSIM",
            "v6 SSIM",
            "Delta SSIM",
            "Paper LPIPS",
            "v6 LPIPS",
            "Delta LPIPS",
        ],
        combined_rows,
    )
    write_csv(
        output_dir / "paper_aligned_summary.csv",
        ["Metric", "Paper Avg", "v6 Avg", "Delta (v6 - paper)", "Wins"],
        avg_rows,
    )
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
            "Frame Count",
            "Tail Frame Count",
        ],
        wandering_rows,
    )

    full_coverage = len(sequences_present) == len(SEQUENCE_ORDER)
    scope_label = "full 10-sequence benchmark" if full_coverage else "available subset"
    avg_sequence_label = (
        "Full 10-sequence avg." if full_coverage else f"{len(sequences_present)}-sequence avg."
    )

    report_lines = [
        "## Experimental Results for v6",
        "",
        "### Experimental scope",
        (
            f"We evaluate `v6` on the {scope_label} of the Wild-SLAM MoCap benchmark "
            f"reported by the WildGS-SLAM paper ({len(sequences_present)}/10 sequences), "
            "and we keep the `iphone_wandering` artifact evaluation as a dedicated "
            "real-world dynamic-scene stress test."
        ),
        "",
        "This report compares only completed runs. No result files were edited or post-hoc adjusted.",
        "",
        "### Table 1. Paper-aligned Wild-SLAM MoCap tracking comparison",
        "",
        "| Sequence | Paper ATE (cm) ↓ | v6 ATE (cm) ↓ | Delta (v6 - paper) |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in tracking_rows:
        report_lines.append(f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} |")

    report_lines += [
        "",
        "### Table 2. Paper-aligned Wild-SLAM MoCap view synthesis comparison",
        "",
        "| Sequence | Paper PSNR ↑ | v6 PSNR ↑ | Delta | Paper SSIM ↑ | v6 SSIM ↑ | Delta | Paper LPIPS ↓ | v6 LPIPS ↓ | Delta |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in nvs_rows:
        report_lines.append(
            f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} | {row[4]} | {row[5]} | {row[6]} | {row[7]} | {row[8]} | {row[9]} |"
        )

    report_lines += [
        "",
        f"### Table 3. {avg_sequence_label} comparison against the WildGS-SLAM paper",
        "",
        "| Metric | Paper Avg | v6 Avg | Delta (v6 - paper) | Wins |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in avg_rows:
        report_lines.append(f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} | {row[4]} |")

    report_lines += [
        "",
        "### Table 4. v6 artifact-focused evaluation on `iphone_wandering`",
        "",
        "| Method | Mean BG PSNR ↑ | Mean BG MAE ↓ | Tail BG PSNR ↑ | Tail BG MAE ↓ | Mean All PSNR ↑ | Mean All MAE ↓ | Tail All PSNR ↑ | Tail All MAE ↓ |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| {wandering_rows[0][0]} | {wandering_rows[0][1]} | {wandering_rows[0][2]} | "
            f"{wandering_rows[0][3]} | {wandering_rows[0][4]} | {wandering_rows[0][5]} | "
            f"{wandering_rows[0][6]} | {wandering_rows[0][7]} | {wandering_rows[0][8]} |"
        ),
        "",
        "### What can be claimed safely",
    ]

    report_lines.extend(
        [
            build_claim_line("ATE", avg_v6_ate - avg_paper_ate, better_when_lower=True),
            build_claim_line("PSNR", avg_v6_psnr - avg_paper_psnr, better_when_lower=False),
            build_claim_line("SSIM", avg_v6_ssim - avg_paper_ssim, better_when_lower=False),
            build_claim_line("LPIPS", avg_v6_lpips - avg_paper_lpips, better_when_lower=True),
            (
                "- `iphone_wandering` provides quantitative evidence for background cleanup "
                "in a paper-acknowledged failure mode."
            ),
        ]
    )

    report_lines += [
        "",
        "### Remaining scope limit",
        "- This package aligns with the Wild-SLAM MoCap benchmark and the `iphone_wandering` failure case.",
        "- Bonn/TUM non-regression is still outside the current package because those datasets are not present locally.",
        "",
    ]

    (output_dir / "report.md").write_text("\n".join(report_lines), encoding="utf-8")
    (output_dir / "experiment_section.md").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )

    summary = {
        "coverage": {
            "sequence_count": len(sequences_present),
            "sequences": sequences_present,
            "full_coverage": full_coverage,
        },
        "wandering_v6": wandering,
        "paper_reference": {
            "ate_cm": {seq: PAPER_ATE_CM[seq] for seq in sequences_present},
            "nvs": {seq: PAPER_NVS[seq] for seq in sequences_present},
        },
        "mocap_v6": mocap_v6,
        "comparison": {
            "wins": wins,
            "averages": {
                "paper": {
                    "ate_cm": avg_paper_ate,
                    "psnr": avg_paper_psnr,
                    "ssim": avg_paper_ssim,
                    "lpips": avg_paper_lpips,
                },
                "v6": {
                    "ate_cm": avg_v6_ate,
                    "psnr": avg_v6_psnr,
                    "ssim": avg_v6_ssim,
                    "lpips": avg_v6_lpips,
                },
                "delta_v6_minus_paper": {
                    "ate_cm": avg_v6_ate - avg_paper_ate,
                    "psnr": avg_v6_psnr - avg_paper_psnr,
                    "ssim": avg_v6_ssim - avg_paper_ssim,
                    "lpips": avg_v6_lpips - avg_paper_lpips,
                },
            },
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
