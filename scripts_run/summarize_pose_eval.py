import argparse
import ast
import csv
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize WildGS-SLAM full-trajectory ATE RMSE results as CSV."
    )
    parser.add_argument(
        "--dataset-output",
        type=Path,
        help="Summarize a single output dataset directory, e.g. ./output/Wild_SLAM_Mocap_paper.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        help="CSV path for --dataset-output mode. Defaults to <dataset-output>_eval.csv.",
    )
    parser.add_argument(
        "--scene-order",
        type=str,
        default="",
        help="Optional comma-separated scene order for the CSV columns.",
    )
    parser.add_argument(
        "--row-name",
        type=str,
        default="wildgs-slam",
        help="Row label to use in the generated CSV.",
    )
    return parser.parse_args()


def extract_rmse(result_file: Path):
    if not result_file.exists():
        return None

    for line in result_file.read_text().splitlines():
        line = line.strip()
        if line.startswith("{") and "'rmse':" in line:
            stats = ast.literal_eval(line)
            return float(stats["rmse"])

    raise ValueError(f"Could not parse rmse from {result_file}")


def ordered_scenes(dataset_path: Path, scene_order):
    scene_dirs = sorted(
        path.name
        for path in dataset_path.iterdir()
        if path.is_dir() and ((path / "cfg.yaml").exists() or (path / "traj").is_dir())
    )
    if not scene_order:
        return scene_dirs

    ordered = []
    seen = set()
    for scene in scene_order:
        if scene not in seen:
            ordered.append(scene)
            seen.add(scene)

    for scene in scene_dirs:
        if scene not in seen:
            ordered.append(scene)
            seen.add(scene)

    return ordered


def summarize_dataset(dataset_path: Path, csv_path: Path, scene_order, row_name: str):
    scenes = ordered_scenes(dataset_path, scene_order)
    rmses = []
    row_data = []

    for scene in scenes:
        result_file = dataset_path / scene / "traj" / "metrics_full_traj.txt"
        rmse = extract_rmse(result_file)
        if rmse is None:
            row_data.append("N/A")
            continue

        row_data.append(f"{rmse * 1e2:.2f}")
        rmses.append(rmse)

    if rmses:
        avg_value = f"{(sum(rmses) / len(rmses)) * 1e2:.2f}"
    else:
        avg_value = "N/A"

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["", *scenes, "Average"])
        writer.writerow([row_name, *row_data, avg_value])

    print(f"Results saved to {csv_path}")


def main():
    args = parse_args()
    scene_order = [scene.strip() for scene in args.scene_order.split(",") if scene.strip()]

    if args.dataset_output is not None:
        dataset_path = args.dataset_output
        csv_path = args.output_csv or dataset_path.with_name(f"{dataset_path.name}_eval.csv")
        summarize_dataset(dataset_path, csv_path, scene_order, args.row_name)
        return

    output_root = Path("./output")
    for dataset_path in sorted(path for path in output_root.iterdir() if path.is_dir()):
        csv_path = output_root / f"{dataset_path.name}_eval.csv"
        summarize_dataset(dataset_path, csv_path, scene_order, args.row_name)


if __name__ == "__main__":
    main()
        