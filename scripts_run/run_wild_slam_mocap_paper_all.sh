#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

usage() {
  cat <<'EOF'
Usage: bash scripts_run/run_wild_slam_mocap_paper_all.sh [options]

Options:
  --output-root PATH    Output root for isolated paper-fidelity runs.
                        Default: ./output/Wild_SLAM_Mocap_paper
  --skip-existing       Skip sequences that already have metrics_full_traj.txt.
  --dry-run             Print the commands that would run, then exit.
  --keep-config         Keep generated temporary config files.
  --help                Show this help message.
EOF
}

run_python() {
  local env_name="${ENV_NAME:-wildgs-slam}"
  local micromamba_bin="${MICROMAMBA_BIN:-$HOME/.local/bin/micromamba}"

  if [ "${CONDA_DEFAULT_ENV:-}" = "$env_name" ]; then
    python "$@"
    return
  fi

  if [ ! -x "$micromamba_bin" ]; then
    echo "Micromamba not found at $micromamba_bin." >&2
    echo "Activate the $env_name environment first or set MICROMAMBA_BIN." >&2
    exit 1
  fi

  "$micromamba_bin" run -n "$env_name" python "$@"
}

output_root="${OUTPUT_ROOT:-./output/Wild_SLAM_Mocap_paper}"
skip_existing=0
dry_run=0
keep_config=0

while [ "$#" -gt 0 ]; do
  case "$1" in
    --output-root)
      if [ "$#" -lt 2 ]; then
        echo "--output-root requires a path." >&2
        exit 1
      fi
      output_root="$2"
      shift 2
      ;;
    --skip-existing)
      skip_existing=1
      shift
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    --keep-config)
      keep_config=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

output_root="${output_root%/}"
sequence_script="./scripts_run/run_wild_slam_mocap_paper_sequence.sh"
summary_csv="${output_root}_eval.csv"
scene_order="ANYmal1,ANYmal2,Ball,Crowd,Person,Racket,Stones,Table1,Table2,Umbrella"
sequences=(
  anymal1
  anymal2
  ball
  crowd
  person
  racket
  stones
  table1
  table2
  umbrella
)

for sequence in "${sequences[@]}"; do
  cmd=(bash "$sequence_script" "$sequence" --output-root "$output_root")
  if [ "$skip_existing" -eq 1 ]; then
    cmd+=(--skip-existing)
  fi
  if [ "$dry_run" -eq 1 ]; then
    cmd+=(--dry-run)
  fi
  if [ "$keep_config" -eq 1 ]; then
    cmd+=(--keep-config)
  fi

  "${cmd[@]}"
done

if [ "$dry_run" -eq 1 ]; then
  echo "Dry run only. No sequences were launched."
  echo "Summary CSV would be written to: $summary_csv"
  exit 0
fi

run_python scripts_run/summarize_pose_eval.py \
  --dataset-output "$output_root" \
  --output-csv "$summary_csv" \
  --scene-order "$scene_order"

echo "Finished Wild-SLAM MoCap paper-fidelity batch run."
echo "Summary CSV: $summary_csv"
