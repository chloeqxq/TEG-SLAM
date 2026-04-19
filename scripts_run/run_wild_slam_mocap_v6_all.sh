#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

usage() {
  cat <<'EOF'
Usage: bash scripts_run/run_wild_slam_mocap_v6_all.sh [options]

Options:
  --output-root PATH      Output root for v6 MoCap runs.
                          Default: ./output/Wild_SLAM_Mocap_report_v6
  --profile NAME          Temporal config profile to use: report, conservative.
                          Default: report
  --wandering-json PATH   Wandering metrics JSON used in the final report.
                          Default: ./output/report_metrics/wandering_v6.metrics.json
  --report-dir PATH       Directory for the aggregated report package.
                          Default: ./output/report_v6_full
  --max-frames N          Forwarded smoke/tuning frame limit.
  --skip-existing         Reuse existing SLAM/NVS outputs when present.
  --skip-report           Skip final report generation.
  --help                  Show this help message.
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

output_root="${OUTPUT_ROOT:-./output/Wild_SLAM_Mocap_report_v6}"
profile="${V6_PROFILE:-report}"
wandering_json="./output/report_metrics/wandering_v6.metrics.json"
report_dir="./output/report_v6_full"
max_frames_override=""
skip_existing=0
skip_report=0

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
    --profile)
      if [ "$#" -lt 2 ]; then
        echo "--profile requires a value." >&2
        exit 1
      fi
      profile="$2"
      shift 2
      ;;
    --wandering-json)
      if [ "$#" -lt 2 ]; then
        echo "--wandering-json requires a path." >&2
        exit 1
      fi
      wandering_json="$2"
      shift 2
      ;;
    --report-dir)
      if [ "$#" -lt 2 ]; then
        echo "--report-dir requires a path." >&2
        exit 1
      fi
      report_dir="$2"
      shift 2
      ;;
    --max-frames)
      if [ "$#" -lt 2 ]; then
        echo "--max-frames requires an integer value." >&2
        exit 1
      fi
      max_frames_override="$2"
      shift 2
      ;;
    --skip-existing)
      skip_existing=1
      shift
      ;;
    --skip-report)
      skip_report=1
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
  cmd=(
    bash
    ./scripts_run/run_wild_slam_mocap_v6_sequence.sh
    "$sequence"
    --output-root
    "$output_root"
    --profile
    "$profile"
  )
  if [ -n "$max_frames_override" ]; then
    cmd+=(--max-frames "$max_frames_override")
  fi
  if [ "$skip_existing" -eq 1 ]; then
    cmd+=(--skip-existing)
  fi
  "${cmd[@]}"
done

if [ "$skip_report" -eq 1 ]; then
  echo "Skipped final report generation."
  exit 0
fi

if [ ! -f "$wandering_json" ]; then
  echo "Missing wandering metrics JSON: $wandering_json" >&2
  exit 1
fi

scene_order=(ANYmal1 ANYmal2 Ball Crowd Person Racket Stones Table1 Table2 Umbrella)
mocap_args=()
for scene_name in "${scene_order[@]}"; do
  mocap_args+=(--mocap "${scene_name}=${output_root%/}/${scene_name}_v6_${profile}")
done

run_python scripts_run/build_v6_paper_aligned_report.py \
  --wandering-json "$wandering_json" \
  --output-dir "$report_dir" \
  "${mocap_args[@]}"

echo "Finished v6 Wild-SLAM MoCap batch."
echo "Report package: $report_dir"
