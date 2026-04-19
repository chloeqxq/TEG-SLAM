#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

usage() {
  cat <<'EOF'
Usage: bash scripts_run/run_wild_slam_mocap_v6_sequence.sh <sequence> [options]

Sequence keys:
  anymal1 anymal2 ball crowd person racket stones table1 table2 umbrella

Options:
  --output-root PATH    Output root for v6 MoCap runs.
                        Default: ./output/Wild_SLAM_Mocap_report_v6
  --profile NAME        Temporal config profile to use: report, conservative.
                        Default: report
  --max-frames N        Override max_frames for smoke or tuning runs.
  --skip-existing       Reuse existing SLAM/NVS outputs when present.
  --skip-nvs            Skip the NVS evaluation stage.
  --dry-run             Print the generated config and commands, then exit.
  --keep-config         Keep the generated temporary config file after exit.
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

list_multiprocessing_workers() {
  python - <<'PY'
import subprocess

out = subprocess.check_output(["ps", "-eo", "pid=,stat=,cmd="], text=True)
for line in out.splitlines():
    parts = line.strip().split(None, 2)
    if len(parts) != 3:
        continue
    pid, stat, cmd = parts
    if "Z" in stat:
        continue
    if (
        "from multiprocessing.spawn import spawn_main" in cmd
        or "from multiprocessing.resource_tracker import main" in cmd
    ):
        print(pid)
PY
}

if [ "$#" -lt 1 ]; then
  usage >&2
  exit 1
fi

sequence_key="$1"
shift

output_root="${OUTPUT_ROOT:-./output/Wild_SLAM_Mocap_report_v6}"
profile="${V6_PROFILE:-report}"
max_frames_override=""
skip_existing=0
skip_nvs=0
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
    --profile)
      if [ "$#" -lt 2 ]; then
        echo "--profile requires a value." >&2
        exit 1
      fi
      profile="$2"
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
    --skip-nvs)
      skip_nvs=1
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

case "${sequence_key,,}" in
  anymal1)
    base_config="./configs/Dynamic/Wild_SLAM_Mocap/ANYmal1.yaml"
    paper_scene="ANYmal1"
    log_stem="anymal1"
    input_folder="./datasets/Wild_SLAM_Mocap/scene2/ANYmal1"
    download_hint="scripts_downloading/download_wild_slam_mocap_scene2.sh"
    ;;
  anymal2)
    base_config="./configs/Dynamic/Wild_SLAM_Mocap/ANYmal2.yaml"
    paper_scene="ANYmal2"
    log_stem="anymal2"
    input_folder="./datasets/Wild_SLAM_Mocap/scene2/ANYmal2"
    download_hint="scripts_downloading/download_wild_slam_mocap_scene2.sh"
    ;;
  ball|basketball)
    base_config="./configs/Dynamic/Wild_SLAM_Mocap/ball.yaml"
    paper_scene="Ball"
    log_stem="ball"
    input_folder="./datasets/Wild_SLAM_Mocap/scene1/ball"
    download_hint="scripts_downloading/download_wild_slam_mocap_scene1.sh"
    ;;
  crowd)
    base_config="./configs/Dynamic/Wild_SLAM_Mocap/crowd.yaml"
    paper_scene="Crowd"
    log_stem="crowd"
    input_folder="./datasets/Wild_SLAM_Mocap/scene1/crowd"
    download_hint="scripts_downloading/download_wild_slam_mocap_scene1.sh"
    ;;
  person|person_tracking)
    base_config="./configs/Dynamic/Wild_SLAM_Mocap/person_tracking.yaml"
    paper_scene="Person"
    log_stem="person"
    input_folder="./datasets/Wild_SLAM_Mocap/scene1/person_tracking"
    download_hint="scripts_downloading/download_wild_slam_mocap_scene1.sh"
    ;;
  racket)
    base_config="./configs/Dynamic/Wild_SLAM_Mocap/racket.yaml"
    paper_scene="Racket"
    log_stem="racket"
    input_folder="./datasets/Wild_SLAM_Mocap/scene1/racket"
    download_hint="scripts_downloading/download_wild_slam_mocap_scene1.sh"
    ;;
  stones)
    base_config="./configs/Dynamic/Wild_SLAM_Mocap/stones.yaml"
    paper_scene="Stones"
    log_stem="stones"
    input_folder="./datasets/Wild_SLAM_Mocap/scene1/stones"
    download_hint="scripts_downloading/download_wild_slam_mocap_scene1.sh"
    ;;
  table1|table_tracking1)
    base_config="./configs/Dynamic/Wild_SLAM_Mocap/table_tracking1.yaml"
    paper_scene="Table1"
    log_stem="table1"
    input_folder="./datasets/Wild_SLAM_Mocap/scene1/table_tracking1"
    download_hint="scripts_downloading/download_wild_slam_mocap_scene1.sh"
    ;;
  table2|table_tracking2)
    base_config="./configs/Dynamic/Wild_SLAM_Mocap/table_tracking2.yaml"
    paper_scene="Table2"
    log_stem="table2"
    input_folder="./datasets/Wild_SLAM_Mocap/scene1/table_tracking2"
    download_hint="scripts_downloading/download_wild_slam_mocap_scene1.sh"
    ;;
  umbrella)
    base_config="./configs/Dynamic/Wild_SLAM_Mocap/umbrella.yaml"
    paper_scene="Umbrella"
    log_stem="umbrella"
    input_folder="./datasets/Wild_SLAM_Mocap/scene1/umbrella"
    download_hint="scripts_downloading/download_wild_slam_mocap_scene1.sh"
    ;;
  *)
    echo "Unknown Wild-SLAM MoCap sequence: $sequence_key" >&2
    usage >&2
    exit 1
    ;;
esac

case "${profile,,}" in
  report)
    person_prior_confidence_threshold="0.35"
    insertion_dynamic_threshold="0.30"
    insertion_person_confidence_threshold="0.35"
    insertion_person_dilation_radius="2"
    post_cleanup_dynamic_threshold="0.45"
    post_cleanup_opacity_decay_factor="0.45"
    post_cleanup_opacity_decay_max_ratio="0.05"
    ;;
  conservative)
    person_prior_confidence_threshold="0.40"
    insertion_dynamic_threshold="0.35"
    insertion_person_confidence_threshold="0.40"
    insertion_person_dilation_radius="1"
    post_cleanup_dynamic_threshold="0.45"
    post_cleanup_opacity_decay_factor="0.40"
    post_cleanup_opacity_decay_max_ratio="0.035"
    ;;
  *)
    echo "Unknown v6 profile: $profile" >&2
    echo "Supported profiles: report, conservative" >&2
    exit 1
    ;;
esac

output_root="${output_root%/}"
scene_name="${paper_scene}_v6_${profile}"
output_dir="$output_root/$scene_name"
metrics_file="$output_dir/traj/metrics_full_traj.txt"
nvs_file="$output_dir/nvs/final_result.json"
log_dir="$output_root/logs"
log_file="$log_dir/${log_stem}_${profile}.log"
tmp_config_dir="$REPO_ROOT/.tmp_wildgs_slam_configs"
required_file="$input_folder/rgb.txt"
post_run_wait_secs="${POST_RUN_WAIT_SECS:-7200}"

mkdir -p "$log_dir" "$tmp_config_dir"

temp_config="$(mktemp "$tmp_config_dir/${log_stem}.${profile}.XXXXXX.yaml")"
cleanup() {
  if [ "$keep_config" -ne 1 ]; then
    rm -f "$temp_config"
  fi
}
trap cleanup EXIT

cat > "$temp_config" <<EOF
inherit_from: "$base_config"
scene: "$scene_name"

data:
  output: "$output_root"

mapping:
  uncertainty_params:
    temporal_params:
      activate: True

      tracking_use_temporal_posterior: True
      mapping_use_temporal_posterior: True
      fusion_mode: log_odds
      log_odds_measurement_gain: 1.0
      log_odds_probability_floor: 0.15
      log_odds_clip: 4.0

      person_prior_activate: True
      person_prior_class_id: 15
      person_prior_confidence_threshold: ${person_prior_confidence_threshold}
      person_prior_dilation_radius: 1
      person_prior_min_dynamic: 0.70

      insertion_mask_activate: True
      insertion_dynamic_threshold: ${insertion_dynamic_threshold}
      insertion_dilation_radius: 2
      insertion_max_mask_ratio: 0.45
      insertion_min_remaining_depth_ratio: 0.20
      insertion_uncertainty_threshold: -1.0

      insertion_use_person_segmentation: True
      insertion_person_mask_mode: score_floor
      insertion_person_class_id: 15
      insertion_person_confidence_threshold: ${insertion_person_confidence_threshold}
      insertion_person_dilation_radius: ${insertion_person_dilation_radius}
      insertion_person_erosion_radius: 0
      insertion_person_min_dynamic: 0.90

      insertion_enable_adaptive_threshold: False
      insertion_enable_min_remaining_adaptive_threshold: False
      insertion_fail_closed_skip: True
      insertion_fail_closed_ratio: 0.50

      post_cleanup_activate: False
      post_cleanup_opacity_decay_activate: True
      post_cleanup_dynamic_threshold: ${post_cleanup_dynamic_threshold}
      post_cleanup_dilation_radius: 1
      post_cleanup_max_mask_ratio: 0.20
      post_cleanup_min_mask_ratio: 0.005

      post_cleanup_opacity_decay_min_opacity: 0.08
      post_cleanup_opacity_decay_erosion_radius: 1
      post_cleanup_opacity_decay_max_ratio: ${post_cleanup_opacity_decay_max_ratio}
      post_cleanup_opacity_decay_factor: ${post_cleanup_opacity_decay_factor}
      post_cleanup_opacity_decay_floor: 0.03
      post_cleanup_opacity_decay_mode: gaussian_evidence

      post_cleanup_evidence_view_min_opacity: 0.05
      post_cleanup_evidence_min_observations: 2
      post_cleanup_evidence_min_dynamic_hits: 2
      post_cleanup_evidence_dynamic_ratio_threshold: 0.55
      post_cleanup_evidence_max_static_ratio: 0.45
      post_cleanup_evidence_dynamic_margin: 0.0
      post_cleanup_evidence_depth_abs_tolerance: 0.08
      post_cleanup_evidence_depth_rel_tolerance: 0.05

      demo_dynamic_eraser_activate: False
EOF

if [ -n "$max_frames_override" ]; then
  cat >> "$temp_config" <<EOF

max_frames: ${max_frames_override}
EOF
fi

if [ "$skip_existing" -eq 1 ] && [ -f "$metrics_file" ]; then
  echo "Reusing existing SLAM output for $scene_name because $metrics_file already exists."
else
  if [ "$dry_run" -eq 1 ]; then
    echo "Base config: $base_config"
    echo "Paper scene: $paper_scene"
    echo "Profile: $profile"
    echo "Output dir: $output_dir"
    echo "Expected data index: $required_file"
    echo "Generated config: $temp_config"
    echo "Command: python -u run.py \"$temp_config\""
    if [ "$skip_nvs" -ne 1 ]; then
      echo "NVS command: python scripts_run/eval_single_wild_slam_nvs.py --exp-dir \"$output_dir\""
    fi
    exit 0
  fi

  if [ ! -f "$required_file" ]; then
    echo "Missing dataset sequence for $paper_scene: $required_file" >&2
    echo "Download the full Wild-SLAM MoCap data first:" >&2
    echo "  bash $download_hint" >&2
    exit 1
  fi

  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"

  echo "Running WildGS-SLAM v6 sequence: $scene_name"
  echo "Base config: $base_config"
  echo "Profile: $profile"
  echo "Output dir: $output_dir"
  echo "Log file: $log_file"

  mapfile -t baseline_workers < <(list_multiprocessing_workers)

  set +e
  run_python -u run.py "$temp_config" 2>&1 | tee "$log_file"
  run_status=${PIPESTATUS[0]}
  set -e

  deadline=$(( $(date +%s) + post_run_wait_secs ))
  while [ ! -f "$metrics_file" ]; do
    mapfile -t current_workers < <(list_multiprocessing_workers)
    new_workers=()
    for pid in "${current_workers[@]}"; do
      is_baseline=0
      for baseline_pid in "${baseline_workers[@]}"; do
        if [ "$pid" = "$baseline_pid" ]; then
          is_baseline=1
          break
        fi
      done
      if [ "$is_baseline" -eq 0 ]; then
        new_workers+=("$pid")
      fi
    done

    if [ "${#new_workers[@]}" -eq 0 ]; then
      break
    fi

    if [ "$(date +%s)" -ge "$deadline" ]; then
      break
    fi

    echo "run.py parent exited; waiting for worker processes to finish: ${new_workers[*]}"
    sleep 30
  done

  if [ ! -f "$metrics_file" ]; then
    if [ "$run_status" -ne 0 ]; then
      echo "run.py exited with status $run_status." >&2
    fi
    echo "Run finished but $metrics_file was not generated." >&2
    exit 1
  fi
fi

if [ "$skip_nvs" -eq 1 ]; then
  echo "Skipping NVS for $scene_name."
  exit 0
fi

if [ "$skip_existing" -eq 1 ] && [ -f "$nvs_file" ]; then
  echo "Reusing existing NVS output for $scene_name because $nvs_file already exists."
  exit 0
fi

if [ ! -f "$output_dir/final_gs.ply" ]; then
  echo "Missing Gaussian map for NVS evaluation: $output_dir/final_gs.ply" >&2
  exit 1
fi

echo "Running NVS evaluation for $scene_name"
run_python scripts_run/eval_single_wild_slam_nvs.py --exp-dir "$output_dir"
echo "Finished $scene_name"
echo "Trajectory metrics: $metrics_file"
echo "NVS metrics: $nvs_file"
