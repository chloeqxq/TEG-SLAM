from typing import Tuple

import torch
import torch.nn.functional as F


def max_pool_spatial_map(tensor: torch.Tensor, radius: int) -> torch.Tensor:
    """Max-pool a 2D map to add small spatial tolerance."""
    if radius <= 0:
        return tensor

    kernel_size = radius * 2 + 1
    return (
        F.max_pool2d(
            tensor.view(1, 1, *tensor.shape[-2:]),
            kernel_size=kernel_size,
            stride=1,
            padding=radius,
        )
        .squeeze(0)
        .squeeze(0)
    )


def resample_spatial_map(
    tensor: torch.Tensor,
    target_shape: Tuple[int, int],
    mode: str = "bilinear",
) -> torch.Tensor:
    """Resize a 2D spatial tensor without introducing batch semantics."""
    if tuple(tensor.shape[-2:]) == tuple(target_shape):
        return tensor

    tensor = tensor.view(1, 1, *tensor.shape[-2:])
    return (
        F.interpolate(tensor, size=target_shape, mode=mode, align_corners=False)
        .squeeze(0)
        .squeeze(0)
    )


def apply_uncertainty_data_rate(
    uncertainty: torch.Tensor,
    data_rate: float,
) -> torch.Tensor:
    processed_uncertainty = torch.clip(uncertainty, min=0.1) + 1e-3
    return (processed_uncertainty - 0.1) * data_rate + 0.1


def uncertainty_to_weight(
    adjusted_uncertainty: torch.Tensor,
    min_weight: float = 0.0,
) -> torch.Tensor:
    weights = 0.5 / torch.clamp(adjusted_uncertainty, min=1e-3) ** 2
    return torch.clamp(weights, min=min_weight, max=1.0)


def weight_to_uncertainty(weight: torch.Tensor) -> torch.Tensor:
    safe_weight = torch.clamp(weight, min=1e-6, max=1.0)
    return torch.sqrt(0.5 / safe_weight)


def weight_to_dynamic_score(weight: torch.Tensor) -> torch.Tensor:
    return torch.clamp(1.0 - weight, min=0.0, max=1.0)


def dynamic_score_to_weight(
    dynamic_score: torch.Tensor,
    min_weight: float,
) -> torch.Tensor:
    return torch.clamp(1.0 - dynamic_score, min=min_weight, max=1.0)


def dynamic_score_to_log_odds(
    dynamic_score: torch.Tensor,
    probability_floor: float = 0.15,
    eps: float = 1e-4,
) -> torch.Tensor:
    scaled_probability = probability_floor + (1.0 - 2.0 * probability_floor) * torch.clamp(
        dynamic_score, min=0.0, max=1.0
    )
    scaled_probability = torch.clamp(scaled_probability, min=eps, max=1.0 - eps)
    return torch.log(scaled_probability / (1.0 - scaled_probability))


def log_odds_to_dynamic_score(
    log_odds: torch.Tensor,
    probability_floor: float = 0.15,
) -> torch.Tensor:
    scaled_probability = torch.sigmoid(log_odds)
    dynamic_score = (scaled_probability - probability_floor) / max(
        1.0 - 2.0 * probability_floor, 1e-6
    )
    return torch.clamp(dynamic_score, min=0.0, max=1.0)


def _coords_grid(height: int, width: int, device: torch.device) -> torch.Tensor:
    y, x = torch.meshgrid(
        torch.arange(height, device=device).float(),
        torch.arange(width, device=device).float(),
        indexing="ij",
    )
    return torch.stack([x, y], dim=-1)


def _normalize_grid(coords: torch.Tensor, height: int, width: int) -> torch.Tensor:
    scale_x = max(width - 1, 1)
    scale_y = max(height - 1, 1)
    x = (2.0 * coords[..., 0] / scale_x) - 1.0
    y = (2.0 * coords[..., 1] / scale_y) - 1.0
    return torch.stack([x, y], dim=-1)


def sample_previous_map(
    previous_map: torch.Tensor,
    coords: torch.Tensor,
    valid_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample a previous-frame map using coordinates expressed in that frame."""
    height, width = previous_map.shape[-2:]
    in_bounds = (
        (coords[..., 0] >= 0.0)
        & (coords[..., 0] <= (width - 1))
        & (coords[..., 1] >= 0.0)
        & (coords[..., 1] <= (height - 1))
    )
    combined_valid = valid_mask.bool() & in_bounds

    normalized_grid = _normalize_grid(coords, height, width)
    sampled = F.grid_sample(
        previous_map.view(1, 1, height, width),
        normalized_grid.view(1, height, width, 2),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    ).view(height, width)

    sampled = sampled * combined_valid.float()
    flow_magnitude = torch.linalg.norm(
        coords - _coords_grid(height, width, coords.device), dim=-1
    )
    flow_magnitude = flow_magnitude * combined_valid.float()
    return sampled, combined_valid.float(), flow_magnitude


def fuse_dynamic_scores(
    raw_dynamic: torch.Tensor,
    prior_dynamic: torch.Tensor,
    prior_static_evidence: torch.Tensor,
    valid_mask: torch.Tensor,
    flow_magnitude: torch.Tensor,
    on_threshold: float,
    off_threshold: float,
    release_threshold: float,
    evidence_decay: float,
    min_parallax_px: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fuse single-frame dynamic scores with propagated memory."""
    valid_mask = valid_mask.float()
    raw_static_conf = torch.clamp(1.0 - raw_dynamic, min=0.0, max=1.0)
    raw_static_conf = raw_static_conf * (raw_dynamic < off_threshold).float()

    parallax_gate = torch.clamp(
        flow_magnitude / max(min_parallax_px, 1e-6), min=0.0, max=1.0
    )
    static_measure = raw_static_conf * parallax_gate * valid_mask
    static_evidence = evidence_decay * prior_static_evidence * valid_mask
    static_evidence = static_evidence + (1.0 - evidence_decay) * static_measure
    static_evidence = torch.where(
        raw_dynamic > on_threshold,
        torch.zeros_like(static_evidence),
        static_evidence,
    )

    hold_mask = (
        (prior_dynamic > off_threshold)
        & (static_evidence < release_threshold)
        & valid_mask.bool()
    )
    fused_dynamic = torch.where(
        hold_mask, torch.maximum(raw_dynamic, prior_dynamic), raw_dynamic
    )
    return torch.clamp(fused_dynamic, min=0.0, max=1.0), static_evidence


def fuse_dynamic_scores_log_odds(
    raw_dynamic: torch.Tensor,
    prior_dynamic: torch.Tensor,
    valid_mask: torch.Tensor,
    measurement_gain: float = 1.0,
    probability_floor: float = 0.15,
    log_odds_clip: float = 4.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fuse propagated and current dynamic scores by additive log-odds."""
    measurement_log_odds = measurement_gain * dynamic_score_to_log_odds(
        raw_dynamic, probability_floor=probability_floor
    )
    prior_log_odds = dynamic_score_to_log_odds(
        prior_dynamic, probability_floor=probability_floor
    )
    fused_log_odds = torch.where(
        valid_mask.bool(),
        prior_log_odds + measurement_log_odds,
        measurement_log_odds,
    )
    if log_odds_clip > 0:
        fused_log_odds = torch.clamp(
            fused_log_odds, min=-log_odds_clip, max=log_odds_clip
        )

    fused_dynamic = log_odds_to_dynamic_score(
        fused_log_odds, probability_floor=probability_floor
    )
    static_evidence = torch.clamp(1.0 - fused_dynamic, min=0.0, max=1.0)
    return fused_dynamic, static_evidence


def fuse_uncertainty_with_prior(
    adjusted_uncertainty: torch.Tensor,
    prior_dynamic: torch.Tensor,
    static_evidence: torch.Tensor,
    off_threshold: float,
    release_threshold: float,
    min_weight: float,
    fusion_mode: str = "heuristic",
    log_odds_measurement_gain: float = 1.0,
    log_odds_probability_floor: float = 0.15,
    log_odds_clip: float = 4.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply the stored temporal state to mapping-time uncertainty."""
    raw_weight = uncertainty_to_weight(adjusted_uncertainty, min_weight=min_weight)
    raw_dynamic = weight_to_dynamic_score(raw_weight)
    if fusion_mode == "log_odds":
        fused_dynamic, _ = fuse_dynamic_scores_log_odds(
            raw_dynamic,
            prior_dynamic,
            valid_mask=torch.ones_like(prior_dynamic),
            measurement_gain=log_odds_measurement_gain,
            probability_floor=log_odds_probability_floor,
            log_odds_clip=log_odds_clip,
        )
        fused_weight = dynamic_score_to_weight(fused_dynamic, min_weight=min_weight)
        return weight_to_uncertainty(fused_weight), torch.clamp(
            fused_dynamic, min=0.0, max=1.0
        )

    hold_mask = (prior_dynamic > off_threshold) & (static_evidence < release_threshold)
    fused_dynamic = torch.where(
        hold_mask, torch.maximum(raw_dynamic, prior_dynamic), raw_dynamic
    )
    fused_weight = dynamic_score_to_weight(fused_dynamic, min_weight=min_weight)
    return weight_to_uncertainty(fused_weight), torch.clamp(
        fused_dynamic, min=0.0, max=1.0
    )
