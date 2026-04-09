from __future__ import annotations

from typing import Dict, Optional

import cv2
import numpy as np
import torch


def _ensure_bool_mask(region_mask: torch.Tensor) -> torch.Tensor:
    if region_mask.dtype != torch.bool:
        return region_mask > 0.5
    return region_mask


def regularize_region_to_plane(depth: torch.Tensor, region_mask: torch.Tensor, strength: float = 0.25) -> torch.Tensor:
    region_mask = _ensure_bool_mask(region_mask)
    if region_mask.sum() < 20:
        return depth

    h, w = depth.shape
    ys, xs = torch.where(region_mask)
    x_n = xs.float() / max(w - 1, 1)
    y_n = ys.float() / max(h - 1, 1)
    A = torch.stack([x_n, y_n, torch.ones_like(x_n)], dim=-1)
    z = depth[ys, xs].float()

    coeff = torch.linalg.lstsq(A, z.unsqueeze(-1)).solution.squeeze(-1)
    gy, gx = torch.meshgrid(
        torch.linspace(0, 1, h, device=depth.device),
        torch.linspace(0, 1, w, device=depth.device),
        indexing="ij",
    )
    plane = coeff[0] * gx + coeff[1] * gy + coeff[2]

    out = depth.clone()
    blend = float(np.clip(strength, 0.0, 0.7))
    out[region_mask] = (1.0 - blend) * depth[region_mask] + blend * plane[region_mask]
    return out


def reinforce_edge_boundary(depth_edges: torch.Tensor, region_mask: torch.Tensor, strength: float = 0.5) -> torch.Tensor:
    region_mask = _ensure_bool_mask(region_mask)
    if region_mask.sum() < 8:
        return depth_edges

    region_np = region_mask.detach().cpu().numpy().astype(np.uint8) * 255
    boundary = cv2.morphologyEx(region_np, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0

    reinforced = depth_edges.clone().float()
    boundary_t = torch.from_numpy(boundary).to(reinforced.device)
    reinforced[boundary_t] = torch.maximum(reinforced[boundary_t], torch.tensor(float(strength), device=reinforced.device))
    return reinforced > 0.3


def apply_surface_continuity_prior(depth: torch.Tensor, region_mask: torch.Tensor, strength: float = 0.2) -> torch.Tensor:
    region_mask = _ensure_bool_mask(region_mask)
    depth_np = depth.detach().cpu().numpy().astype(np.float32)
    smooth_np = cv2.GaussianBlur(depth_np, (9, 9), sigmaX=0)
    smooth = torch.from_numpy(smooth_np).to(depth.device)

    out = depth.clone()
    blend = float(np.clip(strength, 0.0, 0.45))
    if region_mask.sum() > 0:
        out[region_mask] = (1.0 - blend) * depth[region_mask] + blend * smooth[region_mask]
    return out


def apply_opening_prior(mesh_mask: torch.Tensor, region_mask: torch.Tensor, strength: float = 0.2) -> torch.Tensor:
    region_mask = _ensure_bool_mask(region_mask)
    out = mesh_mask.clone()
    if region_mask.sum() == 0:
        return out
    if strength > 0.15:
        out[region_mask] = False
    return out


def apply_unknown_region_mask(mesh_mask: torch.Tensor, unknown_mask: torch.Tensor, strength: float = 0.2) -> torch.Tensor:
    unknown_mask = _ensure_bool_mask(unknown_mask)
    out = mesh_mask.clone()
    if unknown_mask.sum() == 0:
        return out

    if strength > 0.25:
        out[unknown_mask] = False
    return out


def build_constraint_debug_overlay(base_rgb: np.ndarray, applied_masks: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
    if len(applied_masks) == 0:
        return None

    overlay = base_rgb.copy().astype(np.float32)
    color_lut = {
        "flat_wall": np.array([0, 255, 0], dtype=np.float32),
        "sharp_corner": np.array([255, 255, 0], dtype=np.float32),
        "same_surface": np.array([0, 200, 255], dtype=np.float32),
        "opening": np.array([255, 80, 80], dtype=np.float32),
        "unknown": np.array([160, 160, 160], dtype=np.float32),
    }
    for ctype, mask in applied_masks.items():
        if mask.sum() == 0:
            continue
        color = color_lut.get(ctype, np.array([255, 0, 255], dtype=np.float32))
        overlay[mask] = overlay[mask] * 0.55 + color * 0.45

    return np.clip(overlay, 0, 255).astype(np.uint8)
