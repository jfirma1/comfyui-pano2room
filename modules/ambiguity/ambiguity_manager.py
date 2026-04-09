from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch

from .constraint_injection import (
    apply_opening_prior,
    apply_surface_continuity_prior,
    apply_unknown_region_mask,
    build_constraint_debug_overlay,
    regularize_region_to_plane,
    reinforce_edge_boundary,
)
from .constraint_types import ALLOWED_CHOICES, ClarificationQuery, ConstraintLabel, RegionConstraint


class AmbiguityManager:
    def __init__(self, max_regions: int = 3):
        self.max_regions = max(1, min(int(max_regions), 3))

    def detect_initial_ambiguity(
        self,
        pano_rgb: torch.Tensor,
        init_depth: torch.Tensor,
        depth_edges: np.ndarray,
        sup_pool: Any = None,
    ) -> Dict[str, Any]:
        rgb_np = pano_rgb.detach().cpu().permute(1, 2, 0).numpy()
        rgb_u8 = np.clip(rgb_np * 255.0, 0, 255).astype(np.uint8)
        gray = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

        depth_np = init_depth.detach().cpu().numpy().astype(np.float32)
        edge_np = (depth_edges > 0).astype(np.float32)

        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(gx * gx + gy * gy)
        low_texture = 1.0 - self._normalize(grad_mag)

        stable_edges = cv2.Canny((depth_np / (depth_np.max() + 1e-6) * 255).astype(np.uint8), 50, 140) > 0
        depth_blur = cv2.GaussianBlur(depth_np, (5, 5), sigmaX=0)
        perturbed_edges = cv2.Canny((depth_blur / (depth_blur.max() + 1e-6) * 255).astype(np.uint8), 50, 140) > 0
        edge_instability = np.logical_xor(stable_edges, perturbed_edges).astype(np.float32)

        lap = cv2.Laplacian(depth_np, cv2.CV_32F, ksize=3)
        high_curv = self._normalize(np.abs(lap))
        smoothness_edge_disagree = np.clip(high_curv - edge_np, 0.0, 1.0)
        lowtex_highcurv = low_texture * high_curv

        conflict_signal = np.zeros_like(depth_np, dtype=np.float32)
        if sup_pool is not None:
            stats = getattr(sup_pool, "last_geo_check_stats", None)
            if isinstance(stats, dict) and stats.get("conflict_density", None) is not None:
                conflict_signal += float(np.clip(stats["conflict_density"], 0.0, 1.0))

        ambiguity_map = (
            0.30 * low_texture
            + 0.25 * edge_instability
            + 0.20 * smoothness_edge_disagree
            + 0.20 * lowtex_highcurv
            + 0.05 * conflict_signal
        )
        ambiguity_map = self._normalize(ambiguity_map)

        proposals = self._propose_regions(ambiguity_map, low_texture, edge_instability, smoothness_edge_disagree, lowtex_highcurv)

        return {
            "ambiguity_map": ambiguity_map,
            "region_proposals": proposals,
            "debug_meta": {
                "low_texture_mean": float(low_texture.mean()),
                "edge_instability_mean": float(edge_instability.mean()),
                "smooth_edge_disagree_mean": float(smoothness_edge_disagree.mean()),
                "high_curv_lowtex_mean": float(lowtex_highcurv.mean()),
            },
        }

    def build_queries(self, region_proposals: List[Dict[str, Any]]) -> List[ClarificationQuery]:
        queries = []
        for proposal in region_proposals:
            mask = proposal["mask"].astype(np.uint8)
            query = ClarificationQuery(
                region_id=proposal["region_id"],
                question_type="region_geometry_relation",
                allowed_choices=list(ALLOWED_CHOICES),
                bbox=tuple(proposal["bbox"]),
                mask_rle=self._encode_mask_rle(mask, proposal["bbox"]),
                explanation=proposal.get("reason", "high ambiguity region"),
                debug={
                    "score": proposal.get("score", 0.0),
                    "signal_breakdown": proposal.get("signal_breakdown", {}),
                },
            )
            queries.append(query)
        return queries

    def load_constraints(self, path_or_obj: Optional[Union[str, Dict[str, Any], List[Dict[str, Any]]]]) -> List[RegionConstraint]:
        if path_or_obj is None:
            return []

        raw: Any
        if isinstance(path_or_obj, str):
            if not os.path.exists(path_or_obj):
                return []
            with open(path_or_obj, "r", encoding="utf-8") as f:
                raw = json.load(f)
        else:
            raw = path_or_obj

        if isinstance(raw, dict):
            entries = raw.get("constraints", raw.get("answers", []))
        else:
            entries = raw

        constraints = []
        for entry in entries:
            try:
                constraints.append(RegionConstraint.from_dict(entry))
            except Exception:
                continue
        return constraints

    def apply_constraints(
        self,
        init_depth: torch.Tensor,
        depth_edges: torch.Tensor,
        mesh_mask: torch.Tensor,
        constraints: List[RegionConstraint],
        pano_rgb: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        out_depth = init_depth.clone()
        out_edges = depth_edges.clone().bool()
        out_mesh_mask = mesh_mask.clone().bool()

        h, w = out_depth.shape
        applied_masks: Dict[str, np.ndarray] = {}
        applied = []
        for c in constraints:
            x0, y0, x1, y1 = [int(v) for v in c.bbox]
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(w - 1, x1), min(h - 1, y1)
            if x1 <= x0 or y1 <= y0:
                continue

            region_mask = torch.zeros((h, w), device=out_depth.device, dtype=torch.bool)
            region_mask[y0 : y1 + 1, x0 : x1 + 1] = True
            strength = float(np.clip(c.confidence, 0.0, 1.0))
            ctype = c.constraint_type

            if ctype == ConstraintLabel.FLAT_WALL.value:
                out_depth = regularize_region_to_plane(out_depth, region_mask, 0.15 + 0.35 * strength)
            elif ctype == ConstraintLabel.SHARP_CORNER.value:
                out_edges = reinforce_edge_boundary(out_edges, region_mask, 0.4 + 0.5 * strength)
            elif ctype == ConstraintLabel.SAME_SURFACE.value:
                out_depth = apply_surface_continuity_prior(out_depth, region_mask, 0.1 + 0.25 * strength)
            elif ctype == ConstraintLabel.OPENING.value:
                out_mesh_mask = apply_opening_prior(out_mesh_mask, region_mask, 0.2 + 0.6 * strength)
            else:
                out_mesh_mask = apply_unknown_region_mask(out_mesh_mask, region_mask, 0.2 + 0.6 * strength)

            applied_masks[ctype] = applied_masks.get(ctype, np.zeros((h, w), dtype=bool)) | region_mask.detach().cpu().numpy()
            applied.append(c.to_dict())

        debug_overlay = None
        if pano_rgb is not None and len(applied) > 0:
            rgb = np.clip(pano_rgb.detach().cpu().permute(1, 2, 0).numpy() * 255.0, 0, 255).astype(np.uint8)
            debug_overlay = build_constraint_debug_overlay(rgb, applied_masks)

        return {
            "depth": out_depth,
            "depth_edges": out_edges,
            "mesh_mask": out_mesh_mask,
            "applied": applied,
            "debug_overlay": debug_overlay,
        }

    def export_queries_json(self, queries: List[ClarificationQuery], output_path: str) -> None:
        payload = {
            "schema_version": "0.1",
            "description": "MVP ambiguity clarification queries for Pano2Room",
            "queries": [q.to_dict() for q in queries],
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def _propose_regions(
        self,
        ambiguity_map: np.ndarray,
        low_texture: np.ndarray,
        edge_instability: np.ndarray,
        smoothness_edge_disagree: np.ndarray,
        lowtex_highcurv: np.ndarray,
    ) -> List[Dict[str, Any]]:
        threshold = np.percentile(ambiguity_map, 93)
        mask = (ambiguity_map >= threshold).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        proposals: List[Dict[str, Any]] = []
        for label in range(1, num_labels):
            x, y, w, h, area = stats[label]
            if area < 1200:
                continue
            comp_mask = labels == label
            score = float(ambiguity_map[comp_mask].mean())
            proposals.append(
                {
                    "region_id": f"region_{label}",
                    "bbox": [int(x), int(y), int(x + w - 1), int(y + h - 1)],
                    "mask": comp_mask,
                    "score": score,
                    "signal_breakdown": {
                        "low_texture": float(low_texture[comp_mask].mean()),
                        "edge_instability": float(edge_instability[comp_mask].mean()),
                        "smoothness_edge_disagree": float(smoothness_edge_disagree[comp_mask].mean()),
                        "lowtex_highcurv": float(lowtex_highcurv[comp_mask].mean()),
                    },
                    "reason": "High ambiguity score from combined handcrafted signals",
                }
            )

        proposals = sorted(proposals, key=lambda x: x["score"], reverse=True)[: self.max_regions]
        return proposals

    def _encode_mask_rle(self, mask: np.ndarray, bbox: Tuple[int, int, int, int]) -> List[List[int]]:
        x0, y0, x1, y1 = bbox
        crop = mask[y0 : y1 + 1, x0 : x1 + 1].astype(np.uint8).reshape(-1)
        if crop.size == 0:
            return []

        rle = []
        prev = int(crop[0])
        count = 1
        for val in crop[1:]:
            v = int(val)
            if v == prev:
                count += 1
            else:
                rle.append([prev, count])
                prev = v
                count = 1
        rle.append([prev, count])
        return rle

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32)
        mn, mx = float(np.min(x)), float(np.max(x))
        if mx - mn < 1e-6:
            return np.zeros_like(x, dtype=np.float32)
        return (x - mn) / (mx - mn)
