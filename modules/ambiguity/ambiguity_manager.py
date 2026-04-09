import json
from typing import Dict, List, Tuple

import cv2
import numpy as np

from .constraint_types import ALLOWED_CHOICES, RegionProposal


class AmbiguityManager:
    def __init__(self, top_k: int = 3):
        self.top_k = max(1, min(3, top_k))

    def _norm(self, x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32)
        mn, mx = float(x.min()), float(x.max())
        if mx - mn < 1e-6:
            return np.zeros_like(x, dtype=np.float32)
        return (x - mn) / (mx - mn)

    def compute(self, pano_rgb: np.ndarray, init_depth: np.ndarray, depth_edges: np.ndarray) -> Dict:
        gray = cv2.cvtColor((np.clip(pano_rgb, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_energy = np.sqrt(grad_x ** 2 + grad_y ** 2)
        low_texture = 1.0 - self._norm(grad_energy)

        depth = init_depth.astype(np.float32)
        depth_dx = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
        depth_dy = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)
        depth_grad = np.sqrt(depth_dx ** 2 + depth_dy ** 2)

        depth_blur = cv2.GaussianBlur(depth, (0, 0), sigmaX=1.0)
        perturbed_edges = cv2.Canny(((depth_blur / max(depth_blur.max(), 1e-6)) * 255).astype(np.uint8), 60, 150)
        edge_instability = np.abs((depth_edges > 0).astype(np.float32) - (perturbed_edges > 0).astype(np.float32))

        smoothness_disagree = np.abs(self._norm(depth_grad) - self._norm((depth_edges > 0).astype(np.float32)))

        lap = cv2.Laplacian(depth, cv2.CV_32F, ksize=3)
        curvature = np.abs(lap)
        high_curvature_low_texture = self._norm(curvature) * low_texture

        ambiguity = (
            0.35 * low_texture
            + 0.25 * self._norm(edge_instability)
            + 0.20 * self._norm(smoothness_disagree)
            + 0.20 * self._norm(high_curvature_low_texture)
        )
        ambiguity = self._norm(ambiguity)

        proposals = self._propose_regions(ambiguity, low_texture, edge_instability, smoothness_disagree, high_curvature_low_texture)
        heatmap = cv2.applyColorMap((ambiguity * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlay = self._make_overlay(pano_rgb, heatmap, proposals)

        queries = self._build_queries(proposals)
        return {
            "ambiguity_map": ambiguity,
            "heatmap": heatmap,
            "overlay": overlay,
            "proposals": [p.__dict__ for p in proposals],
            "queries": queries,
        }

    def _propose_regions(
        self,
        ambiguity: np.ndarray,
        low_texture: np.ndarray,
        edge_instability: np.ndarray,
        smoothness_disagree: np.ndarray,
        curvature_low_texture: np.ndarray,
    ) -> List[RegionProposal]:
        threshold = np.percentile(ambiguity, 90)
        mask = (ambiguity >= threshold).astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

        regions: List[RegionProposal] = []
        for label in range(1, num_labels):
            x, y, w, h, area = stats[label]
            if area < 200:
                continue
            ys, xs = np.where(labels == label)
            score = float(ambiguity[ys, xs].mean())
            reasons = {
                "low_texture": float(low_texture[ys, xs].mean()),
                "edge_instability": float(edge_instability[ys, xs].mean()),
                "smoothness_edge_disagreement": float(smoothness_disagree[ys, xs].mean()),
                "curvature_in_low_texture": float(curvature_low_texture[ys, xs].mean()),
            }
            regions.append(
                RegionProposal(
                    region_id=f"r{len(regions)+1}",
                    bbox=[int(x), int(y), int(x + w), int(y + h)],
                    score=score,
                    reasons=reasons,
                )
            )

        regions = sorted(regions, key=lambda r: r.score, reverse=True)[: self.top_k]
        return regions

    def _make_overlay(self, pano_rgb: np.ndarray, heatmap_rgb: np.ndarray, proposals: List[RegionProposal]) -> np.ndarray:
        base = (np.clip(pano_rgb, 0, 1) * 255).astype(np.uint8)
        overlay = cv2.addWeighted(base, 0.6, heatmap_rgb, 0.4, 0)
        for proposal in proposals:
            x1, y1, x2, y2 = proposal.bbox
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 255), 2)
            cv2.putText(overlay, proposal.region_id, (x1, max(20, y1 + 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return overlay

    def _build_queries(self, proposals: List[RegionProposal]) -> Dict:
        queries = []
        for proposal in proposals:
            queries.append(
                {
                    "region_id": proposal.region_id,
                    "question_type": "layout_classification",
                    "question": "How should this ambiguous region be interpreted?",
                    "allowed_choices": ALLOWED_CHOICES,
                    "bbox": proposal.bbox,
                    "debug": {"score": proposal.score, "reasons": proposal.reasons},
                }
            )
        return {"queries": queries}


def dump_queries_json(path: str, queries: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(queries, f, indent=2)
