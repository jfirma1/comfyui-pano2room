import json
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .constraint_types import ALLOWED_CHOICES, RegionProposal


class AmbiguityManager:
    def __init__(
        self,
        top_k: int = 3,
        nms_iou_threshold: float = 0.45,
        near_duplicate_center_radius_ratio: float = 0.10,
        near_duplicate_score_delta: float = 0.04,
        diversity_radius_ratio: float = 0.22,
        diversity_penalty: float = 0.20,
        cluster_suppression_radius_ratio: float = 0.18,
        cluster_overlap_threshold: float = 0.10,
        distinct_area_scale_ratio: float = 1.80,
        distinct_structural_priority_delta: float = 0.20,
        distinct_edge_behavior_delta: float = 0.18,
        weak_low_texture_threshold: float = 0.72,
        weak_edge_instability_threshold: float = 0.12,
        weak_disagreement_threshold: float = 0.12,
        weak_structural_priority_threshold: float = 0.28,
    ):
        self.top_k = max(1, min(3, top_k))
        self.nms_iou_threshold = float(nms_iou_threshold)
        self.near_duplicate_center_radius_ratio = float(near_duplicate_center_radius_ratio)
        self.near_duplicate_score_delta = float(near_duplicate_score_delta)
        self.diversity_radius_ratio = float(diversity_radius_ratio)
        self.diversity_penalty = float(diversity_penalty)
        self.cluster_suppression_radius_ratio = float(cluster_suppression_radius_ratio)
        self.cluster_overlap_threshold = float(cluster_overlap_threshold)
        self.distinct_area_scale_ratio = float(distinct_area_scale_ratio)
        self.distinct_structural_priority_delta = float(distinct_structural_priority_delta)
        self.distinct_edge_behavior_delta = float(distinct_edge_behavior_delta)
        self.weak_low_texture_threshold = float(weak_low_texture_threshold)
        self.weak_edge_instability_threshold = float(weak_edge_instability_threshold)
        self.weak_disagreement_threshold = float(weak_disagreement_threshold)
        self.weak_structural_priority_threshold = float(weak_structural_priority_threshold)

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

        edge_instability_n = self._norm(edge_instability)
        smoothness_disagree_n = self._norm(smoothness_disagree)
        curvature_low_texture_n = self._norm(high_curvature_low_texture)

        # Rebalanced ambiguity map:
        #  - lower low-texture dominance
        #  - emphasize edge/smoothness disagreement and structural cues
        ambiguity = (
            0.18 * low_texture
            + 0.30 * edge_instability_n
            + 0.27 * smoothness_disagree_n
            + 0.25 * curvature_low_texture_n
        )
        ambiguity = self._norm(ambiguity)

        proposals = self._propose_regions(
            ambiguity,
            low_texture,
            edge_instability,
            smoothness_disagree,
            high_curvature_low_texture,
        )
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
        image_h, image_w = ambiguity.shape
        edge_instability_n = self._norm(edge_instability)
        smoothness_disagree_n = self._norm(smoothness_disagree)
        curvature_low_texture_n = self._norm(curvature_low_texture)

        regions: List[RegionProposal] = []
        for label in range(1, num_labels):
            x, y, w, h, area = stats[label]
            if area < 200:
                continue
            ys, xs = np.where(labels == label)
            base_score = float(ambiguity[ys, xs].mean())
            center_x = float(x + 0.5 * w)
            center_y = float(y + 0.5 * h)
            normalized_area = float(area) / float(max(1, image_h * image_w))
            reasons = self._compute_reason_terms(
                xs,
                ys,
                x,
                y,
                w,
                h,
                image_h,
                image_w,
                low_texture,
                edge_instability_n,
                smoothness_disagree_n,
                curvature_low_texture_n,
            )
            struct_priority = float(reasons["structural_priority"])
            tiny_penalty = float(reasons["tiny_region_penalty"])
            fragment_penalty = float(reasons["edge_fragment_penalty"])
            score = float(0.45 * base_score + 0.55 * struct_priority - tiny_penalty - fragment_penalty)
            reasons = {
                "low_texture": float(low_texture[ys, xs].mean()),
                "edge_instability": float(edge_instability[ys, xs].mean()),
                "smoothness_edge_disagreement": float(smoothness_disagree[ys, xs].mean()),
                "curvature_in_low_texture": float(curvature_low_texture[ys, xs].mean()),
                "base_ambiguity": base_score,
                "structural_priority": struct_priority,
                "tiny_region_penalty": tiny_penalty,
                "edge_fragment_penalty": fragment_penalty,
                "normalized_area": normalized_area,
            }
            regions.append(
                RegionProposal(
                    region_id=f"r{len(regions)+1}",
                    bbox=[int(x), int(y), int(x + w), int(y + h)],
                    score=score,
                    reasons=reasons,
                    debug={
                        "center_xy": [center_x, center_y],
                        "area_px": int(area),
                        "pre_nms_rank_score": score,
                    },
                )
            )

        ranked_candidates = sorted(regions, key=lambda r: r.score, reverse=True)
        pre_nms_count = len(ranked_candidates)
        nms_regions = self._suppress_duplicates(ranked_candidates, image_w=image_w, image_h=image_h)
        post_nms_count = len(nms_regions)
        selected, rejected = self._select_final_topk(nms_regions, image_w=image_w, image_h=image_h)

        for idx, proposal in enumerate(selected):
            proposal.region_id = f"r{idx + 1}"
            proposal.debug = proposal.debug or {}
            proposal.debug["pre_nms_candidate_count"] = pre_nms_count
            proposal.debug["post_nms_candidate_count"] = post_nms_count
            proposal.debug["rejected_candidate_count"] = len(rejected)

        self._last_rejected_candidates = rejected
        return selected

    def _compute_reason_terms(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        x: int,
        y: int,
        w: int,
        h: int,
        image_h: int,
        image_w: int,
        low_texture: np.ndarray,
        edge_instability_n: np.ndarray,
        smoothness_disagree_n: np.ndarray,
        curvature_low_texture_n: np.ndarray,
    ) -> Dict[str, float]:
        e = float(edge_instability_n[ys, xs].mean())
        s = float(smoothness_disagree_n[ys, xs].mean())
        c = float(curvature_low_texture_n[ys, xs].mean())
        lt = float(low_texture[ys, xs].mean())

        # Lightweight structural priority terms (heuristic-only, deterministic)
        center_x = x + 0.5 * w
        center_y = y + 0.5 * h
        lower_half_bias = np.clip((center_y / max(1.0, float(image_h))) - 0.45, 0.0, 0.55) / 0.55

        aspect = float(w) / max(1.0, float(h))
        elongatedness = min(abs(np.log(max(aspect, 1e-6))), 2.0) / 2.0
        boundary_proximity = 1.0 - min(
            center_x / max(1.0, image_w),
            center_y / max(1.0, image_h),
            (image_w - center_x) / max(1.0, image_w),
            (image_h - center_y) / max(1.0, image_h),
        )
        boundary_proximity = float(np.clip(boundary_proximity, 0.0, 1.0))

        area = float(len(xs))
        tiny_region_penalty = float(np.clip((300.0 - area) / 300.0, 0.0, 1.0) * 0.15)
        edge_fragment_penalty = float(np.clip((e - c) * (1.0 - s), 0.0, 1.0) * 0.12)

        structural_priority = (
            0.32 * e
            + 0.24 * s
            + 0.18 * c
            + 0.14 * boundary_proximity
            + 0.08 * lower_half_bias
            + 0.04 * elongatedness
            - 0.08 * np.clip(lt - 0.65, 0.0, 1.0)  # reduce over-prioritizing pure low-texture patches
        )
        structural_priority = float(np.clip(structural_priority, 0.0, 1.0))

        return {
            "structural_priority": structural_priority,
            "tiny_region_penalty": tiny_region_penalty,
            "edge_fragment_penalty": edge_fragment_penalty,
            "boundary_proximity": boundary_proximity,
            "lower_half_bias": float(lower_half_bias),
            "elongatedness": float(elongatedness),
        }

    def _bbox_iou(self, a: List[int], b: List[int]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
        inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
        iw = max(0, inter_x2 - inter_x1)
        ih = max(0, inter_y2 - inter_y1)
        inter = float(iw * ih)
        if inter <= 0:
            return 0.0
        area_a = float(max(1, ax2 - ax1) * max(1, ay2 - ay1))
        area_b = float(max(1, bx2 - bx1) * max(1, by2 - by1))
        return inter / max(area_a + area_b - inter, 1e-6)

    def _proposal_center(self, proposal: RegionProposal) -> Tuple[float, float]:
        x1, y1, x2, y2 = proposal.bbox
        return (0.5 * (x1 + x2), 0.5 * (y1 + y2))

    def _bbox_intersection_over_min_area(self, a: List[int], b: List[int]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
        inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
        iw = max(0, inter_x2 - inter_x1)
        ih = max(0, inter_y2 - inter_y1)
        inter = float(iw * ih)
        if inter <= 0:
            return 0.0
        area_a = float(max(1, ax2 - ax1) * max(1, ay2 - ay1))
        area_b = float(max(1, bx2 - bx1) * max(1, by2 - by1))
        return inter / max(min(area_a, area_b), 1e-6)

    def _is_weak_low_structure_region(self, proposal: RegionProposal) -> bool:
        reasons = proposal.reasons or {}
        low_texture = float(reasons.get("low_texture", 0.0))
        edge_instability = float(reasons.get("edge_instability", 0.0))
        disagreement = float(reasons.get("smoothness_edge_disagreement", 0.0))
        structural_priority = float(reasons.get("structural_priority", 0.0))
        return (
            low_texture >= self.weak_low_texture_threshold
            and edge_instability <= self.weak_edge_instability_threshold
            and disagreement <= self.weak_disagreement_threshold
            and structural_priority <= self.weak_structural_priority_threshold
        )

    def _is_materially_distinct(self, candidate: RegionProposal, selected: RegionProposal) -> bool:
        c_reasons = candidate.reasons or {}
        s_reasons = selected.reasons or {}
        c_area = float(c_reasons.get("normalized_area", 0.0))
        s_area = float(s_reasons.get("normalized_area", 0.0))
        area_scale = max(c_area, s_area) / max(min(c_area, s_area), 1e-6)
        structural_delta = abs(
            float(c_reasons.get("structural_priority", 0.0)) - float(s_reasons.get("structural_priority", 0.0))
        )
        edge_delta = max(
            abs(float(c_reasons.get("edge_instability", 0.0)) - float(s_reasons.get("edge_instability", 0.0))),
            abs(
                float(c_reasons.get("smoothness_edge_disagreement", 0.0))
                - float(s_reasons.get("smoothness_edge_disagreement", 0.0))
            ),
            abs(float(c_reasons.get("curvature_in_low_texture", 0.0)) - float(s_reasons.get("curvature_in_low_texture", 0.0))),
        )
        return (
            area_scale >= self.distinct_area_scale_ratio
            or structural_delta >= self.distinct_structural_priority_delta
            or edge_delta >= self.distinct_edge_behavior_delta
        )

    def _suppress_duplicates(self, ranked_candidates: List[RegionProposal], image_w: int, image_h: int) -> List[RegionProposal]:
        kept: List[RegionProposal] = []
        diag = float(np.hypot(image_w, image_h))
        center_radius = self.near_duplicate_center_radius_ratio * diag

        for candidate in ranked_candidates:
            suppressed_by: Optional[Dict] = None
            c_cx, c_cy = self._proposal_center(candidate)
            for kept_region in kept:
                iou = self._bbox_iou(candidate.bbox, kept_region.bbox)
                kx, ky = self._proposal_center(kept_region)
                center_dist = float(np.hypot(c_cx - kx, c_cy - ky))
                score_gap = abs(float(candidate.score - kept_region.score))
                is_near_duplicate = center_dist <= center_radius and score_gap <= self.near_duplicate_score_delta
                if iou >= self.nms_iou_threshold or is_near_duplicate:
                    suppressed_by = {
                        "kept_region_id": kept_region.region_id,
                        "iou": float(iou),
                        "center_distance": center_dist,
                        "score_gap": score_gap,
                        "reason": "iou_nms" if iou >= self.nms_iou_threshold else "near_duplicate",
                    }
                    break

            candidate.debug = candidate.debug or {}
            candidate.debug["suppressed_by"] = suppressed_by
            if suppressed_by is None:
                kept.append(candidate)
        return kept

    def _select_final_topk(
        self, candidates: List[RegionProposal], image_w: int, image_h: int
    ) -> Tuple[List[RegionProposal], List[Dict]]:
        diag = float(np.hypot(image_w, image_h))
        diversity_radius = self.diversity_radius_ratio * diag
        cluster_radius = self.cluster_suppression_radius_ratio * diag
        selected: List[RegionProposal] = []
        rejected: List[Dict] = []
        cluster_count = 0

        ranked = sorted(
            candidates,
            key=lambda c: (
                float(c.score),
                float((c.reasons or {}).get("structural_priority", 0.0)),
                float((c.reasons or {}).get("edge_instability", 0.0))
                + float((c.reasons or {}).get("smoothness_edge_disagreement", 0.0)),
            ),
            reverse=True,
        )

        for candidate in ranked:
            if len(selected) >= self.top_k:
                rejected.append(
                    {
                        "region_id": candidate.region_id,
                        "inclusion_reason": "rejected_topk_limit",
                    }
                )
                continue

            cx, cy = self._proposal_center(candidate)
            nearest = None
            min_dist = diag
            for proposal in selected:
                px, py = self._proposal_center(proposal)
                dist = float(np.hypot(cx - px, cy - py))
                if dist < min_dist:
                    min_dist = dist
                    nearest = proposal

            if self._is_weak_low_structure_region(candidate):
                rejected.append(
                    {
                        "region_id": candidate.region_id,
                        "nearest_selected_region_id": nearest.region_id if nearest else None,
                        "cluster_id": (nearest.debug or {}).get("cluster_id") if nearest else None,
                        "inclusion_reason": "rejected_low_structure",
                    }
                )
                continue

            overlap_with_nearest = (
                self._bbox_intersection_over_min_area(candidate.bbox, nearest.bbox) if nearest is not None else 0.0
            )
            near_cluster = (
                nearest is not None and min_dist <= cluster_radius and overlap_with_nearest >= self.cluster_overlap_threshold
            )

            if near_cluster and not self._is_materially_distinct(candidate, nearest):
                rejected.append(
                    {
                        "region_id": candidate.region_id,
                        "nearest_selected_region_id": nearest.region_id,
                        "cluster_id": (nearest.debug or {}).get("cluster_id"),
                        "center_distance_to_nearest": float(min_dist),
                        "overlap_to_nearest": float(overlap_with_nearest),
                        "inclusion_reason": "rejected_near_duplicate",
                    }
                )
                continue

            diversity_penalty = self.diversity_penalty * max(0.0, 1.0 - (min_dist / max(diversity_radius, 1e-6)))
            final_score = float(candidate.score - diversity_penalty)
            candidate.debug = candidate.debug or {}
            candidate.debug["survived_cluster_suppression"] = True
            candidate.debug["passed_weak_structure_rejection"] = True
            candidate.debug["cluster_radius"] = float(cluster_radius)
            candidate.debug["cluster_overlap_threshold"] = float(self.cluster_overlap_threshold)
            candidate.debug["diversity_radius"] = float(diversity_radius)
            candidate.debug["min_distance_to_selected"] = float(min_dist)
            candidate.debug["diversity_penalty"] = float(diversity_penalty)
            candidate.debug["final_score"] = final_score
            candidate.debug["nearest_selected_region_id"] = nearest.region_id if nearest else None
            if near_cluster:
                candidate.debug["cluster_id"] = (nearest.debug or {}).get("cluster_id")
                candidate.debug["inclusion_reason"] = "selected_structural_diversity"
            else:
                cluster_count += 1
                candidate.debug["cluster_id"] = f"c{cluster_count}"
                candidate.debug["inclusion_reason"] = "selected_primary_in_cluster"
            selected.append(candidate)

        return selected, rejected

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
            debug_payload = {"score": proposal.score, "reasons": proposal.reasons}
            if proposal.debug:
                debug_payload["selection"] = proposal.debug
            queries.append(
                {
                    "region_id": proposal.region_id,
                    "question_type": "layout_classification",
                    "question": "How should this ambiguous region be interpreted?",
                    "allowed_choices": ALLOWED_CHOICES,
                    "bbox": proposal.bbox,
                    "debug": debug_payload,
                }
            )
        return {"queries": queries, "rejected_candidates": getattr(self, "_last_rejected_candidates", [])}


def dump_queries_json(path: str, queries: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(queries, f, indent=2)
