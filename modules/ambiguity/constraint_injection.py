from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch

from .constraint_types import ClarificationConstraint


def _clamp_bbox(bbox, h, w):
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(x1 + 1, min(w, x2))
    y2 = max(y1 + 1, min(h, y2))
    return x1, y1, x2, y2


def apply_constraints(
    init_depth: torch.Tensor,
    depth_edges: torch.Tensor,
    depth_edge_inpaint_mask: torch.Tensor,
    constraints: List[ClarificationConstraint],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
    depth = init_depth.detach().cpu().numpy().astype(np.float32)
    edges = (depth_edges.detach().cpu().numpy() > 0).astype(np.uint8) * 255
    inpaint_mask = depth_edge_inpaint_mask.detach().cpu().numpy().astype(np.uint8)

    h, w = depth.shape
    debug = np.zeros((h, w, 3), dtype=np.uint8)

    for item in constraints:
        x1, y1, x2, y2 = _clamp_bbox(item.bbox, h, w)
        roi = depth[y1:y2, x1:x2]

        if item.answer == "flat_wall":
            plane_value = float(np.median(roi))
            smoothed = cv2.GaussianBlur(roi, (0, 0), sigmaX=2.0)
            depth[y1:y2, x1:x2] = 0.65 * smoothed + 0.35 * plane_value
            edges[y1:y2, x1:x2] = cv2.erode(edges[y1:y2, x1:x2], np.ones((3, 3), np.uint8), iterations=1)
            debug[y1:y2, x1:x2, 1] = 180

        elif item.answer == "sharp_corner":
            cv2.rectangle(edges, (x1, y1), (x2 - 1, y2 - 1), 255, 2)
            inpaint_mask[y1:y2, x1:x2] = np.minimum(inpaint_mask[y1:y2, x1:x2], 1)
            debug[y1:y2, x1:x2, 2] = 220

        elif item.answer == "same_surface":
            local = depth[y1:y2, x1:x2]
            blur = cv2.GaussianBlur(local, (0, 0), sigmaX=3.0)
            depth[y1:y2, x1:x2] = 0.5 * local + 0.5 * blur
            debug[y1:y2, x1:x2, 0] = 200
            debug[y1:y2, x1:x2, 1] = 120

        elif item.answer == "opening":
            inpaint_mask[y1:y2, x1:x2] = 0
            edges[y1:y2, x1:x2] = cv2.dilate(edges[y1:y2, x1:x2], np.ones((3, 3), np.uint8), iterations=1)
            debug[y1:y2, x1:x2, 0] = 255

        else:  # unknown
            depth[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (0, 0), sigmaX=1.0)
            debug[y1:y2, x1:x2, :] = np.array([80, 80, 80], dtype=np.uint8)

        cv2.rectangle(debug, (x1, y1), (x2 - 1, y2 - 1), (255, 255, 255), 1)

    new_depth = torch.from_numpy(depth).to(init_depth.device, dtype=init_depth.dtype)
    new_edges = torch.from_numpy(edges).to(depth_edges.device, dtype=depth_edges.dtype)
    new_inpaint_mask = torch.from_numpy(inpaint_mask.astype(bool)).to(depth_edge_inpaint_mask.device)
    return new_depth, new_edges, new_inpaint_mask, debug
