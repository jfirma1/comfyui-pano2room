"""
Utilities for fitting dominant architectural planes via iterative RANSAC.

Coordinate convention assumed throughout: **Y is up** (positive Y = ceiling).
"""

import numpy as np
import torch


def fit_dominant_planes(points_3d, n_planes=3, min_inlier_ratio=0.1,
                        ransac_threshold=0.05, max_trials=500):
    """
    Iteratively fit dominant architectural planes using RANSAC.

    After each fit the inlier points are removed from the pool so the next
    iteration finds a *different* plane.  Planes whose normals are inconsistent
    with floors, ceilings, or walls are skipped (see ``_is_architectural_surface``).

    Args:
        points_3d:         [N, 3] CPU tensor or numpy array of world 3-D points.
        n_planes:          Maximum number of planes to return (≤ 3 is typical).
        min_inlier_ratio:  A fitted plane is kept only when its inlier count is at
                           least ``min_inlier_ratio * N`` of the *original* N.
        ransac_threshold:  Point-to-plane distance (scene units) used to decide
                           inliers during RANSAC.
        max_trials:        RANSAC iterations per plane candidate.

    Returns:
        List of ``(plane_normal, plane_d, inlier_mask)`` CPU tuples:
            plane_normal – [3] float32 tensor, unit-length normal
            plane_d      – scalar float32 tensor  s.t. ``dot(n, p) ≈ d``
            inlier_mask  – [N] bool tensor  (indexed into the *original* N points)
    """
    if isinstance(points_3d, torch.Tensor):
        pts = points_3d.detach().cpu().numpy().astype(np.float64)
    else:
        pts = np.asarray(points_3d, dtype=np.float64)

    N = len(pts)
    if N < 3:
        return []

    min_inliers = max(3, int(N * min_inlier_ratio))
    remaining = np.ones(N, dtype=bool)
    planes = []
    rng = np.random.default_rng(0)

    for _ in range(n_planes):
        idx_rem = np.where(remaining)[0]
        if len(idx_rem) < min_inliers:
            break

        normal, d, local_inliers = _ransac_plane(pts[idx_rem], ransac_threshold, max_trials, rng)

        if normal is None or int(local_inliers.sum()) < min_inliers:
            break

        # Always evict inliers so the next iteration finds a different surface.
        remaining[idx_rem[local_inliers]] = False

        if not _is_architectural_surface(normal):
            continue

        global_inliers = np.zeros(N, dtype=bool)
        global_inliers[idx_rem[local_inliers]] = True

        planes.append((
            torch.from_numpy(normal.astype(np.float32)),
            torch.tensor(float(d), dtype=torch.float32),
            torch.from_numpy(global_inliers),
        ))

    return planes


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ransac_plane(pts, threshold, max_trials, rng):
    """Fit one plane via RANSAC. Returns *(unit_normal, d, inlier_mask)*."""
    N = len(pts)
    best_count = 0
    best_normal = best_d = best_inliers = None

    for _ in range(max_trials):
        idx = rng.choice(N, 3, replace=False)
        v1 = pts[idx[1]] - pts[idx[0]]
        v2 = pts[idx[2]] - pts[idx[0]]
        n = np.cross(v1, v2)
        nlen = np.linalg.norm(n)
        if nlen < 1e-8:
            continue
        n = n / nlen
        d = float(n @ pts[idx[0]])

        dists = np.abs(pts @ n - d)
        inliers = dists < threshold
        count = int(inliers.sum())
        if count > best_count:
            best_count = count
            best_normal = n.copy()
            best_d = d
            best_inliers = inliers.copy()

    if best_normal is None:
        return None, None, None

    # Least-squares refinement over all inliers via SVD.
    inlier_pts = pts[best_inliers]
    if len(inlier_pts) >= 3:
        centroid = inlier_pts.mean(0)
        _, _, Vt = np.linalg.svd(inlier_pts - centroid, full_matrices=False)
        n = Vt[-1]          # smallest singular value → plane normal
        d = float(n @ centroid)
        if d < 0:           # canonical orientation: d ≥ 0
            n, d = -n, -d
        best_inliers = np.abs(pts @ n - d) < threshold
        best_normal, best_d = n, d

    return best_normal, float(best_d), best_inliers


def _is_architectural_surface(normal, threshold_deg=20.0):
    """
    Return *True* if *normal* is consistent with a floor/ceiling or wall.

    Y-up convention:
    - Floor / ceiling: ``|n · ŷ| > cos(20°)`` — normal nearly vertical.
    - Wall           : ``|n · ŷ| < sin(20°)`` — normal nearly horizontal.

    Slanted surfaces (furniture, countertops) fall between the two bands
    and return *False*.
    """
    cos_t = float(np.cos(np.radians(threshold_deg)))
    sin_t = float(np.sin(np.radians(threshold_deg)))
    y = abs(float(normal[1]))
    return y > cos_t or y < sin_t
