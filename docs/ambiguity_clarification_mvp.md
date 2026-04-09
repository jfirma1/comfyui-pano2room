# Ambiguity-triggered Clarification MVP

This MVP adds an early selective clarification layer before first `pano_distance_to_mesh()`.

## Flow
1. Compute initial depth edges.
2. Build ambiguity map from handcrafted signals (texture, edge instability, smoothness/edge disagreement, curvature in low-texture areas).
3. Propose top 1-3 ambiguous regions.
4. Export machine-consumable query JSON (`clarification_queries.json`).
5. Optionally load human constraints JSON (`--clarification_json`).
6. Inject conservative constraints into depth/edge/mesh-mask preprocessing.
7. Continue normal pipeline execution.

## Query/Constraint schema (MVP)
- `region_id`: stable proposal ID.
- `bbox`: `[x0, y0, x1, y1]`.
- `question_type`: currently `region_geometry_relation`.
- `allowed_choices`: `flat_wall`, `sharp_corner`, `same_surface`, `opening`, `unknown`.
- `mask_rle`: cropped run-length encoded mask for the bbox.

Constraint entries support:
- `region_id`
- `constraint_type`
- `confidence` in `[0, 1]`
- `bbox`
- optional `metadata`

## Conflict summary plumbing
`SupInfoPool.geo_check()` now stores `last_geo_check_stats` with:
- `total_pixels`
- `keep_ratio`
- `conflict_density`

This is intended for future update-stage ambiguity escalation.
