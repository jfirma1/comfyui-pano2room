# Two-stage ambiguity clarification workflow (MVP)

## Workflow

### Stage 1: Probe (`prepare_ambiguity_probe`)
1. Load panorama and runtime settings.
2. Predict initial depth and compute depth edges.
3. Compute ambiguity heatmap from lightweight signals:
   - low texture energy,
   - edge instability under depth perturbation,
   - smoothness/edge disagreement,
   - depth curvature in low-texture areas.
4. Select top 1–3 ambiguous regions and generate query JSON.
5. Save intermediate state and debug artifacts.
6. Stop before the first `pano_distance_to_mesh()` call.

### Stage 2: Resume (`resume_from_clarification`)
1. Load `intermediate_state.pt`.
2. Load answers from JSON string or JSON file path.
3. Map answers to constraints by `region_id` and query bbox.
4. Apply conservative, localized constraints before meshing.
5. Resume original flow (initial mesh → greedy inpainting → GS training/eval).

## Query schema

```json
{
  "queries": [
    {
      "region_id": "r1",
      "question_type": "layout_classification",
      "question": "How should this ambiguous region be interpreted?",
      "allowed_choices": ["flat_wall", "sharp_corner", "same_surface", "opening", "unknown"],
      "bbox": [120, 48, 360, 290],
      "debug": {
        "score": 0.81,
        "reasons": {
          "low_texture": 0.77,
          "edge_instability": 0.45,
          "smoothness_edge_disagreement": 0.51,
          "curvature_in_low_texture": 0.63
        }
      }
    }
  ]
}
```

## Answer schema

```json
{
  "answers": [
    {"region_id": "r1", "answer": "flat_wall"},
    {"region_id": "r2", "answer": "same_surface"}
  ]
}
```

## Saved state contents

`intermediate_state.pt` stores:
- `pano_rgb`
- `init_depth`
- `depth_edges`
- `depth_edge_inpaint_mask`
- `pano_pose`
- `poses`
- `runtime_settings`
- `scene_depth_max`

## Saved artifacts

- `ambiguity_heatmap.png`
- `ambiguity_overlay.png`
- `queries.json`
- `clarification_answers.json` (if provided)
- `applied_constraints_debug.png` (if constraints applied)
- `state_metadata.json`
