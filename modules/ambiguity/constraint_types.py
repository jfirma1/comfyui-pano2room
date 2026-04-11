import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

LEGACY_ALLOWED_CHOICES = ["flat_wall", "sharp_corner", "same_surface", "opening", "unknown"]
PHENOMENON_CHOICES = [
    "reflection_or_highlight",
    "mirror_surface",
    "glass_or_window",
    "object_edge_not_layout",
    "real_room_boundary",
    "continuous_surface",
    "opening_or_passage",
    "unknown",
]

GEOMETRY_CHOICES = ["flat_wall", "corner", "opening", "continuous_surface", "object_boundary", "unknown"]
ALLOWED_CHOICES = LEGACY_ALLOWED_CHOICES + [c for c in PHENOMENON_CHOICES if c not in LEGACY_ALLOWED_CHOICES]


@dataclass
class RegionProposal:
    region_id: str
    bbox: List[int]
    score: float
    reasons: Dict[str, float]
    debug: Optional[Dict] = None


@dataclass
class ClarificationAnswer:
    region_id: str
    answer: str
    raw_text: str
    geometry_class: str
    phenomenon_class: str
    structural_policy: Dict[str, bool]
    confidence: float
    parse_method: str
    fallback_reason: Optional[str] = None


@dataclass
class ClarificationConstraint:
    region_id: str
    answer: str
    bbox: List[int]
    geometry_class: str
    phenomenon_class: str
    structural_policy: Dict[str, bool]
    raw_text: str
    confidence: float
    parse_method: str
    fallback_reason: Optional[str] = None


POLICY_BY_PHENOMENON = {
    "reflection_or_highlight": {
        "ignore_as_structure": True,
        "downweight_edge_evidence": True,
        "preserve_planar_assumption": True,
        "avoid_forcing_depth_discontinuity": True,
        "allow_layout_boundary": False,
    },
    "mirror_surface": {
        "ignore_as_structure": True,
        "downweight_edge_evidence": True,
        "preserve_planar_assumption": True,
        "avoid_forcing_depth_discontinuity": True,
        "allow_layout_boundary": False,
    },
    "glass_or_window": {
        "ignore_as_structure": False,
        "downweight_edge_evidence": True,
        "preserve_planar_assumption": False,
        "avoid_forcing_depth_discontinuity": True,
        "allow_layout_boundary": True,
    },
    "object_edge_not_layout": {
        "ignore_as_structure": False,
        "downweight_edge_evidence": True,
        "preserve_planar_assumption": False,
        "avoid_forcing_depth_discontinuity": False,
        "allow_layout_boundary": False,
    },
    "real_room_boundary": {
        "ignore_as_structure": False,
        "downweight_edge_evidence": False,
        "preserve_planar_assumption": False,
        "avoid_forcing_depth_discontinuity": False,
        "allow_layout_boundary": True,
    },
    "continuous_surface": {
        "ignore_as_structure": False,
        "downweight_edge_evidence": True,
        "preserve_planar_assumption": True,
        "avoid_forcing_depth_discontinuity": True,
        "allow_layout_boundary": False,
    },
    "opening_or_passage": {
        "ignore_as_structure": False,
        "downweight_edge_evidence": False,
        "preserve_planar_assumption": False,
        "avoid_forcing_depth_discontinuity": False,
        "allow_layout_boundary": True,
    },
    "unknown": {
        "ignore_as_structure": False,
        "downweight_edge_evidence": False,
        "preserve_planar_assumption": False,
        "avoid_forcing_depth_discontinuity": False,
        "allow_layout_boundary": False,
    },
}

LEGACY_TO_INTERPRETATION = {
    "flat_wall": ("flat_wall", "continuous_surface"),
    "sharp_corner": ("corner", "real_room_boundary"),
    "same_surface": ("continuous_surface", "continuous_surface"),
    "opening": ("opening", "opening_or_passage"),
    "unknown": ("unknown", "unknown"),
}

PHRASE_TO_PHENOMENON = [
    (("reflection", "glare", "highlight", "specular"), "reflection_or_highlight"),
    (("mirror", "mirrored"), "mirror_surface"),
    (("glass", "window", "pane"), "glass_or_window"),
    (("desk edge", "lamp edge", "chair edge", "object edge", "not the wall"), "object_edge_not_layout"),
    (("same wall", "same surface", "wall continues", "continuous"), "continuous_surface"),
    (("opening", "doorway", "hallway", "passage"), "opening_or_passage"),
    (("real boundary", "room boundary", "true corner", "real room boundary"), "real_room_boundary"),
]


def load_answer_payload(answer_json: Optional[str] = None, answer_json_path: Optional[str] = None) -> Dict:
    if answer_json_path:
        payload = Path(answer_json_path).read_text(encoding="utf-8")
        return json.loads(payload)
    if answer_json and answer_json.strip():
        return json.loads(answer_json)
    return {"answers": []}

def _classify_text(raw_text: str) -> Dict[str, Any]:
    text = (raw_text or "").strip().lower()
    if not text:
        return {
            "geometry_class": "unknown",
            "phenomenon_class": "unknown",
            "confidence": 0.0,
            "parse_method": "fallback",
            "fallback_reason": "empty_answer",
        }

    for phrases, phenomenon in PHRASE_TO_PHENOMENON:
        if any(phrase in text for phrase in phrases):
            geometry = "unknown"
            if phenomenon == "opening_or_passage":
                geometry = "opening"
            elif phenomenon == "object_edge_not_layout":
                geometry = "object_boundary"
            elif phenomenon == "continuous_surface":
                geometry = "continuous_surface"
            elif phenomenon == "real_room_boundary":
                geometry = "corner"
            return {
                "geometry_class": geometry,
                "phenomenon_class": phenomenon,
                "confidence": 0.9,
                "parse_method": "rule_based",
                "fallback_reason": None,
            }

    return {
        "geometry_class": "unknown",
        "phenomenon_class": "unknown",
        "confidence": 0.2,
        "parse_method": "fallback",
        "fallback_reason": "no_phrase_match",
    }


def _normalize_item(item: Dict[str, Any]) -> ClarificationAnswer:
    region_id = str(item.get("region_id", "")).strip()
    answer = str(item.get("answer", "unknown")).strip()
    source_text = str(item.get("source_text", "")).strip()

    geometry_class = str(item.get("geometry_class", "")).strip()
    phenomenon_class = str(item.get("phenomenon_class", "")).strip()
    parse_method = "explicit"
    fallback_reason = None
    confidence = float(item.get("confidence", 1.0))

    if answer in LEGACY_TO_INTERPRETATION and not (geometry_class or phenomenon_class):
        geometry_class, phenomenon_class = LEGACY_TO_INTERPRETATION[answer]
        parse_method = "legacy_alias"
        confidence = min(confidence, 0.95)
    elif answer in PHENOMENON_CHOICES and not phenomenon_class:
        phenomenon_class = answer
        parse_method = "enum_direct"
    elif answer in GEOMETRY_CHOICES and not geometry_class:
        geometry_class = answer
        parse_method = "enum_direct"

    raw_text = source_text or answer
    if not (geometry_class or phenomenon_class):
        parsed = _classify_text(raw_text)
        geometry_class = parsed["geometry_class"]
        phenomenon_class = parsed["phenomenon_class"]
        parse_method = parsed["parse_method"]
        fallback_reason = parsed["fallback_reason"]
        confidence = parsed["confidence"]

    if geometry_class not in GEOMETRY_CHOICES:
        geometry_class = "unknown"
    if phenomenon_class not in PHENOMENON_CHOICES:
        phenomenon_class = "unknown"
    policy = POLICY_BY_PHENOMENON.get(phenomenon_class, POLICY_BY_PHENOMENON["unknown"]).copy()
    return ClarificationAnswer(
        region_id=region_id,
        answer=answer or "unknown",
        raw_text=raw_text,
        geometry_class=geometry_class,
        phenomenon_class=phenomenon_class,
        structural_policy=policy,
        confidence=float(confidence),
        parse_method=parse_method,
        fallback_reason=fallback_reason,
    )


def parse_answers(payload: Dict) -> List[ClarificationAnswer]:
    answers = []
    for item in payload.get("answers", []):
        normalized = _normalize_item(item)
        region_id = normalized.region_id
        if not region_id:
            continue
        answers.append(normalized)
    return answers


def answers_to_constraints(answers: List[ClarificationAnswer], queries_payload: Dict) -> List[ClarificationConstraint]:
    region_to_bbox = {
        q.get("region_id"): q.get("bbox")
        for q in queries_payload.get("queries", [])
        if q.get("region_id")
    }
    constraints = []
    for answer in answers:
        bbox = region_to_bbox.get(answer.region_id)
        if bbox is None:
            continue
        constraints.append(
            ClarificationConstraint(
                region_id=answer.region_id,
                answer=answer.answer,
                bbox=[int(v) for v in bbox],
                geometry_class=answer.geometry_class,
                phenomenon_class=answer.phenomenon_class,
                structural_policy=answer.structural_policy,
                raw_text=answer.raw_text,
                confidence=answer.confidence,
                parse_method=answer.parse_method,
                fallback_reason=answer.fallback_reason,
            )
        )
    return constraints
