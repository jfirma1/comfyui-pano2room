import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

ALLOWED_CHOICES = ["flat_wall", "sharp_corner", "same_surface", "opening", "unknown"]


@dataclass
class RegionProposal:
    region_id: str
    bbox: List[int]
    score: float
    reasons: Dict[str, float]


@dataclass
class ClarificationAnswer:
    region_id: str
    answer: str


@dataclass
class ClarificationConstraint:
    region_id: str
    answer: str
    bbox: List[int]


def load_answer_payload(answer_json: Optional[str] = None, answer_json_path: Optional[str] = None) -> Dict:
    if answer_json_path:
        payload = Path(answer_json_path).read_text(encoding="utf-8")
        return json.loads(payload)
    if answer_json and answer_json.strip():
        return json.loads(answer_json)
    return {"answers": []}


def parse_answers(payload: Dict) -> List[ClarificationAnswer]:
    answers = []
    for item in payload.get("answers", []):
        region_id = str(item.get("region_id", "")).strip()
        answer = str(item.get("answer", "unknown")).strip()
        if not region_id:
            continue
        if answer not in ALLOWED_CHOICES:
            answer = "unknown"
        answers.append(ClarificationAnswer(region_id=region_id, answer=answer))
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
            ClarificationConstraint(region_id=answer.region_id, answer=answer.answer, bbox=[int(v) for v in bbox])
        )
    return constraints
