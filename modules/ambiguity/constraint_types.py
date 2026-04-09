from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class ConstraintLabel(str, Enum):
    FLAT_WALL = "flat_wall"
    SHARP_CORNER = "sharp_corner"
    SAME_SURFACE = "same_surface"
    OPENING = "opening"
    UNKNOWN = "unknown"


ALLOWED_CHOICES = [item.value for item in ConstraintLabel]


@dataclass
class ClarificationQuery:
    region_id: str
    question_type: str
    allowed_choices: List[str]
    bbox: Tuple[int, int, int, int]
    mask_rle: Optional[List[List[int]]] = None
    explanation: str = ""
    debug: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "region_id": self.region_id,
            "question_type": self.question_type,
            "allowed_choices": self.allowed_choices,
            "bbox": list(self.bbox),
            "mask_rle": self.mask_rle,
            "explanation": self.explanation,
            "debug": self.debug,
        }


@dataclass
class RegionConstraint:
    region_id: str
    constraint_type: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    region_mask: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "region_id": self.region_id,
            "constraint_type": self.constraint_type,
            "confidence": float(self.confidence),
            "bbox": list(self.bbox),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RegionConstraint":
        return cls(
            region_id=str(data["region_id"]),
            constraint_type=str(data.get("constraint_type", ConstraintLabel.UNKNOWN.value)),
            confidence=float(data.get("confidence", 0.5)),
            bbox=tuple(data.get("bbox", [0, 0, 0, 0])),
            metadata=data.get("metadata", {}),
        )
