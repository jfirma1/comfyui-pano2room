import unittest
from pathlib import Path
import importlib.util

ROOT = Path(__file__).resolve().parent


def _load_module(module_name, file_name):
    spec = importlib.util.spec_from_file_location(module_name, ROOT / file_name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


constraint_types = _load_module("constraint_types", "constraint_types.py")
answers_to_constraints = constraint_types.answers_to_constraints
parse_answers = constraint_types.parse_answers


class TestConstraintInterpretation(unittest.TestCase):
    def test_free_text_reflection_mapping(self):
        payload = {"answers": [{"region_id": "r1", "answer": "That's the reflection of the lamp"}]}
        parsed = parse_answers(payload)
        self.assertEqual(parsed[0].phenomenon_class, "reflection_or_highlight")
        self.assertEqual(parsed[0].geometry_class, "unknown")
        self.assertTrue(parsed[0].structural_policy["ignore_as_structure"])

    def test_legacy_alias_mapping(self):
        payload = {"answers": [{"region_id": "r1", "answer": "flat_wall"}]}
        parsed = parse_answers(payload)
        self.assertEqual(parsed[0].phenomenon_class, "continuous_surface")
        self.assertEqual(parsed[0].geometry_class, "flat_wall")

    def test_policy_applied_from_phenomenon(self):
        answers = parse_answers({"answers": [{"region_id": "r1", "answer": "desk edge, not the wall"}]})
        constraints = answers_to_constraints(answers, {"queries": [{"region_id": "r1", "bbox": [0, 0, 3, 3]}]})
        self.assertEqual(constraints[0].phenomenon_class, "object_edge_not_layout")
        self.assertTrue(constraints[0].structural_policy["downweight_edge_evidence"])
        self.assertFalse(constraints[0].structural_policy["allow_layout_boundary"])


if __name__ == "__main__":
    unittest.main()
