import json
import os
import tempfile
import unittest

from bom_builder import (
    bom_to_dicts,
    build_bom,
    export_bom_to_json,
    export_solids_to_json,
    filter_bom,
    solids_to_dicts,
    summarize_bom,
)
from cad_helpers import SolidRow


class TestBomFeatures(unittest.TestCase):
    def setUp(self):
        self.solids = [
            SolidRow(1, "profile", "Beam 1", 200.0, 30.0, 20.0, 120.0, 300.0, 1.2, "sig1"),
            SolidRow(2, "profile", "Beam 2", 200.0, 30.0, 20.0, 120.0, 300.0, 1.2, "sig1"),
            SolidRow(3, "plate", "Gusset A", 100.0, 50.0, 5.0, 25.0, 120.0, 0.25, "sig2"),
            SolidRow(4, "pin", "Pivot Pin", 80.0, 12.0, 12.0, 9.0, 40.0, 0.08, "sig3"),
        ]
        self.bom = build_bom(self.solids)

    def test_summarize_bom(self):
        summary = summarize_bom(self.bom)
        self.assertEqual(summary.total_parts, 4)
        self.assertEqual(summary.unique_items, 3)
        self.assertAlmostEqual(summary.total_weight_kg, 2.73, places=2)
        self.assertIn("profile", summary.class_distribution)
        self.assertIn("plate", summary.class_distribution)
        self.assertIn("pin", summary.class_distribution)
        self.assertEqual(summary.class_distribution["profile"]["count"], 2)

    def test_filter_bom_class(self):
        plates_only = filter_bom(self.bom, class_name="plate")
        self.assertEqual(len(plates_only), 1)
        self.assertEqual(plates_only[0].class_name, "plate")
        self.assertEqual(plates_only[0].pos, 1)

    def test_filter_bom_dimension_and_weight(self):
        # Filter by min length 150mm (only profile)
        long_parts = filter_bom(self.bom, min_length=150.0)
        self.assertEqual(len(long_parts), 1)
        self.assertEqual(long_parts[0].class_name, "profile")

        # Filter by max weight 0.5kg
        light_parts = filter_bom(self.bom, max_weight=0.5)
        self.assertEqual(len(light_parts), 2)  # plate and pin

    def test_json_and_dict_serialization(self):
        bom_dicts = bom_to_dicts(self.bom)
        self.assertEqual(len(bom_dicts), 3)
        self.assertIn("pos", bom_dicts[0])
        self.assertIn("class_name", bom_dicts[0])

        solid_dicts = solids_to_dicts(self.solids)
        self.assertEqual(len(solid_dicts), 4)
        self.assertIn("idx", solid_dicts[0])

        with tempfile.TemporaryDirectory() as tmpdir:
            bom_json = os.path.join(tmpdir, "bom.json")
            export_bom_to_json(self.bom, bom_json, include_summary=True)
            self.assertTrue(os.path.exists(bom_json))

            with open(bom_json, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.assertIn("bom", data)
            self.assertIn("summary", data)

            solids_json = os.path.join(tmpdir, "solids.json")
            export_solids_to_json(self.solids, solids_json)
            self.assertTrue(os.path.exists(solids_json))


if __name__ == "__main__":
    unittest.main()
