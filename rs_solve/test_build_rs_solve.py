import tempfile
import unittest
from pathlib import Path

from PIL import Image

from build_rs_solve import (ABSTAIN, KINDS, candidates, choose_balanced,
                            clipped_area, demo_scenes, read_dota, rectangle,
                            validate)


class BuildRSSolveTests(unittest.TestCase):
    def test_polygon_grid_overlap_uses_area_not_center(self):
        box = rectangle(50, 50, 20, 10)
        self.assertAlmostEqual(clipped_area(box, (40, 45, 50, 55)), 100)
        self.assertAlmostEqual(clipped_area(box, (50, 45, 60, 55)), 100)
        self.assertEqual(clipped_area(box, (61, 45, 70, 55)), 0)

    def test_demo_balance_and_sensor_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            scenes = demo_scenes(Path(tmp))
            threshold = {"value_px": 6.0, "status": "illustrative_unvalidated"}
            sizes = {"baseball-diamond": 5.0, "dam": 5.0, "roundabout": 5.0,
                     "basketball-court": 5.0, "helicopter": 5.0}
            records = choose_balanced(scenes, threshold, sizes, 5)
            self.assertEqual(validate(records, 5)["total"], 35)
            self.assertEqual({r["kind"] for r in records}, set(KINDS))
            self.assertEqual({r["count_gt"] for r in records if r["kind"] == "Q4"}, {3})
            self.assertEqual(sum(r["rho_px"] < 6 for r in records if r["kind"] == "Q3-LF"), 2)
            self.assertTrue(all(r["answer"] == ABSTAIN for r in records if r["kind"] == "Q6"))

    def test_dota_metadata_and_hard_negative_review_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "images").mkdir()
            (root / "labelTxt").mkdir()
            Image.new("RGB", (512, 512)).save(root / "images" / "P0000.png")
            poly = rectangle(430, 100, 90, 58)
            coords = " ".join(str(c) for point in poly for c in point)
            (root / "labelTxt" / "P0000.txt").write_text(
                f"acquisition dates:None\nimagesource:GoogleEarth\ngsd:0.4\n"
                f"{coords} soccer-ball-field 0\n", encoding="utf-8")
            no_review = read_dota(root, {})
            threshold = {"value_px": 6.0, "status": "pilot_human_and_models_calibrated"}
            self.assertAlmostEqual(no_review[0].gsd, 0.4)
            self.assertTrue(Path(no_review[0].image_ref).is_absolute())
            self.assertFalse(candidates(no_review[0], threshold, {"baseball-diamond": 5})["Q2"])
            reviewed = read_dota(root, {"P0000": {"absent_classes": ["baseball-diamond"]}})
            q2 = candidates(reviewed[0], threshold, {"baseball-diamond": 5})["Q2"]
            self.assertEqual(q2[0]["answer"], "Không")
            self.assertEqual(q2[0]["rho_px"], 12.5)


if __name__ == "__main__":
    unittest.main()
