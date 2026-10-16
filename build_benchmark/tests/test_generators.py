import json
import math
import random
import sys
import unittest
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import core_generators as cg
from build_dataset import load_specs, load_config, build_dataset_multi

SAMPLE_DIR = BASE_DIR / "sample_data"


def make_mock_obj(idx, cid, bbox, angle=0.0, cell="A1"):
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    pts = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    return {
        "obj_idx": idx,
        "class_id": cid,
        "bbox": [float(v) for v in bbox],
        "pts": pts,
        "minor_len_px": float(min(w, h)),
        "major_len_px": float(max(w, h)),
        "area_px2": float(w * h),
        "visible_fraction": 1.0,
        "difficult": False,
        "angle_deg": float(angle),
        "cell": cell,
        "center": ((x1 + x2) / 2.0, (y1 + y2) / 2.0),
    }


class TestCoreGenerators(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.specs = load_specs(BASE_DIR / "physical_sizes.json")
        cls.config = {
            "seed": 42,
            "calibration": {
                "id": "TEST-CALIB",
                "status": "calibrated",
                "source": "unit_test",
                "thresholds_px": {
                    "detection": 4.0,
                    "hard_negative": 4.0,
                    "orientation": 6.0,
                    "color_lf": 3.0,
                    "counting": 5.0,
                    "grounding": 5.0,
                },
            },
            "class_prior_approvals": {
                c: {"status": "reviewed", "source": "test_fixture"}
                for c in ("basketball_court", "baseball_diamond", "dam", "bridge")
            },
            "images": {},
        }

    def create_mock_scene(self, objects, *, img_w=300, img_h=300, gsd_m=0.25):
        meta = {
            "source_classes": list({o["class_id"] for o in objects}),
            "complete_classes": list({o["class_id"] for o in objects}),
            "annotation_review_source": "test_fixture",
            "absence_reviews": {
                "bridge": {"absent": True, "source": "test_fixture"},
                "dam": {"absent": True, "source": "test_fixture"},
            },
            "isolation_reviews": {
                str(o["obj_idx"]): {"source": "test_fixture", "nearest_same_class_tokens": 10}
                for o in objects
            },
            "color_reviews": {},
        }
        return cg.Scene(
            image_path="test_scene.png",
            img_w=img_w,
            img_h=img_h,
            gsd_m=gsd_m,
            scale=1.0,
            meta=meta,
            objects=objects,
        )

    def test_polygon_area_and_clip(self):
        pts = [(0, 0), (10, 0), (10, 10), (0, 10)]
        self.assertAlmostEqual(cg.polygon_area(pts), 100.0)

        rect = (0, 0, 5, 10)
        clipped = cg.clip_polygon(pts, rect)
        self.assertAlmostEqual(cg.polygon_area(clipped), 50.0)

    def test_cell_partitioning(self):
        scene = self.create_mock_scene([])
        self.assertEqual(list(cg.cell_box(scene, "A1")), [0.0, 0.0, 100.0, 100.0])
        self.assertEqual(list(cg.cell_box(scene, "B2")), [100.0, 100.0, 200.0, 200.0])
        self.assertEqual(list(cg.cell_box(scene, "C3")), [200.0, 200.0, 300.0, 300.0])

        obj_in_b2 = make_mock_obj(0, "airplane", [120, 120, 160, 160], cell="B2")
        self.assertTrue(cg.in_cell(scene, obj_in_b2))

    def test_orientation_angles_and_constraints(self):
        obj_diag1 = {"angle_deg": 45.0, "minor_len_px": 10.0, "major_len_px": 30.0}
        self.assertEqual(cg.orientation_answer(obj_diag1), cg.ORIENTATION_CHOICES[0])

        obj_diag2 = {"angle_deg": 135.0, "minor_len_px": 10.0, "major_len_px": 30.0}
        self.assertEqual(cg.orientation_answer(obj_diag2), cg.ORIENTATION_CHOICES[1])

        obj_square = {"angle_deg": 45.0, "minor_len_px": 28.0, "major_len_px": 30.0}
        self.assertGreater(obj_square["minor_len_px"] / obj_square["major_len_px"], 0.8)

        obj_boundary = {"angle_deg": 0.0, "minor_len_px": 10.0, "major_len_px": 30.0}
        with self.assertRaises(cg.SkipSample):
            cg.orientation_answer(obj_boundary)

    def test_box_iou(self):
        box1 = [0, 0, 10, 10]
        box2 = [0, 0, 10, 10]
        self.assertAlmostEqual(cg.box_iou(box1, box2), 1.0)

        box3 = [20, 20, 30, 30]
        self.assertAlmostEqual(cg.box_iou(box1, box3), 0.0)

        box4 = [0, 0, 10, 5]
        self.assertAlmostEqual(cg.box_iou(box1, box4), 0.5)

    def test_count_options(self):
        rng = random.Random(42)
        opts = cg.count_options(5, rng)
        self.assertEqual(len(opts), 3)
        self.assertIn("5", opts)
        self.assertEqual(len(set(opts)), 3)

    def test_grounding_distractors_in_same_cell(self):
        obj = make_mock_obj(0, "small_vehicle", [120, 120, 140, 140], cell="B2")
        scene = self.create_mock_scene([obj])
        rng = random.Random(42)
        options, truth = cg.grounding_options(scene, obj, rng)
        self.assertEqual(len(options), 3)
        self.assertIn(truth, options)

        b2_box = cg.cell_box(scene, "B2")
        for opt in options:
            box = json.loads(opt)
            self.assertGreaterEqual(box[0], b2_box[0])
            self.assertGreaterEqual(box[1], b2_box[1])
            self.assertLessEqual(box[2], b2_box[2])
            self.assertLessEqual(box[3], b2_box[3])

    def test_make_sample_structure_and_abstention(self):
        scene = self.create_mock_scene([])
        sample_pass = cg.make_sample(
            scene, "Q1", "airplane", [10, 10, 50, 50], 10.0, 5.0,
            "Có máy bay không?", ["Có", "Không"], "Có", "isolated"
        )
        self.assertTrue(sample_pass["answerable_by_sensor"])
        self.assertEqual(sample_pass["answer"], "Có")
        self.assertIn(cg.ABSTAIN, sample_pass["choices"])

        sample_fail = cg.make_sample(
            scene, "Q1", "airplane", [10, 10, 50, 50], 3.0, 5.0,
            "Có máy bay không?", ["Có", "Không"], "Có", "isolated"
        )
        self.assertFalse(sample_fail["answerable_by_sensor"])
        self.assertEqual(sample_fail["answer"], cg.ABSTAIN)

    def test_q1_generation(self):
        obj = make_mock_obj(0, "airplane", [50, 50, 90, 80], cell="A1")
        scene = self.create_mock_scene([obj])
        rejects = []
        samples = cg.generate_q1(scene, self.specs, self.config, rejects)
        self.assertEqual(len(samples), 1)
        q = samples[0]
        self.assertEqual(q["kind"], "Q1")
        self.assertIn("airplane", q["question"])
        self.assertEqual(q["answer"], "Có")

    def test_q2_hard_negative_generation(self):
        obj = make_mock_obj(0, "dam", [50, 50, 90, 80], cell="A1")
        scene = self.create_mock_scene([obj])
        rejects = []
        samples = cg.generate_q2(scene, self.specs, self.config, rejects)
        self.assertGreaterEqual(len(samples), 1)
        q = samples[0]
        self.assertEqual(q["kind"], "Q2")
        self.assertEqual(q["class_id"], "bridge")
        self.assertEqual(q["confused_present_class"], "dam")
        self.assertEqual(q["answer"], "Không")

    def test_q3_hf_generation(self):
        obj = make_mock_obj(0, "airplane", [120, 120, 140, 180], angle=45.0, cell="B2")
        scene = self.create_mock_scene([obj])
        rejects = []
        samples = cg.generate_q3_hf(scene, self.specs, self.config, rejects)
        self.assertEqual(len(samples), 1)
        q = samples[0]
        self.assertEqual(q["kind"], "Q3-HF")
        self.assertIn(cg.GRID_PREFIX, q["question"])
        self.assertIn(cg.ORIENTATION_CHOICES[0], q["choices"])

    def test_q3_lf_color_extraction_and_generation(self):
        from generate_q3_lf_automated import classify_pixel_hsv, build_q3_lf_sample
        # 1. Kiểm tra phân loại pixel HSV sang 6 màu chuẩn
        self.assertEqual(classify_pixel_hsv(0, 100, 200), "Màu đỏ")
        self.assertEqual(classify_pixel_hsv(100, 150, 200), "Màu xanh dương")
        self.assertEqual(classify_pixel_hsv(50, 150, 200), "Màu xanh lá cây")
        self.assertEqual(classify_pixel_hsv(25, 150, 200), "Màu vàng")
        self.assertEqual(classify_pixel_hsv(0, 10, 220), "Màu trắng")
        self.assertEqual(classify_pixel_hsv(0, 10, 80), "Màu đen / xám")
        self.assertIsNone(classify_pixel_hsv(0, 0, 20))  # Lọc bóng tối sâu

        # 2. Kiểm tra tạo câu hỏi Q3-LF giải được (answerable)
        obj = make_mock_obj(0, "airplane", [120, 120, 150, 150], cell="B2")
        scene = self.create_mock_scene([obj])
        q = build_q3_lf_sample(scene, obj, self.specs, self.config, "Màu trắng", answerable=True)
        self.assertEqual(q["kind"], "Q3-LF")
        self.assertEqual(q["answer"], "Màu trắng")
        self.assertTrue(q["answerable_by_sensor"])
        self.assertEqual(len(q["choices"]), 7)
        self.assertIn("Màu trắng", q["choices"])
        self.assertIn(cg.ABSTAIN, q["choices"])

        # 3. Kiểm tra tạo câu hỏi Q3-LF không giải được (unanswerable / Johnson abstention)
        obj_tiny = make_mock_obj(1, "airplane", [120, 120, 122, 122], cell="B2")
        q_unans = build_q3_lf_sample(scene, obj_tiny, self.specs, self.config, "Màu đỏ", answerable=False)
        self.assertEqual(q_unans["kind"], "Q3-LF")
        self.assertEqual(q_unans["answer"], cg.ABSTAIN)
        self.assertFalse(q_unans["answerable_by_sensor"])
        self.assertIn("unanswerable_reason", q_unans)

    def test_q4_counting_generation(self):
        objs = [
            make_mock_obj(i, "small_vehicle", [120 + i * 20, 120, 135 + i * 20, 140], cell="B2")
            for i in range(3)
        ]
        scene = self.create_mock_scene(objs)
        rejects = []
        samples = cg.generate_q4(scene, self.specs, self.config, rejects)
        self.assertEqual(len(samples), 1)
        q = samples[0]
        self.assertEqual(q["kind"], "Q4")
        self.assertEqual(q["answer"], "3")

    def test_q5_grounding_generation(self):
        objs = [
            make_mock_obj(i, "small_vehicle", [110 + i * 30, 120, 130 + i * 30, 140], cell="B2")
            for i in range(2)
        ]
        scene = self.create_mock_scene(objs)
        rejects = []
        samples = cg.generate_q5(scene, self.specs, self.config, rejects)
        self.assertEqual(len(samples), 1)
        q = samples[0]
        self.assertEqual(q["kind"], "Q5")
        self.assertEqual(q["target_box_xyxy"], [140.0, 120.0, 160.0, 140.0])

    def test_q6_twilight_zone_abstention(self):
        base_sample = {
            "kind": "Q3-HF",
            "p0_U_px": 10.0,
            "rho_px": 7.0,
            "answer": "Từ trên trái xuống dưới phải",
            "choices": ["Từ trên trái xuống dưới phải", "Từ trên phải xuống dưới trái", cg.ABSTAIN],
        }
        q6 = cg.generate_q6(base_sample)
        self.assertIsNotNone(q6)
        self.assertEqual(q6["kind"], "Q6")
        self.assertEqual(q6["answer"], cg.ABSTAIN)
        self.assertFalse(q6["answerable_by_sensor"])

        base_too_small = dict(base_sample, rho_px=3.0)
        self.assertIsNone(cg.generate_q6(base_too_small))

        base_too_clear = dict(base_sample, rho_px=12.0)
        self.assertIsNone(cg.generate_q6(base_too_clear))

    def test_load_specs_integrity(self):
        self.assertIn("airplane", self.specs)
        self.assertIn("small_vehicle", self.specs)
        self.assertIn("ship", self.specs)
        self.assertGreater(len(self.specs), 50)
        for cid, spec in self.specs.items():
            self.assertTrue(spec.get("display_name_vi"))
            self.assertGreater(spec.get("L_min_m", 0), 0)


if __name__ == "__main__":
    unittest.main()
