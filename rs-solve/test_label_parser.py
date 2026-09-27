"""Kiểm thử hồi quy cho CODE THẬT, không chép lại parser/class map.

Chạy: python -m unittest discover -s rs-solve -p test_label_parser.py -v
Ảnh/nhãn kiểm thử nằm trong thư mục tạm và bị dọn sau test. GSD/ngưỡng tại
đây chỉ là số liệu thử phần mềm, tuyệt đối không dùng làm hiệu chỉnh nghiên cứu.
"""
import json
import math
from pathlib import Path
import random
import tempfile
import unittest
from PIL import Image
import build_samples as bs


class GeneratorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.specs = bs.load_physical_sizes()
        self.config = {
            "seed": 42,
            "calibration": {"id": "TEST-ONLY", "status": "calibrated", "source": "unit-test fixture, not research",
                "thresholds_px": {"detection": 2.0, "hard_negative": 4.0, "orientation": 6.0,
                                  "color_lf": 2.0, "counting": 5.0, "grounding": 5.0}},
            "class_prior_approvals": {c: {"status": "reviewed", "source": "test only"}
                                      for c in ("basketball_court", "baseball_diamond", "dam")},
            "images": {},
        }

    @staticmethod
    def rect(cx, cy, width, height, angle=0):
        theta = math.radians(angle)
        return [(cx + x*math.cos(theta)-y*math.sin(theta),
                 cy + x*math.sin(theta)+y*math.cos(theta))
                for x, y in ((-width/2, -height/2), (width/2, -height/2),
                             (width/2, height/2), (-width/2, height/2))]

    def scene(self, items, *, width=300, height=300, name="scene", scale=1.0):
        Image.new("RGB", (width, height), "white").save(self.root / f"{name}.png")
        lines = [str(cid) + " " + " ".join(str(v) for x, y in pts for v in (x/width, y/height))
                 for cid, pts in items]
        (self.root / f"{name}.txt").write_text("\n".join(lines), encoding="utf-8")
        meta = {"label_path": f"{name}.txt", "label_format": "yolo_obb", "gsd_m": .25,
                "gsd_source": "test metadata", "native_to_image_scale": scale,
                "transform_type": "identity" if scale == 1 else "uniform_resize",
                "resampling_method": "nearest (fixture only)",
                "pixel_scale_source": "test identity/resize", "source_classes": list(bs.DOTA_CLASS_MAP.values()),
                "complete_classes": list(bs.DOTA_CLASS_MAP.values()), "annotation_review_source": "test complete annotation",
                "isolation_reviews": {str(i): {"source": "test measurement", "reference_view": "test-view",
                    "nearest_same_class_tokens": 10} for i in range(len(items))}}
        self.config["images"][f"{name}.png"] = meta
        return bs.load_scene(self.root, f"{name}.png", meta)

    def all_kinds_scene(self):
        scene = self.scene([
            (10, self.rect(20, 25, 10, 20)),
            (10, self.rect(50, 25, 20, 30)),
            (10, self.rect(80, 25, 30, 35)),
            (14, self.rect(150, 150, 40, 60)),
            (0, self.rect(250, 150, 40, 15, 45)),
            (4, self.rect(150, 50, 40, 60)),
            (11, self.rect(250, 250, 12, 4, 45)),
        ])
        scene.meta["color_reviews"] = {"3": {"status": "verified", "value": "Màu đỏ", "source": "test fine image review"}}
        return scene

    def test_reads_real_image_dimensions_not_1024(self):
        scene = self.scene([(10, self.rect(160, 80, 32, 16))], width=320, height=160)
        self.assertAlmostEqual(scene.objects[0]["minor_len_px"], 16)
        self.assertEqual(scene.objects[0]["center"], (160, 80))

    def test_yolo_class_map_uses_production_order(self):
        scene = self.scene([(10, self.rect(50, 50, 10, 20)), (5, self.rect(150, 50, 20, 30))])
        self.assertEqual([o["class_id"] for o in scene.objects], ["small_vehicle", "basketball_court"])

    def test_dota_native_format_and_metadata(self):
        path = self.root / "dota.txt"
        path.write_text("imagesource:GoogleEarth\ngsd:0.3\n10 10 50 10 50 30 10 30 plane 0\n", encoding="utf-8")
        self.assertEqual(bs.read_dota_gsd(path), .3)
        obj = bs.parse_dota_label(path, 200, 100, label_format="dota")[0]
        self.assertEqual(obj["obj_idx"], 2)
        self.assertEqual(obj["class_id"], "airplane")
        self.assertAlmostEqual(obj["minor_len_px"], 20)

    def test_missing_gsd_has_no_fallback(self):
        path = self.root / "metadata.txt"
        path.write_text("gsd:None\n", encoding="utf-8")
        self.assertIsNone(bs.read_dota_gsd(path))
        with self.assertRaises(bs.InvalidInput):
            bs.resolve_gsd(self.root, {"gsd_metadata_path": "metadata.txt"})

    def test_conflicting_gsd_is_rejected(self):
        (self.root / "metadata.txt").write_text("gsd:0.3", encoding="utf-8")
        with self.assertRaises(bs.InvalidInput):
            bs.resolve_gsd(self.root, {"gsd_metadata_path": "metadata.txt", "gsd_m": .25, "gsd_source": "fixture"})

    def test_resize_does_not_inflate_sensor_footprint(self):
        scene = self.scene([(10, self.rect(50, 50, 20, 40))], scale=2)
        q = bs.generate_q1(scene, self.specs, self.config, [])[0]
        self.assertAlmostEqual(q["rho_px"], 10)
        self.assertAlmostEqual(q["L_m"], 2.5)
        self.assertEqual(q["gsd_m"], .25)

    def test_unknown_resize_history_is_rejected(self):
        scene = self.scene([(10, self.rect(50, 50, 20, 40))])
        scene.meta["native_to_image_scale"] = None
        with self.assertRaises(bs.InvalidInput):
            bs.load_scene(self.root, "scene.png", scene.meta)

    def test_downsample_does_not_restore_lost_pixels(self):
        scene = self.scene([(10, self.rect(50, 50, 20, 40))], scale=.5)
        q = bs.generate_q1(scene, self.specs, self.config, [])[0]
        self.assertAlmostEqual(q["rho_px"], 20)
        self.assertAlmostEqual(q["gsd_m"], .5)
        self.assertAlmostEqual(q["L_m"], 10)

    def test_resampling_method_required_when_resized(self):
        scene = self.scene([(10, self.rect(50, 50, 20, 40))], scale=2)
        scene.meta["resampling_method"] = None
        with self.assertRaises(bs.InvalidInput):
            bs.load_scene(self.root, "scene.png", scene.meta)

    def test_bad_polygon_and_nan_are_rejected(self):
        for text in ("0 0 0 1 1 1 0 0 1", "0 nan 0 1 0 1 1 0 1"):
            path = self.root / "bad.txt"
            path.write_text(text)
            with self.subTest(text=text), self.assertRaises(bs.InvalidInput):
                bs.parse_dota_label(path, 100, 100)

    def test_positive_y_direction_is_down_right(self):
        scene = self.scene([(10, self.rect(50, 50, 40, 10, 45)), (0, self.rect(150, 50, 40, 10, 135))])
        self.assertEqual(bs.orientation_answer(scene.objects[0]), bs.ORIENTATION_CHOICES[0])
        self.assertEqual(bs.orientation_answer(scene.objects[1]), bs.ORIENTATION_CHOICES[1])

    def test_square_and_boundary_orientation_are_rejected(self):
        scene = self.scene([(10, self.rect(50, 50, 40, 36, 45)), (0, self.rect(150, 50, 40, 10, 0))])
        for obj in scene.objects:
            with self.subTest(obj=obj["obj_idx"]), self.assertRaises(bs.SkipSample):
                bs.orientation_answer(obj)

    def test_no_calibration_no_sample(self):
        scene = self.scene([(10, self.rect(50, 50, 10, 20))])
        self.config["calibration"]["status"] = "pending"
        rejects = []
        self.assertEqual(bs.generate_q1(scene, self.specs, self.config, rejects), [])
        self.assertTrue(any("calibration" in r["reason"] for r in rejects))

    def test_null_negative_or_boolean_threshold_is_rejected(self):
        for value in (None, -1, True, float("nan")):
            self.config["calibration"]["thresholds_px"]["orientation"] = value
            with self.subTest(value=value), self.assertRaises(bs.SkipSample):
                bs.threshold(self.config, "Q3-HF")

    def test_sensor_flag_and_answer_agree_below_threshold(self):
        scene = self.scene([(0, self.rect(50, 50, 20, 4, 45))])
        q = bs.generate_q3_hf(scene, self.specs, self.config, [])[0]
        self.assertFalse(q["answerable_by_sensor"])
        self.assertEqual(q["answer"], bs.ABSTAIN)

    def test_threshold_comparison_does_not_round_up(self):
        scene = self.scene([(10, self.rect(50, 50, 20, 40))])
        q = bs.make_sample(scene, "Q1", "small_vehicle", [0, 0, 20, 40], 11.999, 12,
                           "test", ["Có", "Không"], "Có", "isolated")
        self.assertLess(q["rho_px"], q["p0_U_px"])
        self.assertFalse(q["answerable_by_sensor"])

    def test_q2_does_not_invent_absence_fallback(self):
        scene = self.scene([(5, self.rect(50, 50, 10, 20)), (5, self.rect(70, 50, 10, 20))])
        self.assertEqual(bs.generate_q2(scene, self.specs, self.config, []), [])

    def test_q2_unknown_source_class_needs_review(self):
        scene = self.scene([(8, self.rect(50, 50, 10, 20))])
        self.assertEqual(bs.generate_q2(scene, self.specs, self.config, []), [])
        scene.meta["absence_reviews"] = {"dam": {"absent": True, "source": "test manual review"}}
        self.assertEqual(bs.generate_q2(scene, self.specs, self.config, [])[0]["class_id"], "dam")

    def test_q2_below_threshold_abstains(self):
        scene = self.scene([(4, self.rect(50, 50, 20, 40))])
        self.config["calibration"]["thresholds_px"]["hard_negative"] = 100
        q = bs.generate_q2(scene, self.specs, self.config, [])[0]
        self.assertEqual(q["rho_px"], self.specs["basketball_court"]["L_min_m"] / scene.gsd_m)
        self.assertEqual(q["answer"], bs.ABSTAIN)
        self.assertIsNone(q["target_box_xyxy"])

    def test_q2_unreviewed_prior_is_rejected(self):
        scene = self.scene([(4, self.rect(50, 50, 20, 40))])
        specs_unreviewed = {k: dict(v) for k, v in self.specs.items()}
        specs_unreviewed["basketball_court"]["standard_source"] = ""
        self.config["class_prior_approvals"] = {}
        self.assertEqual(bs.generate_q2(scene, specs_unreviewed, self.config, []), [])

    def test_color_is_reviewed_not_inferred_from_class(self):
        scene = self.scene([(14, self.rect(50, 50, 20, 40))])
        self.assertEqual(bs.generate_q3_lf(scene, self.specs, self.config, []), [])
        scene.meta["color_reviews"] = {"0": {"status": "verified", "value": "Màu đỏ", "source": "test reference image"}}
        q = bs.generate_q3_lf(scene, self.specs, self.config, [])[0]
        self.assertEqual(q["answer"], "Màu đỏ")
        self.assertEqual(len(q["choices"]), 7)

    def test_q4_uses_median_individual_footprints(self):
        scene = self.all_kinds_scene()
        q = bs.generate_q4(scene, self.specs, self.config, [])[0]
        self.assertEqual(q["answer"], "3")
        self.assertAlmostEqual(q["rho_px"], 20)
        self.assertAlmostEqual(q["L_m"], 5)

    def test_cell_filter_uses_area_not_just_center(self):
        scene = self.scene([(10, self.rect(101, 101, 100, 100))])
        obj = scene.objects[0]
        self.assertEqual(obj["cell"], "B2")
        self.assertFalse(bs.in_cell(scene, obj))  # chỉ khoảng 26% nằm trong B2
        self.assertEqual(bs.grouped_objects(scene), [])

    def test_count_distractors_do_not_always_reveal_median(self):
        ranks, positions = set(), set()
        for seed in range(30):
            options = bs.count_options(3, random.Random(seed))
            ranks.add(sorted(map(int, options)).index(3))
            positions.add(options.index("3"))
            self.assertEqual(len(set(options)), 3)
        self.assertEqual(ranks, {0, 1, 2})
        self.assertEqual(positions, {0, 1, 2})

    def test_q5_two_objects_selects_second(self):
        scene = self.scene([(5, self.rect(25, 25, 10, 20)), (5, self.rect(70, 25, 10, 20))])
        q = bs.generate_q5(scene, self.specs, self.config, [])[0]
        self.assertEqual(q["target_box_xyxy"], [65, 15, 75, 35])
        self.assertEqual(json.loads(q["answer"]), q["target_box_xyxy"])

    def test_q5_distractors_all_belong_to_asked_cell(self):
        scene = self.all_kinds_scene()
        q = bs.generate_q5(scene, self.specs, self.config, [])[0]
        box = bs.cell_box(scene, q["target_cell"])
        for text in q["choices"][:-1]:
            x1, y1, x2, y2 = json.loads(text)
            poly = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
            self.assertGreaterEqual(bs.polygon_area(bs.clip_polygon(poly, box)) / bs.polygon_area(poly), .5)
            self.assertTrue(all(isinstance(v, int) for v in (x1, y1, x2, y2)))

    def test_q5_correct_box_not_fixed_coordinate_rank(self):
        scene = self.scene([(5, self.rect(25, 25, 10, 20)), (5, self.rect(55, 45, 10, 20))])
        ranks = set()
        for seed in range(30):
            self.config["seed"] = seed
            q = bs.generate_q5(scene, self.specs, self.config, [])[0]
            ordered = sorted(q["choices"][:-1], key=lambda s: json.loads(s)[0])
            ranks.add(ordered.index(q["answer"]))
        self.assertGreater(len(ranks), 1)

    def test_q4_q5_insufficient_objects_skip_without_crash(self):
        scene = self.scene([(10, self.rect(50, 50, 10, 20))])
        self.assertEqual(bs.generate_q4(scene, self.specs, self.config, []), [])
        self.assertEqual(bs.generate_q5(scene, self.specs, self.config, []), [])

    def test_isolation_is_not_assumed(self):
        scene = self.scene([(10, self.rect(50, 50, 10, 20)), (10, self.rect(150, 50, 10, 20))])
        scene.meta["isolation_reviews"] = {}
        self.assertEqual(bs.generate_q1(scene, self.specs, self.config, []), [])

    def test_q6_band_boundaries_and_no_sedan_fallback(self):
        scene = self.scene([(0, self.rect(50, 50, 20, 4, 45))])
        for rho, selected in ((2.99, False), (3.0, True), (5.99, True), (6.0, False), (7.0, False)):
            base = bs.make_sample(scene, "Q3-HF", "airplane", [1, 1, 20, 10], rho, 6,
                                  "test", list(bs.ORIENTATION_CHOICES), bs.ORIENTATION_CHOICES[0], "isolated", cell="A1")
            result = bs.generate_q6(base)
            self.assertEqual(result is not None, selected)
            if result:
                self.assertEqual(result["answer"], bs.ABSTAIN)
                self.assertFalse(result["answerable_by_sensor"])
                self.assertIn(f"{rho:.6g}", result["unanswerable_reason"])
            elif rho < 3:
                self.assertFalse(base["answerable_by_sensor"])
                self.assertEqual(base["answer"], bs.ABSTAIN)

    def test_full_build_keeps_old_json_fields_for_every_kind(self):
        self.all_kinds_scene()
        samples, report = bs.build_dataset(self.root, self.config, self.specs)
        self.assertTrue(report["complete"], report)
        self.assertEqual(len(samples), 7)
        legacy = bs.load_json(bs.BASE_DIR / "mau_7_questions.json")
        keys = {q["kind"]: set(q) for q in legacy}
        for q in samples:
            self.assertEqual(set(q), keys[q["kind"]])
            self.assertIn(q["answer"], q["choices"])
            self.assertEqual(q["answerable_by_sensor"], q["rho_px"] >= q["p0_U_px"])
            self.assertIsNone(q["rho_tok"])
        self.assertEqual(len({q["id"] for q in samples}), 7)
        self.assertEqual(len({(q["image_path"], q["question"]) for q in samples}), 7)
        samples2, _ = bs.build_dataset(self.root, self.config, self.specs)
        self.assertEqual(samples, samples2)

    def test_json_and_jsonl_match_after_export(self):
        self.all_kinds_scene()
        samples, report = bs.build_dataset(self.root, self.config, self.specs)
        self.assertTrue(bs.write_outputs(self.root, "fixture", samples, report))
        qa = bs.load_json(self.root / "fixture.json")
        lines = [json.loads(x) for x in (self.root / "fixture.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(qa, lines)

    def test_incomplete_build_preserves_existing_outputs(self):
        self.scene([(10, self.rect(50, 50, 10, 20))])
        samples, report = bs.build_dataset(self.root, self.config, self.specs)
        for ext in ("json", "jsonl"):
            (self.root / f"old.{ext}").write_text("old content")
        self.assertFalse(bs.write_outputs(self.root, "old", samples, report))
        self.assertEqual((self.root / "old.json").read_text(), "old content")
        self.assertEqual((self.root / "old.jsonl").read_text(), "old content")
        self.assertTrue((self.root / "old_build_report.json").exists())

    def test_real_p1142_near_square_is_excluded(self):
        objs = bs.parse_dota_label(bs.BASE_DIR / "labels/dota_P1142.txt", 1024, 1024)
        with self.assertRaises(bs.SkipSample):
            bs.orientation_answer(objs[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
