"""Kiểm thử đơn vị cho 5 parser dữ liệu viễn thám trong rs-solve/parsers/."""
import unittest
from pathlib import Path
from PIL import Image

from parsers import (
    parse_dota_label,
    parse_dota_header,
    parse_isaid_label,
    parse_dior_label,
    parse_xview_label,
    parse_visdrone_label,
)

BASE_DIR = Path(__file__).resolve().parent
SAMPLE_DIR = BASE_DIR / "sample_data"


class TestParsers(unittest.TestCase):

    def test_dota_parser(self):
        txt_path = SAMPLE_DIR / "dota" / "labels" / "dota_P1053.txt"
        header = parse_dota_header(txt_path)
        self.assertEqual(header.get("gsd"), "0.3")
        self.assertEqual(header.get("imagesource"), "GoogleEarth")

        objects = parse_dota_label(txt_path, 1024, 1024, label_format="yolo_obb")
        self.assertGreater(len(objects), 0)
        cids = {o["class_id"] for o in objects}
        self.assertIn("swimming_pool", cids)
        self.assertIn("small_vehicle", cids)
        for o in objects:
            self.assertIn(o["cell"][0], "ABC")
            self.assertIn(o["cell"][1], "123")
            self.assertGreater(o["minor_len_px"], 0)
            self.assertGreater(o["major_len_px"], 0)

    def test_visdrone_parser(self):
        txt_path = SAMPLE_DIR / "visdrone" / "annotations" / "0000001_02999_d_0000005.txt"
        img_path = SAMPLE_DIR / "visdrone" / "images" / "0000001_02999_d_0000005.jpg"
        with Image.open(img_path) as img:
            w, h = img.size
        self.assertEqual((w, h), (1920, 1080))

        objects = parse_visdrone_label(txt_path, w, h)
        self.assertGreater(len(objects), 50)
        cids = {o["class_id"] for o in objects}
        self.assertTrue("small_vehicle" in cids or "pedestrian" in cids)
        for o in objects:
            self.assertEqual(len(o["bbox"]), 4)
            self.assertLessEqual(o["bbox"][2], w)
            self.assertLessEqual(o["bbox"][3], h)

    def test_dior_parser(self):
        xml_path = (SAMPLE_DIR / "dior" / "Annotations" / "00001.xml") if (SAMPLE_DIR / "dior" / "Annotations" / "00001.xml").exists() else (SAMPLE_DIR / "dior" / "annotations" / "00001.xml")
        img_path = (SAMPLE_DIR / "dior" / "images" / "00001.jpg") if (SAMPLE_DIR / "dior" / "images" / "00001.jpg").exists() else (SAMPLE_DIR / "dior" / "JPEGImages" / "00001.jpg")
        with Image.open(img_path) as img:
            w, h = img.size
        self.assertEqual((w, h), (800, 800))

        objects = parse_dior_label(xml_path, w, h)
        self.assertGreater(len(objects), 100)
        cids = {o["class_id"] for o in objects}
        self.assertTrue("storage_tank" in cids or "ship" in cids)
        for o in objects:
            self.assertEqual(len(o["pts"]), 4)
            self.assertGreaterEqual(o["area_px2"], 1.0)

    def test_xview_parser(self):
        geojson_path = SAMPLE_DIR / "xview" / "labels" / "xview_sample.geojson"
        img_path = SAMPLE_DIR / "xview" / "images" / "xview_20881.jpg"
        with Image.open(img_path) as img:
            w, h = img.size

        objects = parse_xview_label(geojson_path, img_path.name, w, h)
        self.assertGreater(len(objects), 5)
        for o in objects:
            self.assertTrue(o["class_id"])
            self.assertEqual(len(o["bbox"]), 4)

    def test_isaid_parser(self):
        json_path = SAMPLE_DIR / "isaid" / "annotations" / "iSAID_sample.json"
        img_path = SAMPLE_DIR / "isaid" / "images" / "P1053.jpg"
        with Image.open(img_path) as img:
            w, h = img.size

        objects = parse_isaid_label(json_path, img_path.name, w, h)
        self.assertGreater(len(objects), 5)
        cids = {o["class_id"] for o in objects}
        self.assertIn("swimming_pool", cids)
        for o in objects:
            self.assertEqual(len(o["pts"]), 4)
            self.assertGreaterEqual(o["area_px2"], 1.0)


if __name__ == "__main__":
    unittest.main()
