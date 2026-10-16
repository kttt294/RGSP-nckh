"""Bộ đọc nhãn cho dataset DIOR (Object Detection in Optical Remote Sensing Images).

Hỗ trợ:
- Định dạng chuẩn Pascal VOC XML (chứa thẻ <annotation><object><name><bndbox>).
- Chuyển đổi hộp bao ngang HBB sang 4 đỉnh đa giác OBB chuẩn.
- Ánh xạ 20 lớp DIOR sang tên lớp chuẩn hóa (canonical name).
"""
from __future__ import annotations
import math
from pathlib import Path
import xml.etree.ElementTree as ET
from core_generators import InvalidInput, obb_metrics, polygon_area, clip_polygon

DIOR_NAME_MAP = {
    "airplane": "airplane",
    "airport": "airport",
    "baseballfield": "baseball_diamond",
    "basketballcourt": "basketball_court",
    "bridge": "bridge",
    "chimney": "chimney",
    "dam": "dam",
    "expressway-service-area": "expressway_service_area",
    "expressway-toll-station": "expressway_toll_station",
    "golffield": "golf_course",
    "golfcourse": "golf_course",
    "groundtrackfield": "ground_track_field",
    "harbor": "harbor",
    "overpass": "overpass",
    "ship": "ship",
    "stadium": "stadium",
    "storagetank": "storage_tank",
    "tenniscourt": "tennis_court",
    "trainstation": "train_station",
    "vehicle": "small_vehicle",
    "windmill": "windmill",
}


def parse_dior_label(xml_path, img_w, img_h, *, class_map=None):
    """Phân tích tệp XML Pascal VOC của DIOR và trả về danh sách đối tượng chuẩn hóa."""
    path = Path(xml_path)
    if not path.is_file():
        raise InvalidInput(f"Tệp nhãn DIOR XML không tồn tại: {xml_path}")
        
    try:
        tree = ET.parse(path)
        root = tree.getroot()
    except Exception as exc:
        raise InvalidInput(f"Lỗi đọc file XML {path.name}: {exc}") from exc
        
    name_mapping = DIOR_NAME_MAP if class_map is None else class_map
    objects = []
    
    for index, obj_elem in enumerate(root.findall("object")):
        raw_name = obj_elem.findtext("name", "").strip().lower()
        canonical_name = name_mapping.get(raw_name, raw_name)
        
        bnd = obj_elem.find("bndbox")
        if bnd is None:
            continue
            
        try:
            xmin = float(bnd.findtext("xmin", 0))
            ymin = float(bnd.findtext("ymin", 0))
            xmax = float(bnd.findtext("xmax", 0))
            ymax = float(bnd.findtext("ymax", 0))
        except (ValueError, TypeError) as exc:
            raise InvalidInput(f"{path.name}, đối tượng {index+1}: Tọa độ không hợp lệ") from exc
            
        xmin, xmax = min(xmin, xmax), max(xmin, xmax)
        ymin, ymax = min(ymin, ymax), max(ymin, ymax)
        
        w = xmax - xmin
        h = ymax - ymin
        if w <= 0 or h <= 0:
            continue
            
        pts = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
        minor = min(w, h)
        major = max(w, h)
        angle = 0.0
        area = w * h
        
        visible = clip_polygon(pts, (0, 0, img_w, img_h))
        if polygon_area(visible) <= 0:
            continue
            
        cx = xmin + w / 2.0
        cy = ymin + h / 2.0
        col = min(2, max(0, int(cx * 3 / img_w)))
        row = min(2, max(0, int(cy * 3 / img_h)))
        
        difficult = obj_elem.findtext("difficult", "0").strip() == "1"
        
        objects.append({
            "obj_idx": index,
            "class_id": canonical_name,
            "pts": pts,
            "bbox": [min(p[0] for p in visible), min(p[1] for p in visible),
                     max(p[0] for p in visible), max(p[1] for p in visible)],
            "center": (cx, cy),
            "cell": f"{'ABC'[col]}{row+1}",
            "minor_len_px": max(1.0, minor),
            "major_len_px": max(1.0, major),
            "angle_deg": angle,
            "area_px2": max(1.0, area),
            "visible_fraction": polygon_area(visible) / max(1.0, area),
            "difficult": difficult
        })
        
    return objects
