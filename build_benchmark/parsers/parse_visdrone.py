"""Bộ đọc nhãn cho dataset VisDrone (VisDrone-DET / AISKYEYE).

Hỗ trợ:
- Định dạng chuẩn TXT 8 cột:
  <bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>
- Lọc bỏ vùng bỏ qua (score == 0 hoặc object_category == 0).
- Xử lý trạng thái che khuất (occlusion == 2 -> difficult / che khuất > 50%).
- Ánh xạ 10 lớp VisDrone sang tên lớp chuẩn hóa (canonical name).
"""
from __future__ import annotations
import math
from pathlib import Path
from core_generators import InvalidInput, obb_metrics, polygon_area, clip_polygon

VISDRONE_NAME_MAP = {
    1: "pedestrian",
    2: "people",
    3: "bicycle",
    4: "small_vehicle",   # car
    5: "van",             # van
    6: "truck",           # truck
    7: "tricycle",
    8: "awning_tricycle",
    9: "bus",             # bus
    10: "motorcycle",     # motor
}


def parse_visdrone_label(txt_path, img_w, img_h, *, class_map=None):
    """Phân tích tệp chú thích định dạng TXT của VisDrone và trả về danh sách đối tượng."""
    path = Path(txt_path)
    if not path.is_file():
        raise InvalidInput(f"Tệp nhãn VisDrone không tồn tại: {txt_path}")
        
    name_mapping = VISDRONE_NAME_MAP if class_map is None else class_map
    objects = []
    
    for index, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines()):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        p = [v.strip() for v in line.split(",") if v.strip()]
        if len(p) < 8:
            p = [v.strip() for v in line.split() if v.strip()]
        if len(p) < 8:
            continue
            
        try:
            x, y, w, h = map(float, p[:4])
            score = int(p[4])
            cat_id = int(p[5])
            truncation = int(p[6])
            occlusion = int(p[7])
        except (ValueError, TypeError) as exc:
            continue
            
        # Bỏ qua các vùng không đánh giá (score == 0 hoặc ignored region cat_id == 0)
        if score == 0 or cat_id == 0 or cat_id not in name_mapping:
            continue
        if w <= 0 or h <= 0:
            continue
            
        canonical_name = name_mapping[cat_id]
        xmin, xmax = x, x + w
        ymin, ymax = y, y + h
        
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
        
        # occlusion == 2 nghĩa là vật thể bị che khuất trên 50%
        difficult = (occlusion == 2 or truncation == 1)
        
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
            "visible_fraction": 0.4 if occlusion == 2 else (polygon_area(visible) / max(1.0, area)),
            "difficult": difficult
        })
        
    return objects
