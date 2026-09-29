"""Bộ đọc nhãn cho dataset iSAID (Instance Segmentation in Aerial Images).

Hỗ trợ:
- Định dạng chuẩn COCO JSON (chứa đa giác segmentation polygons, category_id, bbox).
- Tính toán footprint và OBB từ đa giác mặt nạ segmentation.
- Ánh xạ 15 lớp iSAID sang tên lớp chuẩn hóa (canonical name).
"""
from __future__ import annotations
import json
import math
from pathlib import Path
from core_generators import InvalidInput, obb_metrics, polygon_area, clip_polygon

ISAID_CLASSES = (
    "ship", "storage_tank", "baseball_diamond", "tennis_court", "basketball_court",
    "ground_track_field", "bridge", "large_vehicle", "small_vehicle", "helicopter",
    "swimming_pool", "roundabout", "soccer_ball_field", "plane", "harbor"
)
ISAID_CLASS_MAP = {i + 1: c for i, c in enumerate(ISAID_CLASSES)}
ISAID_NAME_MAP = {c: ("airplane" if c == "plane" else c) for c in ISAID_CLASSES}


def polygon_to_points(flat_coords):
    """Chuyển đổi danh sách tọa độ phẳng [x1, y1, x2, y2, ...] thành [(x1, y1), ...]."""
    return list(zip(flat_coords[::2], flat_coords[1::2]))


def parse_isaid_label(json_path, image_name_or_id, img_w, img_h, *, category_map=None):
    """Trích xuất danh sách đối tượng của một ảnh cụ thể từ tệp chú thích iSAID COCO JSON."""
    path = Path(json_path)
    if not path.is_file():
        raise InvalidInput(f"Tệp nhãn iSAID không tồn tại: {json_path}")
    
    data = json.loads(path.read_text(encoding="utf-8"))
    cats = {c["id"]: c["name"] for c in data.get("categories", [])} if category_map is None else category_map

    # Tìm image_id tương ứng
    target_img_id = None
    target_stem = Path(str(image_name_or_id)).stem.lower()
    
    for img in data.get("images", []):
        file_stem = Path(img.get("file_name", "")).stem.lower()
        if file_stem == target_stem or str(img.get("id")) == str(image_name_or_id):
            target_img_id = img["id"]
            break
            
    if target_img_id is None:
        # Nếu không tìm thấy theo tên, thử dùng trực tiếp nếu image_name_or_id là số
        try:
            target_img_id = int(image_name_or_id)
        except (ValueError, TypeError):
            pass

    objects = []
    anns = [a for a in data.get("annotations", []) if target_img_id is None or a.get("image_id") == target_img_id]
    
    for index, ann in enumerate(anns):
        cat_id = ann.get("category_id")
        raw_name = cats.get(cat_id, "unknown")
        canonical_name = ISAID_NAME_MAP.get(raw_name, raw_name)
        
        seg = ann.get("segmentation", [])
        if not seg or not isinstance(seg, list):
            continue
            
        flat_poly = seg[0] if isinstance(seg[0], list) else seg
        if len(flat_poly) < 6 or len(flat_poly) % 2 != 0:
            continue
            
        all_pts = polygon_to_points(flat_poly)
        
        # Nếu là tứ giác lồi, dùng trực tiếp; nếu nhiều hơn 4 điểm, tìm 4 đỉnh xấp xỉ OBB
        if len(all_pts) == 4:
            pts = all_pts
        else:
            # Tạo 4 đỉnh từ min/max hộp bao định hướng
            xs, ys = [p[0] for p in all_pts], [p[1] for p in all_pts]
            min_x, max_x, min_y, max_y = min(xs), max(xs), min(ys), max(ys)
            pts = [(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)]
            
        try:
            minor, major, angle = obb_metrics(pts)
        except Exception:
            # Dự phòng cho đa giác phẳng ngang
            w, h = max(p[0] for p in pts) - min(p[0] for p in pts), max(p[1] for p in pts) - min(p[1] for p in pts)
            minor, major, angle = min(w, h), max(w, h), 0.0
            
        area = polygon_area(all_pts) if len(all_pts) >= 3 else (major * minor)
        visible = clip_polygon(pts, (0, 0, img_w, img_h))
        if polygon_area(visible) <= 0:
            continue
            
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        col = min(2, max(0, int(cx * 3 / img_w)))
        row = min(2, max(0, int(cy * 3 / img_h)))
        
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
            "difficult": ann.get("iscrowd", 0) == 1
        })
        
    return objects
