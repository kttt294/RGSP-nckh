"""Bộ đọc nhãn cho dataset xView (DIUx xView 2018 Challenge).

Hỗ trợ:
- Định dạng chuẩn GeoJSON (FeatureCollection với bounds_imcoords và type_id).
- Ánh xạ 60 mã lớp số (type_id 11..94) của xView sang tên lớp chuẩn hóa (canonical name).
- GSD cố định là 0.3 m/pixel (ảnh vệ tinh WorldView-3).
"""
from __future__ import annotations
import json
import math
from pathlib import Path
from core_generators import InvalidInput, obb_metrics, polygon_area, clip_polygon

# Ánh xạ mã lớp số tiêu chuẩn của xView Challenge
XVIEW_ID_MAP = {
    # Mã số tiêu chuẩn DIUx xView Challenge (11..94)
    11: "airplane", 12: "small_aircraft", 13: "airplane", 15: "helicopter",
    17: "small_vehicle", 18: "small_vehicle", 19: "pickup_truck", 20: "utility_truck",
    21: "truck", 23: "truck", 24: "truck_tractor_w_box_trailer", 25: "truck_tractor",
    26: "trailer", 27: "truck_tractor_w_flatbed_trailer", 28: "truck_tractor_w_liquid_tank",
    29: "bus", 32: "crane_truck", 33: "railway_vehicle", 34: "railway_passenger_car",
    35: "railway_cargo_car", 36: "railway_flat_car", 37: "railway_tank_car", 38: "locomotive",
    40: "ship", 41: "motorboat", 42: "sailboat", 44: "tugboat", 45: "barge",
    47: "fishing_vessel", 49: "ferry", 50: "yacht", 51: "container_ship", 52: "oil_tanker",
    53: "engineering_vehicle", 54: "tower_crane", 55: "container_crane", 56: "reach_stacker",
    57: "straddle_carrier", 59: "mobile_crane", 60: "dump_truck", 61: "haul_truck",
    62: "scraper_tractor", 63: "bulldozer", 64: "excavator", 65: "cement_mixer",
    66: "ground_grader", 71: "hut_tent", 72: "shed", 73: "building", 74: "aircraft_hangar",
    76: "damaged_building", 77: "facility", 79: "construction_site", 83: "vehicle_lot",
    84: "helipad", 86: "storage_tank", 89: "shipping_container_lot", 91: "shipping_container",
    93: "pylon", 94: "tower",

    # Chỉ mục 0..59 từ HuggingFace Parquet (HichTala/xview)
    0: "airplane", 1: "small_aircraft", 2: "airplane", 3: "helicopter",
    4: "small_vehicle", 5: "small_vehicle", 6: "bus", 7: "pickup_truck",
    8: "utility_truck", 9: "truck", 10: "truck", 11: "truck_tractor_w_box_trailer",
    12: "truck_tractor", 13: "trailer", 14: "truck_tractor_w_flatbed_trailer", 15: "truck_tractor_w_liquid_tank",
    16: "crane_truck", 17: "railway_vehicle", 18: "railway_passenger_car", 19: "railway_cargo_car",
    20: "railway_flat_car", 21: "railway_tank_car", 22: "locomotive", 23: "ship",
    24: "motorboat", 25: "sailboat", 26: "tugboat", 27: "barge", 28: "fishing_vessel",
    29: "ferry", 30: "yacht", 31: "container_ship", 32: "oil_tanker", 33: "engineering_vehicle",
    34: "tower_crane", 35: "container_crane", 36: "reach_stacker", 37: "straddle_carrier",
    38: "mobile_crane", 39: "dump_truck", 40: "haul_truck", 41: "scraper_tractor",
    42: "bulldozer", 43: "excavator", 44: "cement_mixer", 45: "ground_grader",
    46: "hut_tent", 47: "shed", 48: "building", 49: "aircraft_hangar",
    50: "damaged_building", 51: "facility", 52: "construction_site", 53: "vehicle_lot",
    54: "helipad", 55: "storage_tank", 56: "shipping_container_lot", 57: "shipping_container",
    58: "pylon", 59: "tower",
}


_XVIEW_CACHE = {}


def _get_xview_features(geojson_path):
    p_str = str(Path(geojson_path).resolve())
    if p_str not in _XVIEW_CACHE:
        path = Path(geojson_path)
        if not path.is_file():
            raise InvalidInput(f"Tệp nhãn xView GeoJSON không tồn tại: {geojson_path}")
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise InvalidInput(f"Lỗi đọc file GeoJSON {path.name}: {exc}") from exc
        
        features = data.get("features", [])
        by_name = {}
        for f in features:
            img_id = f.get("properties", {}).get("image_id", "").lower()
            if img_id:
                by_name.setdefault(img_id, []).append(f)
                stem = Path(img_id).stem.lower()
                if stem != img_id:
                    by_name.setdefault(stem, []).append(f)
        _XVIEW_CACHE[p_str] = (by_name, features)
    return _XVIEW_CACHE[p_str]


def parse_xview_label(geojson_path, image_name, img_w, img_h, *, class_map=None):
    """Trích xuất danh sách đối tượng cho một ảnh từ tệp GeoJSON của xView."""
    by_name, features = _get_xview_features(geojson_path)
    id_mapping = XVIEW_ID_MAP if class_map is None else class_map
    target_name = Path(image_name).name.lower()
    target_stem = Path(image_name).stem.lower()
    objects = []
    
    matched_features = by_name.get(target_name) or by_name.get(target_stem) or []
    
    # Nếu tệp geojson chỉ dành riêng cho 1 ảnh (như file mẫu) thì dùng toàn bộ
    if not matched_features and len(features) > 0 and len({f.get("properties", {}).get("image_id") for f in features}) <= 1:
        matched_features = features
        
    for index, feat in enumerate(matched_features):
        props = feat.get("properties", {})
        type_id = props.get("type_id", -1)
        if type_id not in id_mapping:
            continue
        canonical_name = id_mapping[type_id]
        
        # Đọc bounds_imcoords (dạng 'xmin,ymin,xmax,ymax')
        bounds_str = props.get("bounds_imcoords")
        if bounds_str and isinstance(bounds_str, str):
            try:
                coords = [float(v.strip()) for v in bounds_str.split(",")]
                xmin, ymin, xmax, ymax = coords[:4]
            except Exception:
                xmin = ymin = xmax = ymax = 0
        else:
            # Dự phòng đọc từ geometry.coordinates
            geom = feat.get("geometry", {})
            coords = geom.get("coordinates", [[]])[0]
            if len(coords) >= 4:
                xs = [p[0] for p in coords]
                ys = [p[1] for p in coords]
                xmin, xmax = min(xs), max(xs)
                ymin, ymax = min(ys), max(ys)
            else:
                continue
                
        xmin, xmax = min(xmin, xmax), max(xmin, xmax)
        ymin, ymax = min(ymin, ymax), max(ymin, ymax)
        w, h = xmax - xmin, ymax - ymin
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
            "difficult": False
        })
        
    return objects
