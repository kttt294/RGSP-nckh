"""Bộ đọc nhãn cho dataset DOTA (DOTA-v1.0, DOTA-v1.5, DOTA-v2.0).

Hỗ trợ:
- Định dạng DOTA gốc (8 tọa độ pixel + tên lớp + difficult).
- Định dạng YOLO-OBB (class_id + 8 tọa độ chuẩn hóa 0..1).
- Trích xuất tự động siêu dữ liệu GSD và imagesource từ 2 dòng đầu tệp nhãn.
"""
from __future__ import annotations
import math
from pathlib import Path
from core_generators import InvalidInput, obb_metrics, polygon_area, clip_polygon

DOTA_CLASS_MAP = dict(enumerate((
    "airplane", "ship", "storage_tank", "baseball_diamond", "tennis_court",
    "basketball_court", "ground_track_field", "harbor", "bridge", "large_vehicle",
    "small_vehicle", "helicopter", "roundabout", "soccer_ball_field",
    "swimming_pool", "container_crane", "airport", "helipad"
)))
DOTA_NAME_MAP = {c.replace("_", "-"): c for c in DOTA_CLASS_MAP.values()}
DOTA_NAME_MAP["plane"] = "airplane"


def parse_dota_header(txt_path):
    """Đọc thông tin gsd và imagesource từ header của tệp nhãn DOTA nếu có."""
    info = {}
    path = Path(txt_path)
    if not path.is_file():
        return info
    for line in path.read_text(encoding="utf-8-sig").splitlines()[:10]:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, val = line.partition(":")
        if sep:
            k = key.strip().strip("'\"").lower()
            v = val.strip().strip("'\"")
            if k in ("gsd", "imagesource", "acquisition dates"):
                info[k] = v
    return info


def parse_dota_label(txt_path, img_w, img_h, *, label_format="auto", class_map=None):
    """Phân tích nhãn DOTA từ file txt và trả về danh sách đối tượng chuẩn hóa."""
    if label_format not in ("auto", "yolo_obb", "dota"):
        raise InvalidInput(f"label_format chưa hỗ trợ: {label_format}")
    mapping = DOTA_CLASS_MAP if class_map is None else {int(k): v for k, v in class_map.items()}
    objects = []
    
    raw_lines = Path(txt_path).read_text(encoding="utf-8-sig").splitlines()
    
    effective_fmt = label_format
    if effective_fmt == "auto":
        effective_fmt = "yolo_obb"
        for line in raw_lines:
            s = line.strip()
            if not s or s.startswith("#") or s.partition(":")[0].strip().strip("'\"").lower() in ("gsd", "imagesource", "acquisition dates"):
                continue
            tokens = s.split()
            if len(tokens) == 10 and tokens[8] in DOTA_NAME_MAP:
                effective_fmt = "dota"
            break

    for index, raw in enumerate(raw_lines):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.partition(":")[0].strip().strip("'\"").lower() in ("gsd", "imagesource", "acquisition dates"):
            continue
        p = line.split()
        try:
            if effective_fmt == "yolo_obb":
                if len(p) != 9:
                    raise InvalidInput("YOLO-OBB cần đúng 9 cột")
                cid = float(p[0])
                if not math.isfinite(cid) or not cid.is_integer() or int(cid) not in mapping:
                    raise InvalidInput("ID lớp không có trong class_map")
                name, difficult = mapping[int(cid)], False
                coords = list(map(float, p[1:]))
                pts = [(coords[i] * img_w, coords[i+1] * img_h) for i in range(0, 8, 2)]
            else:
                if len(p) != 10 or p[8] not in DOTA_NAME_MAP or p[9] not in ("0", "1"):
                    raise InvalidInput("DOTA cần 8 tọa độ + tên lớp + difficult 0/1")
                name, difficult = DOTA_NAME_MAP[p[8]], p[9] == "1"
                coords = list(map(float, p[:8]))
                pts = list(zip(coords[::2], coords[1::2]))
                
            if not all(math.isfinite(v) for pt in pts for v in pt):
                raise InvalidInput("Tọa độ không hữu hạn")
            minor, major, angle = obb_metrics(pts)
            area = polygon_area(pts)
            visible = clip_polygon(pts, (0, 0, img_w, img_h))
            if polygon_area(visible) <= 0:
                raise InvalidInput("Annotation hoàn toàn ngoài ảnh")
            cx, cy = (sum(pt[j] for pt in pts) / 4 for j in (0, 1))
            col = min(2, max(0, int(cx * 3 / img_w)))
            row = min(2, max(0, int(cy * 3 / img_h)))
            objects.append({
                "obj_idx": index, "class_id": name, "pts": pts,
                "bbox": [min(pt[0] for pt in visible), min(pt[1] for pt in visible),
                         max(pt[0] for pt in visible), max(pt[1] for pt in visible)],
                "center": (cx, cy), "cell": f"{'ABC'[col]}{row+1}",
                "minor_len_px": minor, "major_len_px": major, "angle_deg": angle,
                "area_px2": area, "visible_fraction": polygon_area(visible) / area,
                "difficult": difficult
            })
        except (ValueError, OverflowError) as exc:
            raise InvalidInput(f"{Path(txt_path).name}, dòng {index+1}: {exc}") from exc
    return objects
