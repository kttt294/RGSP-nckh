"""Sinh QA trắc nghiệm RS-Solve, giữ nguyên các trường JSON của bản cũ.

Chạy: python build_samples.py --per-kind 1
      python build_samples.py --per-kind 5 --output-prefix mau_35_questions

Nguyên tắc:
- GSD lấy từ metadata có nguồn; không đoán từ JPEG/tên dataset.
- OBB thật -> footprint ở mức lấy mẫu tham chiếu -> L_m; Q2 mới dùng L_min.
- Ngưỡng lấy từ calibration; không gán sẵn 4/5/6/8/12 rồi gọi là ngưỡng người.
- Không đủ bằng chứng -> bỏ candidate và ghi báo cáo, không bịa đáp án.
- Q6 dùng câu Q3-HF/Q4/Q5 cận ngưỡng. Tất cả đầu ra vẫn là trắc nghiệm.
Xem HUONG_DAN_BUILD.md để điền build_config.json.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
ABSTAIN = "Không thể xác định từ ảnh"
KINDS = ("Q1", "Q2", "Q3-HF", "Q3-LF", "Q4", "Q5", "Q6")
GRID_PREFIX = "Chia ảnh thành lưới 3x3: cột A-C từ trái sang phải, hàng 1-3 từ trên xuống. "
# Đây là class map của BA FILE YOLO-OBB hiện tại. DOTA gốc dùng tên lớp,
# không bảo đảm một thứ tự numeric ID chung cho mọi công cụ chuyển đổi.
DOTA_CLASS_MAP = dict(enumerate((
    "airplane", "ship", "storage_tank", "baseball_diamond", "tennis_court",
    "basketball_court", "ground_track_field", "harbor", "bridge", "large_vehicle",
    "small_vehicle", "helicopter", "roundabout", "soccer_ball_field",
    "swimming_pool", "container_crane", "airport", "helipad")))
DOTA_NAME_MAP = {c.replace("_", "-"): c for c in DOTA_CLASS_MAP.values()}
DOTA_NAME_MAP["plane"] = "airplane"
CONFUSING_PAIRS = (("soccer_ball_field", "baseball_diamond"),
                   ("tennis_court", "basketball_court"), ("bridge", "dam"),
                   ("storage_tank", "roundabout"), ("helicopter", "small_aircraft"))
TASKS = {"Q1": "detection", "Q2": "hard_negative", "Q3-HF": "orientation",
         "Q3-LF": "color_lf", "Q4": "counting", "Q5": "grounding"}
ORIENTATION_CHOICES = ("Từ trên trái xuống dưới phải", "Từ trên phải xuống dưới trái")
DEFAULT_COLORS = ("Màu đỏ", "Màu xanh lá cây", "Màu xanh dương",
                  "Màu vàng", "Màu trắng", "Màu đen / xám")


class InvalidInput(ValueError):
    """Dữ liệu/cấu hình chưa đủ rõ để dùng cho benchmark."""


class SkipSample(ValueError):
    """Candidate không đáp ứng điều kiện sinh câu; ghi lý do và bỏ qua."""


def positive_number(value, name):
    # bool cũng là int trong Python, nên phải loại riêng True/False.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidInput(f"{name}: cần số dương đã xác minh, hiện là {value!r}")
    if not math.isfinite(value) or value <= 0:
        raise InvalidInput(f"{name}: cần số hữu hạn > 0")
    return float(value)


def has_source(value):
    return isinstance(value, str) and bool(value.strip())


def load_json(path):
    def unique_keys(pairs):
        result = {}
        for k, v in pairs:
            if k in result:
                raise InvalidInput(f"Khóa JSON bị lặp: {k}")
            result[k] = v
        return result
    with Path(path).open(encoding="utf-8-sig") as f:
        return json.load(f, object_pairs_hook=unique_keys)


def load_physical_sizes(path=None):
    specs = load_json(path or BASE_DIR / "physical_sizes.json")
    for cid, r in specs.items():
        if r.get("class_id") != cid:
            raise InvalidInput(f"class_id không khớp khóa: {cid}")
        if positive_number(r.get("L_min_m"), cid) > positive_number(r.get("L_typ_m"), cid):
            raise InvalidInput(f"{cid}: L_min > L_typ")
    return specs


def read_dota_gsd(path):
    """Đọc dòng gsd:...; None là chưa biết. JFIF density/DPI không phải GSD."""
    values = []
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        key, sep, value = line.partition(":")
        if not sep or key.strip().strip("'\"").lower() != "gsd":
            continue
        value = value.strip().strip("'\"")
        if value.lower() not in ("none", "null", "nan", ""):
            try:
                values.append(positive_number(float(value), "gsd metadata"))
            except ValueError as exc:
                raise InvalidInput(f"GSD không hợp lệ trong {path}: {value}") from exc
    if values and any(not math.isclose(v, values[0], rel_tol=1e-9) for v in values):
        raise InvalidInput("Nhiều GSD mâu thuẫn trong metadata")
    return values[0] if values else None


def resolve_gsd(root, meta):
    values = []
    paths = [meta["gsd_metadata_path"]] if meta.get("gsd_metadata_path") else []
    if meta.get("label_path") and meta["label_path"] not in paths:
        paths.append(meta["label_path"])
    for path in paths:
        value = read_dota_gsd(root / path)
        if value is not None:
            values.append(value)
    if meta.get("gsd_m") is not None:
        if not has_source(meta.get("gsd_source")):
            raise InvalidInput("gsd_m đã điền nhưng thiếu gsd_source")
        values.append(positive_number(meta["gsd_m"], "gsd_m"))
    if not values:
        raise InvalidInput("Chưa có GSD gốc đã xác minh; không dùng 0.25/0.30 mặc định")
    if any(not math.isclose(v, values[0], rel_tol=1e-9) for v in values):
        raise InvalidInput("GSD trong cấu hình và metadata không thống nhất")
    return values[0]


def polygon_area(points):
    return abs(sum(x * points[(i+1) % len(points)][1] - y * points[(i+1) % len(points)][0]
                   for i, (x, y) in enumerate(points))) / 2 if points else 0.0


def clip_polygon(points, box):
    """Cắt polygon theo hình chữ nhật bằng Sutherland–Hodgman.

    Tỷ lệ trong ô dùng diện tích polygon, tránh đếm cả nền trong HBB của vật thể
    xoay. Mẫu số là polygon đầy đủ, gồm cả phần bị cắt khỏi ảnh.
    """
    output = list(points)
    for axis, bound, greater in ((0, box[0], True), (0, box[2], False),
                                  (1, box[1], True), (1, box[3], False)):
        original, output = output, []
        if not original:
            break
        previous = original[-1]
        def inside(p):
            return p[axis] >= bound if greater else p[axis] <= bound
        for current in original:
            if inside(current) != inside(previous):
                t = (bound - previous[axis]) / (current[axis] - previous[axis])
                output.append(tuple(previous[j] + t * (current[j] - previous[j]) for j in (0, 1)))
            if inside(current):
                output.append(current)
            previous = current
    return output


def obb_metrics(points):
    """Rectangle bao nhỏ nhất của tứ giác lồi; trả minor, major, góc y-down.

    DOTA có tứ giác hơi lệch. Duyệt hướng bốn cạnh thay vì giả định hai cạnh
    đầu luôn tạo rectangle hoàn hảo. Đỉnh sai thứ tự/tự cắt bị báo lỗi.
    """
    crosses, candidates = [], []
    for i, p in enumerate(points):
        q, r = points[(i+1) % 4], points[(i+2) % 4]
        dx, dy = q[0] - p[0], q[1] - p[1]
        length = math.hypot(dx, dy)
        if length <= 1e-9:
            raise InvalidInput("OBB có cạnh bằng 0")
        crosses.append(dx * (r[1] - q[1]) - dy * (r[0] - q[0]))
        ux, uy = dx / length, dy / length
        a = [x * ux + y * uy for x, y in points]
        b = [-x * uy + y * ux for x, y in points]
        w, h = max(a) - min(a), max(b) - min(b)
        angle = (math.degrees(math.atan2(dy, dx)) + (90 if h > w else 0)) % 180
        candidates.append((w * h, min(w, h), max(w, h), angle))
    if not (all(v > 1e-9 for v in crosses) or all(v < -1e-9 for v in crosses)):
        raise InvalidInput("OBB không lồi hoặc thứ tự đỉnh bị chéo")
    _, minor, major, angle = min(candidates, key=lambda c: c[0])
    return minor, major, angle


def parse_dota_label(txt_path, img_w, img_h, *, label_format="yolo_obb", class_map=None):
    """img_w/img_h bắt buộc lấy từ ảnh thật, không mặc định 1024.

    YOLO-OBB: class_id + 8 tọa độ chuẩn hóa; DOTA: 8 tọa độ pixel + tên lớp +
    difficult. Tọa độ phải khớp ảnh hiện tại. Cho phép đỉnh ngoài biên vì có
    vật thể bị crop; không làm biến dạng OBB bằng cách clamp từng đỉnh.
    obj_idx là chỉ số DÒNG tính từ 0, kể cả các dòng metadata/blank trước đó.
    """
    if label_format not in ("yolo_obb", "dota"):
        raise InvalidInput(f"label_format chưa hỗ trợ: {label_format}")
    mapping = DOTA_CLASS_MAP if class_map is None else {int(k): v for k, v in class_map.items()}
    objects = []
    for index, raw in enumerate(Path(txt_path).read_text(encoding="utf-8-sig").splitlines()):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.partition(":")[0].strip().strip("'\"").lower() in ("gsd", "imagesource", "acquisition dates"):
            continue
        p = line.split()
        try:
            if label_format == "yolo_obb":
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
            objects.append({"obj_idx": index, "class_id": name, "pts": pts,
                "bbox": [min(pt[0] for pt in visible), min(pt[1] for pt in visible),
                         max(pt[0] for pt in visible), max(pt[1] for pt in visible)],
                "center": (cx, cy), "cell": f"{'ABC'[col]}{row+1}",
                "minor_len_px": minor, "major_len_px": major, "angle_deg": angle,
                "area_px2": area, "visible_fraction": polygon_area(visible) / area,
                "difficult": difficult})
        except (ValueError, OverflowError) as exc:
            raise InvalidInput(f"{Path(txt_path).name}, dòng {index+1}: {exc}") from exc
    return objects


@dataclass
class Scene:
    image_path: str
    width: int
    height: int
    gsd_m: float
    scale: float
    objects: list
    meta: dict



def normalize_image_meta(root, image_path, meta=None, config=None):
    """Điền giá trị mặc định cho metadata ảnh khi dev không nhập thủ công."""
    if meta is None:
        meta = {}
    config = config or {}

    # 1. Tự tìm label_path nếu chưa khai báo
    if not meta.get("label_path"):
        stem = Path(image_path).stem
        default_label = Path("labels") / f"{stem}.txt"
        if (root / default_label).exists():
            meta["label_path"] = default_label.as_posix()
        elif (root / f"{stem}.txt").exists():
            meta["label_path"] = f"{stem}.txt"
        else:
            meta["label_path"] = default_label.as_posix()

    # 2. Tự gán label_format nếu chưa khai báo
    if not meta.get("label_format"):
        meta["label_format"] = config.get("default_label_format", "yolo_obb")

    # 3. Kế thừa GSD từ config nếu có default_gsd_m
    if meta.get("gsd_m") is None and config.get("default_gsd_m") is not None:
        meta["gsd_m"] = config["default_gsd_m"]
    if meta.get("gsd_source") is None and config.get("default_gsd_source") is not None:
        meta["gsd_source"] = config["default_gsd_source"]

    # 4. Mặc định ảnh nguyên gốc chưa resize (identity, scale=1.0)
    if "transform_type" not in meta:
        meta["transform_type"] = "identity"
    if "native_to_image_scale" not in meta:
        meta["native_to_image_scale"] = 1.0
    if "pixel_scale_source" not in meta:
        if meta.get("transform_type") == "identity":
            meta["pixel_scale_source"] = "raw dataset image (identity, scale=1.0)"

    # 5. Mặc định taxonomy 18 lớp canonical và annotation hoàn chỉnh
    if "source_classes" not in meta:
        meta["source_classes"] = list(DOTA_CLASS_MAP.values())
    if "complete_classes" not in meta:
        meta["complete_classes"] = list(meta["source_classes"])
    if "annotation_review_source" not in meta:
        meta["annotation_review_source"] = "official dataset benchmark exhaustive annotation"

    # 6. Các review chi tiết mặc định rỗng
    if "absence_reviews" not in meta:
        meta["absence_reviews"] = {}
    if "color_reviews" not in meta:
        meta["color_reviews"] = {}
    if "isolation_reviews" not in meta:
        meta["isolation_reviews"] = {}

    return meta


def load_scene(root, image_path, meta):
    meta = normalize_image_meta(root, image_path, meta)
    with Image.open(root / image_path) as image:
        width, height = image.size
    gsd = resolve_gsd(root, meta)
    scale = positive_number(meta.get("native_to_image_scale"), "native_to_image_scale")
    if not has_source(meta.get("pixel_scale_source")):
        raise InvalidInput("Thiếu nguồn xác minh crop/resize: pixel_scale_source")
    # Chỉ hỗ trợ crop và resize ĐỒNG ĐỀU. Resize méo/perspective cần ánh xạ
    # ngược đầy đủ; một scale duy nhất sẽ làm sai phép đo vật lý.
    if meta.get("transform_type") not in ("identity", "crop", "uniform_resize", "crop_uniform_resize"):
        raise InvalidInput("transform_type chưa xác minh hoặc chưa được hỗ trợ")
    if meta["transform_type"] in ("identity", "crop") and not math.isclose(scale, 1):
        raise InvalidInput("Ảnh gốc/crop không resize phải có scale=1")
    if meta["transform_type"] in ("uniform_resize", "crop_uniform_resize") and not has_source(meta.get("resampling_method")):
        raise InvalidInput("Ảnh đã resize nhưng thiếu resampling_method")
    objs = parse_dota_label(root / meta["label_path"], width, height,
                           label_format=meta["label_format"], class_map=meta.get("class_map"))
    # Giảm mẫu làm MẤT thông tin: không được chia cho scale<1 rồi khôi phục
    # giả số pixel cảm biến ban đầu. Ảnh lấy mẫu lại có kiểm soát dùng mức lấy
    # mẫu mới làm tham chiếu: GSD hiệu dụng tăng, footprint giảm tương ứng.
    # Phóng ảnh chỉ đổi hình học: giữ GSD tham chiếu và loại hệ số phóng khỏi rho.
    # Giả thiết: lịch sử là crop và tối đa MỘT resize đã biết. Với chuỗi giảm
    # rồi phóng, phải khai báo từ raster tham chiếu SAU bước giảm thông tin;
    # không dùng scale tổng của cả chuỗi để che mất bước giảm mẫu.
    information_scale = min(1.0, scale)
    return Scene(image_path, width, height, gsd / information_scale,
                 scale / information_scale, objs, meta)


def threshold(config, kind):
    c = config.get("calibration", {})
    if c.get("status") != "calibrated" or not has_source(c.get("source")) or not has_source(c.get("id")):
        raise SkipSample("Chưa có calibration hoàn tất, ID và nguồn kết quả hiệu chỉnh")
    try:
        return positive_number(c.get("thresholds_px", {}).get(TASKS[kind]), f"p0 {TASKS[kind]}")
    except InvalidInput as exc:
        raise SkipSample(str(exc)) from exc


def require_complete(scene, cid):
    # File nhãn có mặt không tự chứng minh đủ annotation. Cần khai báo phạm vi
    # đã kiểm, nhất là để đếm, xác định thứ tự, isolation và phủ định Q2.
    if cid not in scene.meta.get("complete_classes", []) or not has_source(scene.meta.get("annotation_review_source")):
        raise SkipSample(f"Chưa xác minh nhãn đầy đủ cho lớp {cid}")


def require_isolated(scene, obj):
    require_complete(scene, obj["class_id"])
    same = [o for o in scene.objects if o["class_id"] == obj["class_id"]]
    if len(same) > 1:
        r = scene.meta.get("isolation_reviews", {}).get(str(obj["obj_idx"]), {})
        distance = r.get("nearest_same_class_tokens")
        has_review = has_source(r.get("source")) and has_source(r.get("reference_view"))
        if len(same) > 3 and not has_review:
            raise SkipSample("Q1-iso/Q3 yêu cầu tối đa 3 instance cùng lớp hoặc có review isolation")
        if not has_review:
            raise SkipSample("Thiếu khoảng cách láng giềng theo token từ view tham chiếu")
        if isinstance(distance, bool) or not isinstance(distance, (int, float)) or not math.isfinite(distance) or distance < 4:
            raise SkipSample("Láng giềng cùng lớp gần hơn 4 token hoặc số đo chưa hợp lệ")
    # Chỉ một instance cùng lớp: không có láng giềng cùng lớp để so khoảng cách.


def cell_box(scene, cell):
    c, r = "ABC".index(cell[0]), int(cell[1]) - 1
    return (c * scene.width / 3, r * scene.height / 3,
            (c+1) * scene.width / 3, (r+1) * scene.height / 3)


def in_cell(scene, obj):
    # Chọn ô theo tâm RỒI kiểm diện tích. Vật thể chia đôi đúng biên theo ô
    # bên phải/dưới chứa tâm để không đếm hai lần.
    return polygon_area(clip_polygon(obj["pts"], cell_box(scene, obj["cell"]))) / obj["area_px2"] >= 0.5 - 1e-10


def require_visible(obj):
    if obj["difficult"] or obj["visible_fraction"] < 0.5:
        raise SkipSample("Instance difficult hoặc nhìn thấy dưới 50% diện tích")


def require_unique_target(scene, obj):
    require_visible(obj)
    require_isolated(scene, obj)
    if not in_cell(scene, obj):
        raise SkipSample("Dưới 50% diện tích target nằm trong ô được hỏi")
    if sum(o["class_id"] == obj["class_id"] and o["cell"] == obj["cell"] for o in scene.objects) != 1:
        raise SkipSample("Câu hỏi theo lớp và ô chưa xác định duy nhất target")


def native_footprint(scene, obj):
    # scene.scale = pixel hiện tại / pixel tham chiếu còn mang thông tin.
    # load_scene đã xử lý giảm mẫu; tại đây chỉ loại hệ số phóng hình học.
    return obj["minor_len_px"] / scene.scale


def rng_for(config, scene, kind, identity):
    text = f"{config.get('seed', 42)}|{scene.image_path}|{kind}|{identity}"
    return random.Random(int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big"))


def make_sample(scene, kind, cid, box, rho, p0, question, options, truth, flag, *, cell=None, L_m=None):
    """Gán answer và answerable ở một chỗ để tránh nhãn mâu thuẫn.

    Không làm tròn rho trước khi so ngưỡng (11.999 vẫn <12). rho_tok=None là
    chưa đo; không tự dùng patch size đoán để điền trường này.
    """
    if truth not in options or ABSTAIN in options or len(set(options)) != len(options):
        raise InvalidInput("Lựa chọn không hợp lệ hoặc thiếu đáp án sự thật")
    answerable = rho >= p0
    prompt = (GRID_PREFIX + question) if cell is not None else question
    q = {"id": "", "kind": kind, "image_path": scene.image_path, "gsd_m": scene.gsd_m,
         "question": prompt, "choices": list(options) + [ABSTAIN],
         "answer": truth if answerable else ABSTAIN, "class_id": cid,
         "target_box_xyxy": None if box is None else [round(v, 3) for v in box],
         "L_m": rho * scene.gsd_m if L_m is None else L_m,
         "rho_px": rho, "p0_U_px": p0, "answerable_by_sensor": answerable,
         "rho_tok": None, "isolation_flag": flag}
    if cell is not None:
        q["target_cell"] = cell
    return q


def scan_targets(scene, kind, targets, builder, rejects):
    result = []
    for target in targets:
        try:
            result.append(builder(target))
        except SkipSample as exc:
            rejects.append({"image": scene.image_path, "kind": kind, "reason": str(exc)})
    if not targets:
        rejects.append({"image": scene.image_path, "kind": kind, "reason": "Không có candidate"})
    return result


def generate_q1(scene, specs, config, rejects):
    # Sinh Q1-iso. Không tự đổi sang aggregated khi không xác minh isolation:
    # khả năng nhận ra cụm đông cần hiệu chỉnh riêng, không lấy p0 của iso.
    def build(o):
        require_visible(o)
        require_isolated(scene, o)
        cid = o["class_id"]
        return make_sample(scene, "Q1", cid, o["bbox"], native_footprint(scene, o), threshold(config, "Q1"),
            f"Trong toàn ảnh có {specs[cid]['display_name_vi']} ({cid}) không?",
            ["Có", "Không"], "Có", "isolated")
    largest = {}
    for o in scene.objects:
        if o["class_id"] not in largest or o["minor_len_px"] > largest[o["class_id"]]["minor_len_px"]:
            largest[o["class_id"]] = o
    # Câu toàn ảnh chỉ sinh một lần mỗi lớp, chọn instance lớn nhất làm chứng cứ.
    return scan_targets(scene, "Q1", list(largest.values()), build, rejects)


def generate_q2(scene, specs, config, rejects):
    present = {o["class_id"] for o in scene.objects}
    pairs = [(p, a) for p, a in CONFUSING_PAIRS if p in present and a not in present]
    def build(pair):
        p, a = pair
        if not any(o["class_id"] == p and not o["difficult"] and o["visible_fraction"] >= 0.5 for o in scene.objects):
            raise SkipSample("Lớp gây nhầm không có instance nhìn thấy đủ trong nhãn")
        review = scene.meta.get("absence_reviews", {}).get(a, {})
        if not (review.get("absent") is True and has_source(review.get("source"))):
            # dam không nằm trong DOTA: thiếu nhãn dam không chứng minh nó vắng.
            if a not in scene.meta.get("source_classes", []):
                raise SkipSample(f"Nguồn không gán nhãn {a}; cần review vắng mặt")
            require_complete(scene, a)
        prior = config.get("class_prior_approvals", {}).get(a, {})
        has_config_prior = (prior.get("status") == "reviewed" and has_source(prior.get("source")))
        has_spec_prior = (has_source(specs[a].get("standard_source")) and
                          isinstance(specs[a].get("L_min_m"), (int, float)) and specs[a]["L_min_m"] > 0)
        if not (has_config_prior or has_spec_prior):
            raise SkipSample(f"L_min của {a} chưa được duyệt cho Q2 (cần standard_source trong physical_sizes.json hoặc review trong config)")
        L_m = specs[a]["L_min_m"]
        q = make_sample(scene, "Q2", a, None, L_m / scene.gsd_m, threshold(config, "Q2"),
            f"Trong toàn ảnh có {specs[a]['display_name_vi']} ({a}) không?",
            ["Có", "Không"], "Không", "not_applicable", L_m=L_m)
        # Giữ trường và kiểu chuỗi; giá trị not_applicable nói đúng rằng Q2
        # không có target thật để đánh giá cô lập.
        q["confused_present_class"] = p
        return q
    return scan_targets(scene, "Q2", pairs, build, rejects)


def orientation_answer(o, margin_deg=5.0):
    if o["minor_len_px"] / o["major_len_px"] > 0.8:
        raise SkipSample("OBB gần vuông: cạnh ngắn/cạnh dài > 0.8")
    angle = o["angle_deg"] % 180
    if min(angle % 90, 90 - angle % 90) <= margin_deg:
        raise SkipSample("Hướng gần ranh giới hai bin chéo, nhãn mơ hồ")
    # y xuống dưới: +45 độ là trên trái -> dưới phải. Không gọi hướng địa lý
    # Bắc/Nam khi chưa có north-up. Không đồng nhất trục OBB với heading/mũi.
    return ORIENTATION_CHOICES[0 if angle < 90 else 1]


def generate_q3_hf(scene, specs, config, rejects):
    def build(o):
        require_unique_target(scene, o)
        truth = orientation_answer(o)
        options = list(ORIENTATION_CHOICES)
        rng_for(config, scene, "Q3-HF", o["obj_idx"]).shuffle(options)
        cid = o["class_id"]
        return make_sample(scene, "Q3-HF", cid, o["bbox"], native_footprint(scene, o), threshold(config, "Q3-HF"),
            f"Trục dài của {specs[cid]['display_name_vi']} ở ô {o['cell']} gần với đường chéo nào của ảnh?",
            options, truth, "isolated", cell=o["cell"])
    return scan_targets(scene, "Q3-HF", scene.objects, build, rejects)


def generate_q3_lf(scene, specs, config, rejects):
    palette = config.get("color_palette", list(DEFAULT_COLORS))
    if not isinstance(palette, list) or len(palette) != 6 or any(not isinstance(c, str) for c in palette) or len(set(palette)) != 6 or ABSTAIN in palette:
        raise InvalidInput("color_palette cần đúng sáu nhóm màu khác nhau")
    def build(o):
        require_unique_target(scene, o)
        review = scene.meta.get("color_reviews", {}).get(str(o["obj_idx"]), {})
        # Màu từ ảnh tham chiếu mịn/mask đã kiểm tay. Không suy từ tên lớp và
        # không lấy màu trên mask chỉ có vài pixel. Đổi palette phải hiệu chỉnh lại.
        if review.get("status") != "verified" or not has_source(review.get("source")) or review.get("value") not in palette:
            raise SkipSample("Thiếu màu đã kiểm trên ảnh tham chiếu mịn cho target")
        options = list(palette)
        rng_for(config, scene, "Q3-LF", o["obj_idx"]).shuffle(options)
        cid = o["class_id"]
        return make_sample(scene, "Q3-LF", cid, o["bbox"], native_footprint(scene, o), threshold(config, "Q3-LF"),
            f"{specs[cid]['display_name_vi']} ở ô {o['cell']} có màu chủ đạo nào?",
            options, review["value"], "isolated", cell=o["cell"])
    return scan_targets(scene, "Q3-LF", scene.objects, build, rejects)


def grouped_objects(scene):
    groups = defaultdict(list)
    for o in scene.objects:
        if in_cell(scene, o):
            groups[(o["cell"], o["class_id"])].append(o)
    return list(groups.items())


def count_options(count, rng):
    # Chọn ngẫu nhiên hạng số học của đáp án. Chỉ xáo [n-1,n,n+1] vẫn để lộ
    # đáp án là median. Với n>=2 có đủ số thấp hơn cho cả ba hạng.
    low = rng.randrange(3)
    nums = [count] + rng.sample(range(count), low)
    nums += rng.sample(range(count+1, max(count+5, 2*count+2)), 2-low)
    options = list(map(str, nums))
    rng.shuffle(options)
    return options


def generate_q4(scene, specs, config, rejects):
    def build(group):
        (cell, cid), objects = group
        require_complete(scene, cid)
        if len(objects) < 2:
            raise SkipSample("Cần ít nhất 2 instance để sinh Q4 theo giao thức hiện tại")
        for o in objects:
            require_visible(o)
        count = len(objects)
        rho = statistics.median(native_footprint(scene, o) for o in objects)
        union = [min(o["bbox"][0] for o in objects), min(o["bbox"][1] for o in objects),
                 max(o["bbox"][2] for o in objects), max(o["bbox"][3] for o in objects)]
        # Giữ union box ở target_box_xyxy, nhưng rho là median từng instance;
        # tuyệt đối không đo footprint bằng cả box bao nhóm.
        options = count_options(count, rng_for(config, scene, "Q4", (cell, cid)))
        return make_sample(scene, "Q4", cid, union, rho, threshold(config, "Q4"),
            f"Có chính xác bao nhiêu {specs[cid]['display_name_vi']} trong ô {cell}? "
            "Chỉ tính vật thể có ít nhất 50% diện tích trong ô; nếu chia đôi đúng biên thì theo ô chứa tâm.",
            options, str(count), "aggregated", cell=cell)
    return scan_targets(scene, "Q4", grouped_objects(scene), build, rejects)


def box_iou(a, b):
    intersection = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection
    return intersection / union if union > 0 else 0.0


def box_text(box):
    return json.dumps([int(round(v)) for v in box], ensure_ascii=False)


def grounding_options(scene, target, rng):
    # Hai nhiễu là vùng giả, không phải hai box còn lại của đúng ba vật thể
    # đã sắp trái->phải. Vì thế thứ hạng tọa độ các lựa chọn không còn tự cho
    # đáp án. Vẫn phải chạy baseline text-only khi mở rộng để kiểm bias khác.
    # Lượng hóa mọi box cùng độ chính xác một pixel. Nếu chỉ box đúng là số
    # nguyên còn nhiễu có ba chữ số lẻ, mô hình có thể đoán từ cách viết số.
    truth_box = [int(round(v)) for v in target["bbox"]]
    w, h = truth_box[2]-truth_box[0], truth_box[3]-truth_box[1]
    if w <= 0 or h <= 0:
        raise SkipSample("Box quá nhỏ để biểu diễn bằng tọa độ pixel nguyên")
    boxes = [truth_box]
    region = cell_box(scene, target["cell"])
    for _ in range(500):
        if len(boxes) == 3:
            break
        # Tất cả phương án phải phù hợp với ô được nhắc trong câu hỏi. Nếu
        # hai nhiễu nằm ngoài ô thì chỉ cần đọc tọa độ là loại được chúng.
        cx, cy = rng.uniform(region[0], region[2]), rng.uniform(region[1], region[3])
        x, y = max(0, min(scene.width-w, cx-w/2)), max(0, min(scene.height-h, cy-h/2))
        proposal = [int(round(x)), int(round(y)), int(round(x+w)), int(round(y+h))]
        poly = [(proposal[0], proposal[1]), (proposal[2], proposal[1]),
                (proposal[2], proposal[3]), (proposal[0], proposal[3])]
        if polygon_area(clip_polygon(poly, region)) / (w*h) < 0.5:
            continue
        if all(box_iou(proposal, b) < 0.5 for b in boxes):
            boxes.append(proposal)
    if len(boxes) != 3:
        raise SkipSample("Không tạo được hai box nhiễu phân biệt rõ")
    options = [box_text(b) for b in boxes]
    rng.shuffle(options)
    return options, box_text(truth_box)


def generate_q5(scene, specs, config, rejects):
    def build(group):
        (cell, cid), objects = group
        require_complete(scene, cid)
        if len(objects) < 2:
            raise SkipSample("Hỏi vật thể thứ hai nên cần ít nhất 2 instance")
        for o in objects:
            require_visible(o)
        ordered = sorted(objects, key=lambda o: o["center"][0])
        if any(b["center"][0]-a["center"][0] <= 1.0 for a, b in zip(ordered, ordered[1:])):
            raise SkipSample("Thứ tự trái/phải mơ hồ do các tâm gần trùng hoành độ")
        target = ordered[1]  # Đúng cả khi chỉ có HAI vật thể; bản cũ dùng [0].
        options, truth = grounding_options(scene, target, rng_for(config, scene, "Q5", (cell, cid)))
        return make_sample(scene, "Q5", cid, json.loads(truth), native_footprint(scene, target), threshold(config, "Q5"),
            f"Trong ô {cell}, {specs[cid]['display_name_vi']} thứ hai theo tâm từ trái sang phải "
            "được bao sát nhất bởi hộp nào? [x1,y1,x2,y2] tính bằng pixel ảnh này. "
            "Chỉ xét vật thể có ít nhất 50% diện tích trong ô; chia đôi đúng biên theo ô chứa tâm.",
            options, truth, "aggregated", cell=cell)
    return scan_targets(scene, "Q5", grouped_objects(scene), build, rejects)


def generate_q6(base_sample):
    """Band cận ngưỡng là điều kiện CHỌN Q6, không phải toàn miền unanswerable.

    rho < 0.5*p0 vẫn dưới ngưỡng, nhưng không thuộc Q6 cận ngưỡng. Không được
    trả sedan ở nhánh else, và không đổi GSD để ép đối tượng vào band.
    """
    if base_sample["kind"] not in ("Q3-HF", "Q4", "Q5"):
        return None
    rho, p0 = base_sample["rho_px"], base_sample["p0_U_px"]
    if not (0.5*p0 <= rho < p0):
        return None
    q = dict(base_sample)
    q.update(kind="Q6", answer=ABSTAIN, answerable_by_sensor=False)
    q["unanswerable_reason"] = (f"Câu gốc {base_sample['kind']}; 0.5*p0_U <= rho_px < p0_U "
        f"({0.5*p0:.6g} <= {rho:.6g} < {p0:.6g} pixel tham chiếu); ngưỡng từ calibration đã khai báo.")
    return q


GENERATORS = (generate_q1, generate_q2, generate_q3_hf, generate_q3_lf, generate_q4, generate_q5)


def build_dataset(root, config, specs, per_kind=1):
    rejects, input_errors, candidates, issues = [], [], defaultdict(list), []
    provenance = []
    for kind in TASKS:
        try:
            threshold(config, kind)
        except SkipSample as exc:
            issues.append({"kind": kind, "reason": str(exc)})
    images_dict = config.get("images")
    if images_dict is None or (isinstance(images_dict, dict) and len(images_dict) == 0):
        img_dir = root / "images"
        if img_dir.exists():
            discovered = {}
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff"):
                for p in sorted(img_dir.glob(ext)):
                    rel = p.relative_to(root).as_posix()
                    discovered[rel] = {}
            images_dict = discovered
        else:
            images_dict = {}

    for image_path, raw_meta in sorted(images_dict.items()):
        try:
            meta = normalize_image_meta(root, image_path, raw_meta, config)
            scene = load_scene(root, image_path, meta)
            unknown = {o["class_id"] for o in scene.objects} - specs.keys()
            if unknown:
                raise InvalidInput(f"Nhãn không có trong physical_sizes: {sorted(unknown)}")
            provenance.append({"image": image_path, "size_px": [scene.width, scene.height],
                "gsd_reference_m": scene.gsd_m, "display_to_reference_scale": scene.scale,
                "source_gsd_m": resolve_gsd(root, meta),
                "native_to_image_scale": meta["native_to_image_scale"],
                "resampling_method": meta.get("resampling_method"),
                "gsd_source": meta.get("gsd_source") or f"Metadata header trong {meta['label_path']}", "gsd_metadata_path": meta.get("gsd_metadata_path"),
                "pixel_scale_source": meta.get("pixel_scale_source"),
                "label_path": meta["label_path"], "label_format": meta["label_format"]})
            scene_candidates = []
            for generator in GENERATORS:
                scene_candidates.extend(generator(scene, specs, config, rejects))
            for q in scene_candidates:
                # Chuyển candidate cận ngưỡng thành Q6, không xuất thêm bản sao
                # giống hệt ở Q3/Q4/Q5 gây trùng mẫu giữa các nhóm.
                q6 = generate_q6(q)
                selected = q6 if q6 is not None else q
                candidates[selected["kind"]].append(selected)
        except (InvalidInput, OSError, KeyError) as exc:
            input_errors.append({"image": image_path, "reason": str(exc)})
    output, seen = [], set()
    for kind in KINDS:
        count = 0
        for q in candidates[kind]:
            key = (q["image_path"], q["question"], q["class_id"])
            if key in seen:
                continue
            if count >= per_kind:
                break
            seen.add(key)
            count += 1
            q["id"] = f"RS-SOLVE-{kind.replace('-', '')}-{count:03d}"
            output.append(q)
    counts = {k: sum(q["kind"] == k for q in output) for k in KINDS}
    report = {"requested_per_kind": per_kind, "selected_counts": counts,
        "complete": all(v == per_kind for v in counts.values()),
        "configuration_issues": issues, "input_errors": input_errors,
        "image_provenance": provenance,
        "rejection_counts": dict(Counter(r["reason"] for r in rejects)),
        "rejections": rejects[:1000], "rejections_truncated": len(rejects) > 1000,
        "calibration_id": config.get("calibration", {}).get("id"),
        "note": "Q2 isolation_flag=not_applicable. Với ảnh giảm mẫu có kiểm soát, gsd_m/rho_px dùng mức lấy mẫu mới; nguồn và phép biến đổi ở image_provenance."}
    return output, report


def write_outputs(output_dir, prefix, samples, report, allow_partial=False):
    output_dir.mkdir(parents=True, exist_ok=True)
    # Luôn ghi báo cáo. Không ghi QA rỗng hoặc thiếu nhóm nếu chưa cho phép;
    # chạy với config còn trống sẽ KHÔNG xóa bộ mẫu cũ của người dùng.
    writable = bool(samples) and (report["complete"] or allow_partial)
    report["qa_written"] = writable
    (output_dir / f"{prefix}_build_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    if not writable:
        return False
    (output_dir / f"{prefix}.json").write_text(json.dumps(samples, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output_dir / f"{prefix}.jsonl").write_text("".join(json.dumps(q, ensure_ascii=False, allow_nan=False) + "\n" for q in samples), encoding="utf-8")
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=BASE_DIR / "build_config.json")
    parser.add_argument("--per-kind", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=BASE_DIR)
    parser.add_argument("--output-prefix", default="mau_7_questions")
    parser.add_argument("--allow-partial", action="store_true", help="Xuất số mẫu hợp lệ ít hơn mục tiêu; không bù bằng nhãn giả")
    args = parser.parse_args(argv)
    if args.per_kind < 1 or not args.output_prefix or Path(args.output_prefix).name != args.output_prefix or any(c in args.output_prefix for c in '/\\:'):
        parser.error("per-kind phải >=1; output-prefix chỉ là tên, không chứa đường dẫn")
    try:
        config = load_json(args.config)
        samples, report = build_dataset(BASE_DIR, config, load_physical_sizes(), args.per_kind)
        written = write_outputs(args.output_dir, args.output_prefix, samples, report, args.allow_partial)
    except (InvalidInput, OSError, KeyError, TypeError) as exc:
        print(f"Không thể chạy: {exc}", file=sys.stderr)
        return 2
    print("Số mẫu hợp lệ:", report["selected_counts"])
    print("Báo cáo:", args.output_dir / f"{args.output_prefix}_build_report.json")
    if not written:
        print("Chưa đủ dữ liệu để xuất QA; các JSON/JSONL cũ được giữ nguyên.")
        return 2
    print("Đã ghi JSON và JSONL, giữ nguyên cấu trúc từng phần tử.")
    return 0


if __name__ == "__main__":
    # Import trong notebook/test không thay stdout; chỉ cấu hình console khi chạy.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
