"""Trục lõi dùng chung (Core Generators) cho toàn bộ 5 dataset của RS-Solve.

Chứa các cấu trúc dữ liệu, thuật toán hình học viễn thám, tiêu chuẩn Johnson,
cơ chế tính footprint và các hàm sinh câu hỏi từ Q1 đến Q6.
"""
from __future__ import annotations
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
from PIL import Image

ABSTAIN = "Không thể xác định từ ảnh"
KINDS = ("Q1", "Q2", "Q3-HF", "Q3-LF", "Q4", "Q5", "Q6")
GRID_PREFIX = "Chia ảnh thành lưới 3x3: cột A-C từ trái sang phải, hàng 1-3 từ trên xuống. "

CONFUSING_PAIRS = (
    ("soccer_ball_field", "baseball_diamond"),
    ("baseball_diamond", "soccer_ball_field"),
    ("bridge", "dam"),
    ("dam", "bridge"),
    ("storage_tank", "roundabout"),
    ("roundabout", "storage_tank"),
    ("tennis_court", "basketball_court"),
    ("basketball_court", "tennis_court"),
    ("helicopter", "airplane"),
    ("airplane", "helicopter"),
)

COLOR_CHOICES = (
    "Màu đỏ", "Màu xanh dương", "Màu xanh lá cây", "Màu vàng", "Màu trắng", "Màu đen / xám"
)

ORIENTATION_CHOICES = (
    "Từ trên trái xuống dưới phải",
    "Từ trên phải xuống dưới trái",
)

DEFAULT_CALIBRATION = {
    "Q1": 6.0,
    "Q2": 8.0,
    "Q3-HF": 12.0,
    "Q3-LF": 4.0,
    "Q4": 25.0,
    "Q5": 5.0,
}

TASKS = {
    "Q1": "detection",
    "Q2": "hard_negative",
    "Q3-HF": "orientation",
    "Q3-LF": "color_lf",
    "Q4": "counting",
    "Q5": "grounding"
}

DEFAULT_DOTA_CLASSES = [
    "plane", "ship", "storage-tank", "baseball-diamond", "tennis-court",
    "basketball-court", "ground-track-field", "harbor", "bridge", "large-vehicle",
    "small-vehicle", "helicopter", "roundabout", "soccer-ball-field",
    "swimming-pool", "container-crane", "airport", "helipad"
]


class InvalidInput(Exception):
    """Lỗi dữ liệu đầu vào hoặc siêu dữ liệu không hợp lệ."""
    pass


class SkipSample(Exception):
    """Bỏ qua mẫu do không thỏa mãn tiêu chí kiểm soát chất lượng."""
    pass


@dataclass
class Scene:
    image_path: str
    img_w: int
    img_h: int
    gsd_m: float
    scale: float
    meta: dict
    objects: list[dict]
    pixels: object = None
    gsd_source: str = ""


def polygon_area(points):
    """Diện tích đa giác lồi hoặc đơn theo công thức Shoelace."""
    return 0.5 * abs(sum(
        points[i][0] * points[(i+1) % len(points)][1] -
        points[(i+1) % len(points)][0] * points[i][1]
        for i in range(len(points))
    ))


def clip_polygon(subject, rect):
    """Cắt đa giác theo hình chữ nhật bằng thuật toán Sutherland-Hodgman."""
    output = list(subject)
    for edge_index, (axis, bound, greater) in enumerate([
        (0, rect[0], True), (1, rect[1], True),
        (0, rect[2], False), (1, rect[3], False)
    ]):
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
    """Hình chữ nhật bao nhỏ nhất của tứ giác; trả về minor, major, góc quay."""
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


def cell_box(scene, cell):
    """Tọa độ pixel [xmin, ymin, xmax, ymax] của ô lưới 3x3."""
    col = "ABC".index(cell[0])
    row = int(cell[1]) - 1
    return (
        col * scene.img_w / 3.0,
        row * scene.img_h / 3.0,
        (col + 1) * scene.img_w / 3.0,
        (row + 1) * scene.img_h / 3.0,
    )


def in_cell(scene, obj):
    """Kiểm tra vật thể có ít nhất 50% diện tích nằm trong ô lưới."""
    clipped = clip_polygon(obj["pts"], cell_box(scene, obj["cell"]))
    return polygon_area(clipped) / obj["area_px2"] >= 0.5 - 1e-10


def require_visible(obj):
    """Kiểm tra vật thể không bị che khuất và nhìn thấy ít nhất 50%."""
    if obj["difficult"] or obj["visible_fraction"] < 0.5:
        raise SkipSample("Instance difficult hoặc nhìn thấy dưới 50% diện tích")


def require_isolated(scene, obj):
    """Kiểm tra vật thể có đứng đơn lẻ (cô lập) hay không."""
    same = [o for o in scene.objects if o["class_id"] == obj["class_id"]]
    rev = scene.meta.get("isolation_reviews", {}).get(str(obj["obj_idx"])) or \
          scene.meta.get("token_isolation_reviews", {}).get(str(obj["obj_idx"])) or {}
    
    dist_val = rev.get("nearest_same_class_tokens")
    has_rev = has_source(rev.get("source")) and (
        rev.get("isolated") is True or
        (isinstance(dist_val, (int, float)) and dist_val >= 4.0)
    )

    if len(same) > 3 and not has_rev:
        raise SkipSample("Quá 3 instance cùng lớp trong khung ảnh")

    if not has_rev:
        center = obj["center"]
        min_dist = min(
            (math.hypot(o["center"][0] - center[0], o["center"][1] - center[1])
             for o in scene.objects if o is not obj),
            default=float("inf")
        )
        if min_dist < 4.0 * (14.0 * scene.scale):
            raise SkipSample("Khoảng cách láng giềng nhỏ hơn 4 token")


def require_unique_target(scene, obj):
    """Kiểm tra đối tượng là duy nhất trong ô được hỏi."""
    require_visible(obj)
    require_isolated(scene, obj)
    if not in_cell(scene, obj):
        raise SkipSample("Dưới 50% diện tích target nằm trong ô được hỏi")
    if sum(o["class_id"] == obj["class_id"] and o["cell"] == obj["cell"] for o in scene.objects) != 1:
        raise SkipSample("Câu hỏi theo lớp và ô chưa xác định duy nhất target")


def require_complete(scene, class_id):
    """Kiểm tra lớp đã được gán nhãn đầy đủ trên ảnh nguồn."""
    if class_id not in scene.meta.get("complete_classes", []):
        raise SkipSample(f"Lớp {class_id} chưa được đánh dấu gán nhãn đầy đủ trên ảnh")


def has_source(source):
    """Kiểm tra nguồn siêu dữ liệu có hợp lệ và đáng tin cậy."""
    return isinstance(source, str) and len(source.strip()) >= 5 and source.strip().lower() != "unknown"


def threshold(config, kind):
    """Lấy ngưỡng nhận thức tối thiểu Johnson p0_U theo calibration."""
    cal = config.get("calibration", {})
    t_px = cal.get("thresholds_px", {})
    task_name = TASKS.get(kind)
    if task_name and task_name in t_px:
        return float(t_px[task_name])
    if kind in t_px:
        return float(t_px[kind])
    if kind in cal and isinstance(cal[kind], dict) and "p0_U_px" in cal[kind]:
        return float(cal[kind]["p0_U_px"])
    if kind in DEFAULT_CALIBRATION:
        return float(DEFAULT_CALIBRATION[kind])
    raise InvalidInput(f"Thiếu calibration cho {kind}")


def native_footprint(scene, obj):
    """Kích thước footprint cảm biến tính bằng pixel tham chiếu."""
    return obj["minor_len_px"] / scene.scale


def rng_for(config, scene, kind, identity):
    """Tạo bộ sinh số ngẫu nhiên có tính tất định và truy vết được."""
    text = f"{config.get('seed', 42)}|{scene.image_path}|{kind}|{identity}"
    return random.Random(int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big"))


def make_sample(scene, kind, cid, box, rho, p0, question, options, truth, flag, *, cell=None, L_m=None):
    """Khởi tạo một mẫu câu hỏi chuẩn của RS-Solve.
    
    Quy tắc:
    - Chỉ gắn GRID_PREFIX khi câu hỏi có định vị ô (cell is not None).
    - answerable = rho >= p0. Nếu không thỏa mãn, đáp án bắt buộc là ABSTAIN.
    - rho_tok mặc định là None (chỉ đo động khi chạy inference trên từng mô hình).
    """
    if truth not in options or ABSTAIN in options or len(set(options)) != len(options):
        raise InvalidInput("Lựa chọn không hợp lệ hoặc thiếu đáp án sự thật")
    answerable = rho >= p0
    prompt = (GRID_PREFIX + question) if cell is not None else question
    q = {
        "id": "", "kind": kind, "image_path": scene.image_path, "gsd_m": scene.gsd_m,
        "question": prompt, "choices": list(options) + [ABSTAIN],
        "answer": truth if answerable else ABSTAIN, "class_id": cid,
        "target_box_xyxy": None if box is None else [round(v, 3) for v in box],
        "L_m": rho * scene.gsd_m if L_m is None else L_m,
        "rho_px": rho, "p0_U_px": p0, "answerable_by_sensor": answerable,
        "rho_tok": None, "isolation_flag": flag
    }
    if cell is not None:
        q["target_cell"] = cell
    return q


def scan_targets(scene, kind, targets, builder, rejects):
    """Duyệt danh sách mục tiêu và ghi nhận lý do bỏ qua nếu không thỏa mãn."""
    result = []
    for target in targets:
        try:
            result.append(builder(target))
        except SkipSample as exc:
            rejects.append({
                "image_path": scene.image_path,
                "kind": kind,
                "target": str(target),
                "reason": str(exc)
            })
    return result


def orientation_answer(obj):
    """Xác định hướng trục xoay OBB theo 2 đường chéo ảnh."""
    angle = obj["angle_deg"] % 180
    if 25 <= angle <= 65:
        return ORIENTATION_CHOICES[0]  # Từ trên trái xuống dưới phải
    if 115 <= angle <= 155:
        return ORIENTATION_CHOICES[1]  # Từ trên phải xuống dưới trái
    raise SkipSample(f"Góc {angle:.1f}° quá gần trục ngang/dọc/ranh giới 90°")


def dominant_color(scene, obj):
    """Xác định màu chủ đạo của đối tượng."""
    rev = scene.meta.get("color_reviews", {}).get(str(obj["obj_idx"]), {})
    status = rev.get("status")
    if status in ("verified", "reviewed") and has_source(rev.get("source")):
        color = rev.get("value") or rev.get("color")
        if color in COLOR_CHOICES:
            return color
        raise InvalidInput(f"Màu review không hợp lệ: {color}")
    raise SkipSample("Chưa có review màu cho instance này")


def grouped_objects(scene):
    """Gom nhóm đối tượng theo từng ô lưới 3x3 và theo lớp."""
    groups = defaultdict(list)
    for o in scene.objects:
        if in_cell(scene, o):
            groups[(o["cell"], o["class_id"])].append(o)
    return list(groups.items())


def count_options(count, rng):
    """Sinh các phương án lựa chọn cho câu hỏi đếm (Q4)."""
    low = rng.randrange(3)
    nums = [count] + rng.sample(range(count), low)
    nums += rng.sample(range(count+1, max(count+5, 2*count+2)), 2-low)
    options = list(map(str, nums))
    rng.shuffle(options)
    return options


def box_iou(a, b):
    """Tính Intersection over Union (IoU) giữa 2 hộp bao."""
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection
    return intersection / union if union > 0 else 0.0


def box_text(box):
    return json.dumps([int(round(v)) for v in box], ensure_ascii=False)


def grounding_options(scene, target, rng):
    """Sinh các hộp bao giả (distractor boxes) cho câu hỏi định vị Q5."""
    truth_box = [int(round(v)) for v in target["bbox"]]
    w, h = truth_box[2] - truth_box[0], truth_box[3] - truth_box[1]
    if w <= 0 or h <= 0:
        raise SkipSample("Box quá nhỏ để biểu diễn bằng tọa độ pixel nguyên")
    boxes = [truth_box]
    region = cell_box(scene, target["cell"])
    for _ in range(500):
        if len(boxes) == 3:
            break
        cand = [
            int(round(rng.uniform(region[0], region[2] - w))),
            int(round(rng.uniform(region[1], region[3] - h))),
        ]
        cand.extend([cand[0] + w, cand[1] + h])
        if all(box_iou(cand, b) <= 0.1 for b in boxes) and all(box_iou(cand, o["bbox"]) <= 0.2 for o in scene.objects):
            boxes.append(cand)
    if len(boxes) < 3:
        raise SkipSample("Không sinh đủ 2 distractor hợp lệ trong cùng ô")
    rng.shuffle(boxes)
    return [box_text(b) for b in boxes], box_text(truth_box)


# ==============================================================================
# BỘ 6 GENERATOR CÂU HỎI (Q1 ĐẾN Q6)
# ==============================================================================

def generate_q1(scene, specs, config, rejects):
    """Q1: Phát hiện sự hiện diện của đối tượng (Detection)."""
    def build(o):
        require_visible(o)
        require_isolated(scene, o)
        cid = o["class_id"]
        if cid not in specs:
            raise SkipSample(f"Class {cid} không có trong specs")
        return make_sample(
            scene, "Q1", cid, o["bbox"], native_footprint(scene, o), threshold(config, "Q1"),
            f"Trong toàn ảnh có {specs[cid]['display_name_vi']} ({cid}) không?",
            ["Có", "Không"], "Có", "isolated"
        )
    largest = {}
    for o in scene.objects:
        if o["class_id"] not in largest or o["minor_len_px"] > largest[o["class_id"]]["minor_len_px"]:
            largest[o["class_id"]] = o
    return scan_targets(scene, "Q1", list(largest.values()), build, rejects)


def generate_q2(scene, specs, config, rejects):
    """Q2: Phủ định gây nhầm lẫn (Hard Negative Existence)."""
    present = {o["class_id"] for o in scene.objects}
    pairs = [(a, p) for a, p in CONFUSING_PAIRS if a not in present and p in present]

    def build(pair):
        a, p = pair
        if a not in specs:
            raise SkipSample(f"Class {a} không có trong specs")
        review = scene.meta.get("absence_reviews", {}).get(a, {})
        if not (review.get("absent") is True and has_source(review.get("source"))):
            if a not in scene.meta.get("source_classes", []):
                raise SkipSample(f"Nguồn không gán nhãn {a}; cần review vắng mặt")
            require_complete(scene, a)
        prior = config.get("class_prior_approvals", {}).get(a, {})
        has_config_prior = (prior.get("status") == "reviewed" and has_source(prior.get("source")))
        has_spec_prior = (has_source(specs[a].get("standard_source")) and
                          isinstance(specs[a].get("L_min_m"), (int, float)) and specs[a]["L_min_m"] > 0)
        if not (has_config_prior or has_spec_prior):
            raise SkipSample(f"L_min của {a} chưa được duyệt cho Q2")
        L_m = specs[a]["L_min_m"]
        q = make_sample(
            scene, "Q2", a, None, L_m / scene.gsd_m, threshold(config, "Q2"),
            f"Trong toàn ảnh có {specs[a]['display_name_vi']} ({a}) không?",
            ["Có", "Không"], "Không", "not_applicable", L_m=L_m
        )
        q["confused_present_class"] = p
        return q
    return scan_targets(scene, "Q2", pairs, build, rejects)


def generate_q3_hf(scene, specs, config, rejects):
    """Q3-HF: Hướng trục xoay OBB của đối tượng (Orientation)."""
    def build(o):
        require_unique_target(scene, o)
        if o["minor_len_px"] / o["major_len_px"] > 0.8:
            raise SkipSample("Đối tượng gần vuông (w/h > 0.8), trục không xác định")
        cid = o["class_id"]
        if cid not in specs:
            raise SkipSample(f"Class {cid} không có trong specs")
        truth = orientation_answer(o)
        rng = rng_for(config, scene, "Q3-HF", o["obj_idx"])
        options = list(ORIENTATION_CHOICES)
        rng.shuffle(options)
        return make_sample(
            scene, "Q3-HF", cid, o["bbox"], native_footprint(scene, o), threshold(config, "Q3-HF"),
            f"Trục dài của {specs[cid]['display_name_vi']} ở ô {o['cell']} gần với đường chéo nào của ảnh?",
            options, truth, "isolated", cell=o["cell"]
        )
    return scan_targets(scene, "Q3-HF", scene.objects, build, rejects)


def generate_q3_lf(scene, specs, config, rejects):
    """Q3-LF: Màu sắc đặc trưng của đối tượng (Low Frequency / Color)."""
    def build(o):
        require_unique_target(scene, o)
        cid = o["class_id"]
        if cid not in specs:
            raise SkipSample(f"Class {cid} không có trong specs")
        color = dominant_color(scene, o)
        rng = rng_for(config, scene, "Q3-LF", o["obj_idx"])
        options = [color] + rng.sample([c for c in COLOR_CHOICES if c != color], 5)
        rng.shuffle(options)
        return make_sample(
            scene, "Q3-LF", cid, o["bbox"], native_footprint(scene, o), threshold(config, "Q3-LF"),
            f"{specs[cid]['display_name_vi']} ở ô {o['cell']} có màu chủ đạo nào?",
            options, color, "isolated", cell=o["cell"]
        )
    return scan_targets(scene, "Q3-LF", scene.objects, build, rejects)


def generate_q4(scene, specs, config, rejects):
    """Q4: Đếm số lượng đối tượng trong ô lưới (Counting)."""
    def build(group):
        (cell, cid), objects = group
        if cid not in specs:
            raise SkipSample(f"Class {cid} không có trong specs")
        require_complete(scene, cid)
        if len(objects) < 2:
            raise SkipSample("Cần ít nhất 2 instance để sinh Q4")
        for o in objects:
            require_visible(o)
        count = len(objects)
        rho = statistics.median(native_footprint(scene, o) for o in objects)
        union = [
            min(o["bbox"][0] for o in objects), min(o["bbox"][1] for o in objects),
            max(o["bbox"][2] for o in objects), max(o["bbox"][3] for o in objects)
        ]
        options = count_options(count, rng_for(config, scene, "Q4", (cell, cid)))
        return make_sample(
            scene, "Q4", cid, union, rho, threshold(config, "Q4"),
            f"Có chính xác bao nhiêu {specs[cid]['display_name_vi']} trong ô {cell}? "
            "Chỉ tính vật thể có ít nhất 50% diện tích trong ô; nếu chia đôi đúng biên thì theo ô chứa tâm.",
            options, str(count), "aggregated", cell=cell
        )
    return scan_targets(scene, "Q4", grouped_objects(scene), build, rejects)


def generate_q5(scene, specs, config, rejects):
    """Q5: Định vị hộp bao đối tượng theo mô tả không gian (Grounding)."""
    def build(group):
        (cell, cid), objects = group
        if cid not in specs:
            raise SkipSample(f"Class {cid} không có trong specs")
        require_complete(scene, cid)
        if len(objects) < 2:
            raise SkipSample("Hỏi vật thể thứ hai nên cần ít nhất 2 instance")
        for o in objects:
            require_visible(o)
        ordered = sorted(objects, key=lambda o: o["center"][0])
        if any(b["center"][0] - a["center"][0] <= 1.0 for a, b in zip(ordered, ordered[1:])):
            raise SkipSample("Thứ tự trái/phải mơ hồ do các tâm gần trùng hoành độ")
        target = ordered[1]
        options, truth = grounding_options(scene, target, rng_for(config, scene, "Q5", (cell, cid)))
        return make_sample(
            scene, "Q5", cid, json.loads(truth), native_footprint(scene, target), threshold(config, "Q5"),
            f"Trong ô {cell}, {specs[cid]['display_name_vi']} thứ hai theo tâm từ trái sang phải "
            "được bao sát nhất bởi hộp nào? [x1,y1,x2,y2] tính bằng pixel ảnh này. "
            "Chỉ xét vật thể có ít nhất 50% diện tích trong ô; chia đôi đúng biên theo ô chứa tâm.",
            options, truth, "aggregated", cell=cell
        )
    return scan_targets(scene, "Q5", grouped_objects(scene), build, rejects)


def generate_q6(base_sample):
    """Q6: Thử thách từ chối do mờ hoặc dưới ngưỡng cảm biến (Abstention)."""
    if base_sample["kind"] not in ("Q3-HF", "Q4", "Q5"):
        return None
    rho, p0 = base_sample["rho_px"], base_sample["p0_U_px"]
    if not (0.5 * p0 <= rho < p0):
        return None
    q = dict(base_sample)
    q.update(kind="Q6", answer=ABSTAIN, answerable_by_sensor=False)
    q["unanswerable_reason"] = (
        f"Câu gốc {base_sample['kind']}; 0.5*p0_U <= rho_px < p0_U "
        f"({0.5*p0:.6g} <= {rho:.6g} < {p0:.6g} pixel tham chiếu); ngưỡng từ calibration đã khai báo."
    )
    return q


GENERATORS = (generate_q1, generate_q2, generate_q3_hf, generate_q3_lf, generate_q4, generate_q5)
