"""Build a small RS-Solve VQA set from DOTA OBB labels or self-contained fixtures.

The fixture mode demonstrates the schema; only the DOTA mode can yield real-image
candidates. Neither mode estimates a human resolvability threshold or a VLM's
actual visual-token grid.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median

from PIL import Image, ImageDraw


ABSTAIN = "Không thể xác định từ ảnh"
KINDS = ("Q1", "Q2", "Q3-HF", "Q3-LF", "Q4", "Q5", "Q6")
COLORS = ("đỏ", "xanh dương", "xanh lá", "vàng", "trắng", "đen")
CLASS_VI = {
    "small-vehicle": "xe nhỏ", "large-vehicle": "xe lớn", "plane": "máy bay",
    "helicopter": "trực thăng", "ship": "tàu", "bridge": "cầu",
    "dam": "đập", "storage-tank": "bể chứa", "roundabout": "bùng binh",
    "soccer-ball-field": "sân bóng đá", "baseball-diamond": "sân bóng chày",
    "tennis-court": "sân tennis", "basketball-court": "sân bóng rổ",
}
HARD_NEGATIVES = {
    "soccer-ball-field": "baseball-diamond", "bridge": "dam",
    "storage-tank": "roundabout", "tennis-court": "basketball-court",
    "plane": "helicopter",
}
COLOR_RGB = {
    "đỏ": "#cf4f4f", "xanh dương": "#427fc4", "xanh lá": "#4a9663",
    "vàng": "#e4c553", "trắng": "#e8e9dd", "đen": "#263039",
}
GRID_TEXT = "Chia ảnh thành lưới 3×3: cột A–C từ trái sang phải, hàng 1–3 từ trên xuống. "


@dataclass
class Obj:
    id: str
    cls: str
    poly: list[list[float]]
    difficult: bool = False
    color: str | None = None
    color_reviewed: bool = False

    @property
    def center(self) -> tuple[float, float]:
        return (sum(p[0] for p in self.poly) / 4, sum(p[1] for p in self.poly) / 4)

    @property
    def sides(self) -> tuple[float, float]:
        return (math.dist(self.poly[0], self.poly[1]), math.dist(self.poly[1], self.poly[2]))

    @property
    def width_px(self) -> float:
        return min(self.sides)

    @property
    def bbox(self) -> list[int]:
        return [round(min(p[0] for p in self.poly)), round(min(p[1] for p in self.poly)),
                round(max(p[0] for p in self.poly)), round(max(p[1] for p in self.poly))]


@dataclass
class Scene:
    id: str
    image: Path
    width: int
    height: int
    gsd: float
    gsd_source: str
    source: str
    objects: list[Obj]
    absent_verified: list[str] = field(default_factory=list)
    image_ref: str = ""


def rectangle(cx: float, cy: float, long: float, short: float, angle: float = 0) -> list[list[float]]:
    a = math.radians(angle)
    ux, uy = math.cos(a), math.sin(a)
    vx, vy = -uy, ux
    return [[round(cx + sx * long * ux / 2 + sy * short * vx / 2, 3),
             round(cy + sx * long * uy / 2 + sy * short * vy / 2, 3)]
            for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]


def cell_of(obj: Obj, scene: Scene) -> str:
    x, y = obj.center
    col = min(2, max(0, int(x * 3 / scene.width)))
    row = min(2, max(0, int(y * 3 / scene.height)))
    return f"{'ABC'[col]}{row + 1}"


def cell_rect(cell: str, scene: Scene) -> tuple[float, float, float, float]:
    col, row = "ABC".index(cell[0]), int(cell[1]) - 1
    return (col * scene.width / 3, row * scene.height / 3,
            (col + 1) * scene.width / 3, (row + 1) * scene.height / 3)


def polygon_area(poly: list[list[float]]) -> float:
    return abs(sum(poly[i][0] * poly[(i + 1) % len(poly)][1] -
                   poly[(i + 1) % len(poly)][0] * poly[i][1]
                   for i in range(len(poly))) / 2)


def clipped_area(poly: list[list[float]], rect: tuple[float, float, float, float]) -> float:
    """Sutherland-Hodgman clipping against one 3×3 grid cell."""
    xmin, ymin, xmax, ymax = rect
    result = [list(p) for p in poly]
    edges = ((0, xmin, True), (0, xmax, False), (1, ymin, True), (1, ymax, False))
    for axis, bound, lower in edges:
        source, result = result, []
        if not source:
            break
        inside = lambda p: p[axis] >= bound if lower else p[axis] <= bound
        for start, end in zip(source, source[1:] + source[:1]):
            sin, ein = inside(start), inside(end)
            if sin != ein:
                d = end[axis] - start[axis]
                t = (bound - start[axis]) / d if d else 0
                result.append([start[0] + t * (end[0] - start[0]),
                               start[1] + t * (end[1] - start[1])])
            if ein:
                result.append(end)
    return polygon_area(result) if len(result) >= 3 else 0.0


def qualifying_in_cell(scene: Scene, cls: str, cell: str) -> list[Obj]:
    rect = cell_rect(cell, scene)
    return [o for o in scene.objects if o.cls == cls and not o.difficult and
            polygon_area(o.poly) > 0 and
            clipped_area(o.poly, rect) / polygon_area(o.poly) >= 0.5]


def nearest_same_class_px(scene: Scene, target: Obj) -> float | None:
    distances = [math.dist(target.center, o.center) for o in scene.objects
                 if o.id != target.id and o.cls == target.cls and not o.difficult]
    return min(distances) if distances else None


def direction(obj: Obj) -> str | None:
    a, b = obj.sides
    if min(a, b) / max(a, b) > 0.8:
        return None
    p, q = (obj.poly[0], obj.poly[1]) if a >= b else (obj.poly[1], obj.poly[2])
    dx, dy = abs(q[0] - p[0]), abs(q[1] - p[1])
    # Avoid 45-degree bin boundaries: each axis must dominate by at least 25%.
    if max(dx, dy) < 1.25 * min(dx, dy):
        return None
    return "Đông–Tây" if dx > dy else "Bắc–Nam"


def base(scene: Scene, kind: str, target: Obj | None, threshold: dict, question: str,
         choices: list[str], answer: str, *, answerable: bool, extra: dict | None = None) -> dict:
    rho = round(target.width_px, 3) if target else None
    nearest = nearest_same_class_px(scene, target) if target else None
    record = {
        "id": "", "dataset_name": "RS-Solve", "schema_version": "0.1",
        "kind": kind, "source": scene.source, "image_id": scene.id,
        "image_path": scene.image_ref or str(scene.image),
        "image_width": scene.width, "image_height": scene.height,
        "gsd_m_per_px": scene.gsd, "gsd_source": scene.gsd_source,
        "question": GRID_TEXT + question, "choices": choices, "answer": answer,
        "answerable_by_sensor_label": answerable,
        "target_instance_id": target.id if target else None,
        "target_class": target.cls if target else None,
        "target_polygon_xy": target.poly if target else None,
        "target_box_xyxy": target.bbox if target else None,
        "rho_px": rho, "L_m": round(rho * scene.gsd, 4) if rho is not None else None,
        "rho_tok_measured": None, "token_grid_status": "not_measured",
        "nearest_same_class_px": round(nearest, 2) if nearest is not None else None,
        "p0_U_px": threshold["value_px"], "p0_status": threshold["status"],
        "sample_status": "illustrative_only" if scene.source == "synthetic_fixture" else "candidate_needs_visual_QC",
        "notes": [],
    }
    if extra:
        record.update(extra)
    return record


def candidates(scene: Scene, threshold: dict, min_sizes: dict[str, float]) -> dict[str, list[dict]]:
    p0 = threshold["value_px"]
    out: dict[str, list[dict]] = defaultdict(list)
    valid = [o for o in scene.objects if not o.difficult and polygon_area(o.poly) > 0]
    by_class: dict[str, list[Obj]] = defaultdict(list)
    for o in valid:
        by_class[o.cls].append(o)

    # Q1: positive existence, with a single well resolved example preferred.
    for cls, objects in by_class.items():
        large = [o for o in objects if o.width_px >= p0]
        if large:
            target = max(large, key=lambda o: o.width_px)
            rec = base(scene, "Q1", target, threshold,
                       f"Trong toàn ảnh có {CLASS_VI.get(cls, cls)} không?",
                       ["Có", "Không", ABSTAIN], "Có", answerable=True,
                       extra={"q1_group": "iso_candidate" if len(objects) <= 3 else "aggregate",
                              "same_class_count": len(objects)})
            if len(objects) <= 3 and nearest_same_class_px(scene, target) is not None:
                rec["notes"].append("Khoảng cách cô lập theo token cần kiểm tra sau khi đọc lưới token thật.")
            out["Q1"].append(rec)

    # Q2: verified hard negative only. The hypothetical footprint uses an
    # externally supplied conservative class minimum, never the distractor box.
    for present, absent in HARD_NEGATIVES.items():
        if present not in by_class or absent in by_class or absent not in scene.absent_verified:
            continue
        if absent not in min_sizes:
            continue
        hypo_rho = min_sizes[absent] / scene.gsd
        answerable = hypo_rho >= p0
        rec = base(scene, "Q2", None, threshold,
                   f"Trong toàn ảnh có {CLASS_VI.get(absent, absent)} không?",
                   ["Có", "Không", ABSTAIN], "Không" if answerable else ABSTAIN,
                   answerable=answerable,
                   extra={"target_class": absent, "present_distractor_class": present,
                          "rho_px": round(hypo_rho, 3), "L_m": min_sizes[absent],
                          "footprint_basis": "class_min_physical_size", "absence_verified": True})
        out["Q2"].append(rec)

    # Q3-HF and Q6: objective OBB major axis, excluding near-square and
    # near-diagonal cases. Q6 deliberately uses a near-threshold instance.
    for o in valid:
        axis = direction(o)
        if not axis:
            continue
        nearest = nearest_same_class_px(scene, o)
        if nearest is not None and nearest < max(32, 4 * o.width_px):
            continue
        cell = cell_of(o, scene)
        if [item.id for item in qualifying_in_cell(scene, o.cls, cell)] != [o.id]:
            continue
        question = f"Trục dài của {CLASS_VI.get(o.cls, o.cls)} ở ô {cell} gần Bắc–Nam hay Đông–Tây hơn?"
        if o.width_px >= p0:
            out["Q3-HF"].append(base(scene, "Q3-HF", o, threshold, question,
                                      ["Bắc–Nam", "Đông–Tây", ABSTAIN], axis,
                                      answerable=True,
                                      extra={"spatial_frequency": "high", "johnson_level": "orientation",
                                             "target_cell": cell}))
        elif 0.5 * p0 <= o.width_px < p0:
            rec = base(scene, "Q6", o, threshold, question,
                       ["Bắc–Nam", "Đông–Tây", ABSTAIN], ABSTAIN,
                       answerable=False,
                       extra={"underlying_task": "Q3-HF", "spatial_frequency": "high",
                              "johnson_level": "orientation", "target_cell": cell,
                              "unanswerable_reason": "0.5*p0_U <= rho_px < p0_U"})
            rec["notes"].append("Nhãn Q6 chỉ có giá trị nghiên cứu sau pilot người + mô hình.")
            out["Q6"].append(rec)

    # Q3-LF: colors must have been independently reviewed on fine imagery.
    for o in valid:
        if o.color not in COLORS or not o.color_reviewed:
            continue
        cell = cell_of(o, scene)
        if [item.id for item in qualifying_in_cell(scene, o.cls, cell)] != [o.id]:
            continue
        rec = base(scene, "Q3-LF", o, threshold,
                   f"{CLASS_VI.get(o.cls, o.cls).capitalize()} ở ô {cell} có màu chủ đạo nào?",
                   list(COLORS) + [ABSTAIN], o.color, answerable=True,
                   extra={"spatial_frequency": "low", "target_cell": cell,
                          "color_gt_source": "reviewed_fine_image" if scene.source != "synthetic_fixture"
                          else "fixture_generation"})
        out["Q3-LF"].append(rec)

    # Q4: boxes contributing less than half of their area are excluded.
    for cls in by_class:
        for cell in (f"{col}{row}" for row in range(1, 4) for col in "ABC"):
            group = qualifying_in_cell(scene, cls, cell)
            if not group or len(group) > 8:
                continue
            rho = median(o.width_px for o in group)
            n = len(group)
            values = sorted(set((max(0, n - 1), n, n + 1, n + 2)))
            answerable = rho >= p0
            rec = base(scene, "Q4", min(group, key=lambda o: abs(o.width_px - rho)), threshold,
                       f"Có bao nhiêu {CLASS_VI.get(cls, cls)} trong ô {cell}?",
                       [str(x) for x in values] + [ABSTAIN], str(n) if answerable else ABSTAIN,
                       answerable=answerable,
                       extra={"target_cell": cell, "count_gt": n,
                              "target_instance_id": None, "target_polygon_xy": None,
                              "target_box_xyxy": None, "nearest_same_class_px": None,
                              "counted_instance_ids": [o.id for o in group],
                              "rho_px": round(rho, 3), "L_m": round(rho * scene.gsd, 4),
                              "rho_px_aggregation": "median_min_obb_side",
                              "density_count_in_cell": n, "spatial_frequency": "high"})
            out["Q4"].append(rec)

    # Q5: order must be well separated; answer is one of three real GT boxes.
    for cls in by_class:
        for cell in (f"{col}{row}" for row in range(1, 4) for col in "ABC"):
            group = sorted(qualifying_in_cell(scene, cls, cell), key=lambda o: o.center[0])
            if len(group) < 3:
                continue
            triple = group[:3]
            gaps = [triple[i + 1].center[0] - triple[i].center[0] for i in (0, 1)]
            if min(gaps) < max(8, max(o.width_px for o in triple)):
                continue
            target = triple[1]
            answerable = target.width_px >= p0
            boxes = {letter: o.bbox for letter, o in zip("ABC", triple)}
            rec = base(scene, "Q5", target, threshold,
                       f"Trong ô {cell}, hộp bao nào ứng với {CLASS_VI.get(cls, cls)} thứ hai từ trái sang?",
                       [f"{key}: {value}" for key, value in boxes.items()] + [ABSTAIN],
                       f"B: {boxes['B']}" if answerable else ABSTAIN, answerable=answerable,
                       extra={"target_cell": cell, "answer_box_xyxy": target.bbox,
                              "candidate_boxes_xyxy": boxes,
                              "ordered_instance_ids": [o.id for o in triple],
                              "spatial_frequency": "high"})
            out["Q5"].append(rec)
    return out


def read_dota(root: Path, review: dict) -> list[Scene]:
    labels = root / "labelTxt"
    images = root / "images"
    if not labels.is_dir() or not images.is_dir():
        raise ValueError("DOTA root phải chứa images/ và labelTxt/ của cùng phiên bản.")
    scenes = []
    for label in sorted(labels.glob("*.txt")):
        image = next((images / f"{label.stem}{ext}" for ext in (".png", ".jpg", ".tif", ".tiff")
                      if (images / f"{label.stem}{ext}").is_file()), None)
        if image is None:
            continue
        lines = label.read_text(encoding="utf-8-sig").splitlines()
        gsd = None
        objects = []
        info = review.get(label.stem, {})
        for index, line in enumerate(lines):
            stripped = line.strip()
            if stripped.lower().startswith("gsd:"):
                try:
                    gsd = float(stripped.split(":", 1)[1].strip())
                except ValueError:
                    pass
                continue
            parts = stripped.split()
            if len(parts) < 10:
                continue
            try:
                poly = [[float(parts[i]), float(parts[i + 1])] for i in (0, 2, 4, 6)]
            except ValueError:
                continue
            object_id = str(len(objects))
            color = info.get("colors", {}).get(object_id)
            objects.append(Obj(object_id, parts[8], poly, parts[9] == "1",
                               color=color, color_reviewed=color in COLORS))
        if gsd is None or not math.isfinite(gsd) or gsd <= 0:
            continue
        with Image.open(image) as im:
            width, height = im.size
        scenes.append(Scene(label.stem, image, width, height, gsd,
                            "DOTA annotation header", "DOTA", objects,
                            info.get("absent_classes", []),
                            str(image.resolve())))
    return scenes


def demo_scenes(outdir: Path) -> list[Scene]:
    image_dir = outdir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    landmarks = [
        ("soccer-ball-field", "baseball-diamond", (440, 90, 90, 58)),
        ("bridge", "dam", (430, 95, 95, 25)),
        ("storage-tank", "roundabout", (430, 95, 64, 64)),
        ("tennis-court", "basketball-court", (430, 95, 80, 48)),
        ("plane", "helicopter", (430, 95, 80, 35)),
    ]
    scenes = []
    for i, (present, absent, (lx, ly, ll, ls)) in enumerate(landmarks, 1):
        objects = [Obj("0", present, rectangle(lx, ly, ll, ls)),
                   Obj("1", "small-vehicle", rectangle(85, 85, 40, 14, 0 if i % 2 else 90)),
                   Obj("2", "small-vehicle", rectangle(255, 90, 34 if i <= 3 else 10,
                                                          13 if i <= 3 else 4.5),
                       color=COLORS[(i - 1) % 6], color_reviewed=True)]
        for j, x in enumerate((205, 255, 305), 3):
            objects.append(Obj(str(j), "small-vehicle", rectangle(x, 255, 30, 11)))
        objects.append(Obj("6", "small-vehicle", rectangle(425, 425, 11, 4.5)))
        image = image_dir / f"demo_{i:02d}.png"
        draw_fixture(image, objects, i)
        scenes.append(Scene(f"demo_{i:02d}", image, 512, 512, 0.3,
                            "synthetic fixture assumption", "synthetic_fixture", objects,
                            [absent], str(Path("images") / image.name)))
    return scenes


def draw_fixture(path: Path, objects: list[Obj], seed: int) -> None:
    """Draw a labelled schematic, intentionally unlike actual satellite imagery."""
    im = Image.new("RGB", (512, 512), "#879482")
    draw = ImageDraw.Draw(im)
    for k in range(7):
        y = 30 + k * 75 + seed * 3
        draw.line((0, y, 512, y + 45), fill="#98a599", width=5)
    draw.rectangle((175, 175, 345, 335), fill="#666b68")
    for o in objects:
        color = COLOR_RGB.get(o.color, "#e0ddd0" if o.cls == "small-vehicle" else "#b1c3aa")
        x0, y0, x1, y1 = o.bbox
        if o.cls == "bridge":
            draw.rectangle((x0 - 15, y0 - 18, x1 + 15, y1 + 18), fill="#587c8d")
            draw.polygon([tuple(p) for p in o.poly], fill="#b5b5aa", outline="#e8e9dd", width=2)
        elif o.cls == "storage-tank":
            draw.ellipse((x0, y0, x1, y1), fill="#bdc5bb", outline="#4e615b", width=3)
            draw.ellipse((x0 + 8, y0 + 8, x1 - 8, y1 - 8), outline="#818f86", width=2)
        elif o.cls == "plane":
            cx, cy = o.center
            draw.polygon([(cx - 40, cy - 4), (cx - 8, cy - 4), (cx - 6, cy - 17),
                          (cx + 4, cy - 17), (cx + 9, cy - 4), (cx + 35, cy - 4),
                          (cx + 42, cy), (cx + 35, cy + 4), (cx + 9, cy + 4),
                          (cx + 4, cy + 17), (cx - 6, cy + 17), (cx - 8, cy + 4),
                          (cx - 40, cy + 4)], fill="#d7d9d4", outline="#34423e")
        else:
            draw.polygon([tuple(p) for p in o.poly], fill=color, outline="#34423e", width=1)
            if o.cls == "soccer-ball-field":
                draw.rectangle((x0 + 3, y0 + 3, x1 - 3, y1 - 3), outline="#f2f1d9", width=2)
                draw.line(((x0 + x1) / 2, y0 + 3, (x0 + x1) / 2, y1 - 3), fill="#f2f1d9", width=2)
                draw.ellipse(((x0 + x1) / 2 - 8, (y0 + y1) / 2 - 8,
                              (x0 + x1) / 2 + 8, (y0 + y1) / 2 + 8), outline="#f2f1d9", width=1)
            elif o.cls == "tennis-court":
                draw.rectangle((x0 + 4, y0 + 4, x1 - 4, y1 - 4), outline="#f2f1d9", width=2)
                draw.line(((x0 + x1) / 2, y0 + 4, (x0 + x1) / 2, y1 - 4), fill="#f2f1d9", width=2)
                draw.line((x0 + 4, (y0 + y1) / 2, x1 - 4, (y0 + y1) / 2), fill="#f2f1d9", width=1)
    for x in (512 / 3, 1024 / 3):
        draw.line((x, 0, x, 512), fill="#ffffff", width=1)
    for y in (512 / 3, 1024 / 3):
        draw.line((0, y, 512, y), fill="#ffffff", width=1)
    for row, y in enumerate((5, 175, 346), 1):
        for col, x in zip("ABC", (5, 175, 346)):
            draw.text((x, y), f"{col}{row}", fill="#ffffff", stroke_width=1,
                      stroke_fill="#263039")
    im.save(path)


def add_token_proxies(records: list[dict], views: list[dict]) -> None:
    """Attach a clearly named resize-only approximation, never a measured grid."""
    names = set()
    for view in views:
        name = view["name"]
        cap, patch = float(view["max_side_px"]), float(view["effective_patch_px"])
        if not name or name in names or not math.isfinite(cap) or not math.isfinite(patch) or cap <= 0 or patch <= 0:
            raise ValueError("token-config có tên trùng hoặc cap/patch không hợp lệ")
        names.add(name)
    for rec in records:
        rec["rho_tok_proxy_resize_only"] = {
            view["name"]: round(rec["rho_px"] *
                                min(1.0, float(view["max_side_px"]) /
                                    max(rec["image_width"], rec["image_height"])) /
                                float(view["effective_patch_px"]), 5)
            for view in views
        }


def choose_balanced(scenes: list[Scene], threshold: dict, min_sizes: dict[str, float], n: int) -> list[dict]:
    pooled: dict[str, dict[str, list[dict]]] = {}
    for scene in scenes:
        pooled[scene.id] = candidates(scene, threshold, min_sizes)
        pooled[scene.id]["Q4"].sort(key=lambda r: (-r["count_gt"], r["target_class"], r["target_cell"]))
    result = []
    for kind in KINDS:
        selected = []
        # First pass spreads records across images; second pass fills remainder.
        for scene in scenes:
            pool = pooled[scene.id].get(kind, [])
            if pool and len(selected) < n:
                selected.append(pool.pop(0))
        if len(selected) < n:
            for scene in scenes:
                pool = pooled[scene.id].get(kind, [])
                while pool and len(selected) < n:
                    selected.append(pool.pop(0))
        if len(selected) != n:
            raise ValueError(f"Thiếu {kind}: tìm được {len(selected)}/{n}. Bổ sung ảnh/nhãn/review hoặc giảm --per-type.")
        for i, rec in enumerate(selected, 1):
            rec["id"] = f"RS-SOLVE-{kind.replace('-', '')}-{i:03d}"
        result.extend(selected)
    return result


def validate(records: list[dict], expected: int) -> dict:
    counts = Counter(r["kind"] for r in records)
    if counts != Counter({kind: expected for kind in KINDS}):
        raise ValueError(f"Sai phân bổ loại câu hỏi: {counts}")
    if len({r["id"] for r in records}) != len(records):
        raise ValueError("Trùng ID")
    for r in records:
        if r["answer"] not in r["choices"]:
            raise ValueError(f"Đáp án không nằm trong lựa chọn: {r['id']}")
        if r["gsd_m_per_px"] <= 0 or r["rho_px"] is None or r["rho_px"] <= 0:
            raise ValueError(f"Footprint/GSD không hợp lệ: {r['id']}")
        if not math.isclose(r["rho_px"] * r["gsd_m_per_px"], r["L_m"], abs_tol=0.002):
            raise ValueError(f"L/g không khớp: {r['id']}")
        if r["kind"] == "Q6" and not (0.5 * r["p0_U_px"] <= r["rho_px"] < r["p0_U_px"]
                                          and r["answer"] == ABSTAIN):
            raise ValueError(f"Q6 không ở gần dưới ngưỡng: {r['id']}")
        if r["kind"] == "Q2" and not r["absence_verified"]:
            raise ValueError(f"Q2 chưa kiểm phủ định: {r['id']}")
        if r["kind"] == "Q3-LF" and r["color_gt_source"] not in ("reviewed_fine_image", "fixture_generation"):
            raise ValueError(f"Q3-LF thiếu màu đã duyệt: {r['id']}")
    return {"total": len(records), "by_kind": dict(counts),
            "source_status": dict(Counter(r["sample_status"] for r in records)),
            "all_token_grids_measured": all(r["rho_tok_measured"] is not None for r in records)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("demo", "dota"), default="demo")
    ap.add_argument("--dota-root", type=Path, help="Directory containing images/ and labelTxt/")
    ap.add_argument("--review", type=Path, help="JSON manual QC: per-image absent_classes and colors")
    ap.add_argument("--threshold", type=Path, help="JSON calibrated p0_U: value_px, status")
    ap.add_argument("--min-sizes", type=Path, help="JSON class -> conservative L_min (metres)")
    ap.add_argument("--token-config", type=Path,
                    help="Optional JSON with resize-only proxy views; NOT measured visual-token grids")
    ap.add_argument("--output", type=Path, default=Path(__file__).parent / "demo_output")
    ap.add_argument("--per-type", type=int, default=5)
    args = ap.parse_args()
    if args.per_type < 1:
        ap.error("--per-type phải >= 1")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.mode == "demo":
        threshold = {"value_px": 6.0, "status": "illustrative_unvalidated"}
        min_sizes = {v: 5.0 for v in HARD_NEGATIVES.values()}
        scenes = demo_scenes(args.output)
    else:
        if not (args.dota_root and args.review and args.threshold and args.min_sizes):
            ap.error("DOTA mode cần --dota-root, --review, --threshold, --min-sizes")
        threshold = json.loads(args.threshold.read_text(encoding="utf-8"))
        if threshold.get("status") != "pilot_human_and_models_calibrated":
            ap.error("Ngưỡng DOTA phải được hiệu chỉnh từ pilot người + mô hình.")
        min_sizes = json.loads(args.min_sizes.read_text(encoding="utf-8"))
        review = json.loads(args.review.read_text(encoding="utf-8"))
        scenes = read_dota(args.dota_root, review)
    if not math.isfinite(float(threshold["value_px"])) or threshold["value_px"] <= 0:
        ap.error("value_px không hợp lệ")
    records = choose_balanced(scenes, threshold, min_sizes, args.per_type)
    if args.token_config:
        views = json.loads(args.token_config.read_text(encoding="utf-8"))
    elif args.mode == "demo":
        views = [{"name": "illustrative_P32_cap256", "max_side_px": 256,
                  "effective_patch_px": 32},
                 {"name": "illustrative_P32_cap512", "max_side_px": 512,
                  "effective_patch_px": 32}]
    else:
        views = []
    add_token_proxies(records, views)
    summary = validate(records, args.per_type)
    output = args.output / "rs_solve_qa.jsonl"
    with output.open("w", encoding="utf-8", newline="\n") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                                encoding="utf-8")
    print(json.dumps({"output": str(output), **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
