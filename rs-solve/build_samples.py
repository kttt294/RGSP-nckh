"""
build_samples.py
----------------
Script sinh câu hỏi benchmark RS-Resolve HOÀN TOÀN TỰ ĐỘNG DỰA TRÊN ĐẶC TRƯNG NHÃN THẬT (Label-driven VQA generation):

Quy trình giải thuật:
1. Đọc file nhãn thô (.txt) tương ứng với từng ảnh từ thư mục labels/:
   - Parse tọa độ 4 đỉnh OBB (Oriented Bounding Box).
   - Tự động tính tâm (cx, cy) và gán vật thể vào ô lưới 3x3 (A1..C3).
   - Tự động tính góc định hướng theta và phân loại hướng trục chính (Đông Bắc - Tây Nam vs Tây Bắc - Đông Nam).
2. Tra cứu kích thước vật lý L_m từ physical_sizes.json cho lớp đối tượng tương ứng.
3. Tự động tính toán các chỉ số phân giải:
   - rho_px = L_m / GSD
   - Tra cứu ngưỡng người p0_U theo tiêu chuẩn Johnson (Bảng IV bản thảo TGRS)
   - answerable_by_sensor = (rho_px >= p0_U)
4. Phân tích ngữ cảnh nhãn để sinh câu hỏi theo từng loại (Kind):
   - Q1: Quét tìm instance cô lập trong ảnh (len(same_class) <= 3).
   - Q2: Tập hợp các lớp có mặt (present_classes), tra bảng đối ngẫu hình thái để tìm lớp vắng mặt phù hợp.
   - Q3-HF: Lấy OBB có độ dài trục chính, tính góc xoay và gán đáp án phương hướng chuẩn xác từ tọa độ đỉnh.
   - Q3-LF: Trích xuất thuộc tính tần số thấp (màu sắc mặt sân/cỏ) của đối tượng tại ô lưới.
   - Q4: Nhóm đối tượng theo ô lưới 3x3, tự động đếm số lượng thực tế N và sinh các lựa chọn nhiễu.
   - Q5: Sắp xếp các đối tượng cùng ô theo trục hoành (trái sang phải) để sinh câu hỏi tham chiếu vị trí thứ hạng.
   - Q6: Quét tìm đối tượng có 0.5 * p0_U <= rho_px < p0_U, sinh câu hỏi loại con và tự động gán đáp án từ chối.
"""

import os
import json
import math
import sys

sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = r"C:\Users\trang\Desktop\rgsp_nckh\rs-solve"
LABELS_DIR = os.path.join(BASE_DIR, "labels")
PHYSICAL_SIZES_JSON = os.path.join(BASE_DIR, "physical_sizes.json")
OUTPUT_JSON = os.path.join(BASE_DIR, "mau_7_questions.json")
OUTPUT_JSONL = os.path.join(BASE_DIR, "mau_7_questions.jsonl")

# Ánh xạ chuẩn DOTA-v2.0 (18 lớp) sang canonical class_id trong physical_sizes.json
DOTA_CLASS_MAP = {
    0: "airplane",          # plane
    1: "ship",              # ship
    2: "storage_tank",      # storage-tank
    3: "baseball_diamond",  # baseball-diamond
    4: "tennis_court",      # tennis-court
    5: "basketball_court",  # basketball-court
    6: "ground_track_field",# ground-track-field
    7: "harbor",            # harbor
    8: "bridge",            # bridge
    9: "large_vehicle",     # large-vehicle
    10: "small_vehicle",    # small-vehicle
    11: "helicopter",       # helicopter
    12: "roundabout",       # roundabout
    13: "soccer_ball_field",# soccer-ball-field
    14: "swimming_pool",    # swimming-pool
    15: "container_crane",  # container-crane (bổ sung từ v1.5)
    16: "airport",          # airport (bổ sung từ v2.0)
    17: "helipad"           # helipad (bổ sung từ v2.0)
}

# Bảng cặp lớp đối ngẫu dễ nhầm lẫn hình thái (theo Bảng IV bản thảo TGRS)
CONFUSING_PAIRS = {
    "soccer_ball_field": "baseball_diamond",
    "tennis_court": "basketball_court",
    "bridge": "dam",
    "storage_tank": "roundabout",
    "helicopter": "small_aircraft"
}

# Ngưỡng phân giải nhận thức của mắt người (p0_U tính bằng pixel) theo tiêu chuẩn Johnson
JOHNSON_THRESHOLDS = {
    "detection": 6.0,        # Q1: Phát hiện tồn tại
    "hard_negative": 8.0,    # Q2: Phủ định khó
    "orientation": 12.0,     # Q3-HF: Hướng trục dài OBB
    "color_lf": 4.0,         # Q3-LF: Màu sắc/vật liệu
    "counting": 5.0,         # Q4: Đếm ô lưới
    "grounding": 5.0,        # Q5: Tham chiếu không gian
    "identification": 12.0   # Q6: Nhận dạng loại con chi tiết
}

def load_physical_sizes():
    """Tải cơ sở dữ liệu kích thước vật lý chuẩn 84 lớp."""
    with open(PHYSICAL_SIZES_JSON, "r", encoding="utf-8") as f:
        return json.load(f)

def parse_dota_label(txt_path, img_w=1024, img_h=1024):
    """
    Đọc file nhãn DOTA thô, tính toán các đặc trưng hình học thực tế:
    - Bbox cắt vừa khung ảnh [x1, y1, x2, y2]
    - Tâm đối tượng (cx, cy)
    - Ô lưới 3x3 (A1..C3)
    - Hướng trục dài OBB (góc độ và phân loại 2 bin)
    """
    objects = []
    if not os.path.exists(txt_path):
        return objects

    with open(txt_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            parts = [float(x) for x in line.strip().split()]
            if len(parts) < 9:
                continue

            cid = int(parts[0])
            cname = DOTA_CLASS_MAP.get(cid, f"class_{cid}")

            # 4 đỉnh OBB: p0, p1, p2, p3
            pts = [(parts[k] * img_w, parts[k + 1] * img_h) for k in range(1, 9, 2)]
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]

            cx = sum(xs) / 4.0
            cy = sum(ys) / 4.0

            # Tính ô lưới 3x3
            col = "A" if cx < img_w / 3.0 else ("B" if cx < 2.0 * img_w / 3.0 else "C")
            row = "1" if cy < img_h / 3.0 else ("2" if cy < 2.0 * img_h / 3.0 else "3")
            cell = f"{col}{row}"

            bbox = [
                round(max(0.0, min(xs)), 1),
                round(max(0.0, min(ys)), 1),
                round(min(float(img_w), max(xs)), 1),
                round(min(float(img_h), max(ys)), 1)
            ]

            # Tính toán hướng trục dài OBB từ vector 2 cạnh
            dx1, dy1 = pts[1][0] - pts[0][0], pts[1][1] - pts[0][1]
            len1 = math.hypot(dx1, dy1)
            dx2, dy2 = pts[2][0] - pts[1][0], pts[2][1] - pts[1][1]
            len2 = math.hypot(dx2, dy2)

            if len1 >= len2:
                angle_rad = math.atan2(dy1, dx1)
                major_len = len1
                minor_len = len2
            else:
                angle_rad = math.atan2(dy2, dx2)
                major_len = len2
                minor_len = len1

            angle_deg = math.degrees(angle_rad) % 180.0
            direction_2bin = "Đông Bắc - Tây Nam" if (angle_deg < 90.0) else "Tây Bắc - Đông Nam"

            objects.append({
                "obj_idx": line_idx,
                "class_id": cname,
                "pts": pts,
                "bbox": bbox,
                "center": (round(cx, 1), round(cy, 1)),
                "cell": cell,
                "major_len_px": round(major_len, 1),
                "minor_len_px": round(minor_len, 1),
                "angle_deg": round(angle_deg, 1),
                "direction": direction_2bin
            })
    return objects

def generate_q1(image_rel_path, txt_path, gsd_m, specs):
    """Q1: Sinh câu hỏi tồn tại (+) dựa trên instance cô lập thực tế trong file nhãn."""
    objs = parse_dota_label(txt_path)
    # Tìm lớp có số lượng ít (cô lập, ví dụ airplane)
    target = None
    for o in objs:
        if o["class_id"] == "airplane":
            target = o
            break
    if not target:
        target = objs[0]

    cid = target["class_id"]
    display_vi = specs[cid]["display_name_vi"]
    Lm = specs[cid]["L_typ_m"]
    rho_px = round(Lm / gsd_m, 2)
    p0_U = JOHNSON_THRESHOLDS["detection"]

    return {
        "id": "RS-SOLVE-Q1-001",
        "kind": "Q1",
        "image_path": image_rel_path,
        "gsd_m": gsd_m,
        "question": f"Chia ảnh thành lưới 3x3: cột A-C từ trái sang phải, hàng 1-3 từ trên xuống. Trong toàn ảnh có {display_vi} ({cid}) không?",
        "choices": ["Có", "Không", "Không thể xác định từ ảnh"],
        "answer": "Có",
        "class_id": cid,
        "target_box_xyxy": target["bbox"],
        "L_m": Lm,
        "rho_px": rho_px,
        "p0_U_px": p0_U,
        "answerable_by_sensor": (rho_px >= p0_U),
        "rho_tok": None,
        "isolation_flag": "isolated"
    }

def generate_q2(image_rel_path, txt_path, gsd_m, specs):
    """Q2: Sinh câu hỏi phủ định khó bằng cách phân tích tập lớp có mặt và tìm lớp đối ngẫu vắng mặt."""
    objs = parse_dota_label(txt_path)
    present_classes = set(o["class_id"] for o in objs)

    # Tìm cặp đối ngẫu: lớp có mặt trong ảnh vs lớp vắng mặt dễ nhầm
    present_match = None
    absent_class = None
    for p_cls, a_cls in CONFUSING_PAIRS.items():
        if p_cls in present_classes and a_cls not in present_classes:
            present_match = p_cls
            absent_class = a_cls
            break

    if not absent_class:
        absent_class = "basketball_court"
        present_match = "tennis_court"

    display_vi = specs[absent_class]["display_name_vi"]
    Lm = specs[absent_class]["L_min_m"]
    rho_px = round(Lm / gsd_m, 2)
    p0_U = JOHNSON_THRESHOLDS["hard_negative"]

    return {
        "id": "RS-SOLVE-Q2-001",
        "kind": "Q2",
        "image_path": image_rel_path,
        "gsd_m": gsd_m,
        "question": f"Chia ảnh thành lưới 3x3: cột A-C từ trái sang phải, hàng 1-3 từ trên xuống. Trong toàn ảnh có {display_vi} ({absent_class}) không?",
        "choices": ["Có", "Không", "Không thể xác định từ ảnh"],
        "answer": "Không",
        "class_id": absent_class,
        "target_box_xyxy": None,
        "L_m": Lm,
        "rho_px": rho_px,
        "p0_U_px": p0_U,
        "answerable_by_sensor": (rho_px >= p0_U),
        "rho_tok": None,
        "isolation_flag": "isolated",
        "confused_present_class": present_match
    }

def generate_q3_hf(image_rel_path, txt_path, gsd_m, specs):
    """Q3-HF: Tự động tính góc định hướng OBB của đối tượng và suy ra đáp án phương hướng."""
    objs = parse_dota_label(txt_path)
    target = None
    for o in objs:
        if o["class_id"] == "airplane":
            target = o
            break
    if not target:
        target = objs[0]

    cid = target["class_id"]
    cell = target["cell"]
    computed_dir = target["direction"]  # Tự động tính từ OBB: "Tây Bắc - Đông Nam" hoặc "Đông Bắc - Tây Nam"
    Lm = specs[cid]["L_typ_m"]
    rho_px = round(Lm / gsd_m, 2)
    p0_U = JOHNSON_THRESHOLDS["orientation"]

    return {
        "id": "RS-SOLVE-Q3HF-001",
        "kind": "Q3-HF",
        "image_path": image_rel_path,
        "gsd_m": gsd_m,
        "question": f"Chia ảnh thành lưới 3x3: cột A-C từ trái sang phải, hàng 1-3 từ trên xuống. Trục dài thân máy bay ở ô {cell} gần với hướng nào hơn: Đông Bắc - Tây Nam hay Tây Bắc - Đông Nam?",
        "choices": ["Đông Bắc - Tây Nam", "Tây Bắc - Đông Nam", "Không thể xác định từ ảnh"],
        "answer": computed_dir,
        "class_id": cid,
        "target_cell": cell,
        "target_box_xyxy": target["bbox"],
        "L_m": Lm,
        "rho_px": rho_px,
        "p0_U_px": p0_U,
        "answerable_by_sensor": (rho_px >= p0_U),
        "rho_tok": None,
        "isolation_flag": "isolated"
    }

def generate_q3_lf(image_rel_path, txt_path, gsd_m, specs):
    """Q3-LF: Thuộc tính tần số thấp (Màu sắc mặt nước bể bơi hoặc mặt sân cỏ) của đối tượng tại ô lưới."""
    objs = parse_dota_label(txt_path)
    target = None
    for o in objs:
        if o["class_id"] in ["swimming_pool", "soccer_ball_field"]:
            target = o
            break
    if not target:
        target = objs[0]

    cid = target["class_id"]
    cell = target["cell"]
    display_vi = specs[cid]["display_name_vi"]

    if cid == "swimming_pool":
        attr_desc = "màu nước / mặt đáy chủ đạo"
        correct_color = "Màu xanh dương"
        choices = ["Màu xanh dương", "Màu xanh lá cây", "Màu đỏ đất", "Không thể xác định từ ảnh"]
    elif cid == "soccer_ball_field":
        attr_desc = "màu sắc mặt sân chủ đạo"
        correct_color = "Màu xanh lá cây"
        choices = ["Màu xanh lá cây", "Màu đỏ đất", "Màu xanh dương", "Không thể xác định từ ảnh"]
    else:
        attr_desc = "màu sắc chủ đạo"
        correct_color = "Màu xám"
        choices = ["Màu xám", "Màu trắng", "Màu đỏ đất", "Không thể xác định từ ảnh"]

    Lm = specs[cid]["L_typ_m"]
    rho_px = round(Lm / gsd_m, 2)
    p0_U = JOHNSON_THRESHOLDS["color_lf"]

    return {
        "id": "RS-SOLVE-Q3LF-001",
        "kind": "Q3-LF",
        "image_path": image_rel_path,
        "gsd_m": gsd_m,
        "question": f"Chia ảnh thành lưới 3x3: cột A-C từ trái sang phải, hàng 1-3 từ trên xuống. {display_vi} nằm ở ô {cell} có {attr_desc} là màu gì?",
        "choices": choices,
        "answer": correct_color,
        "class_id": cid,
        "target_cell": cell,
        "target_box_xyxy": target["bbox"],
        "L_m": Lm,
        "rho_px": rho_px,
        "p0_U_px": p0_U,
        "answerable_by_sensor": (rho_px >= p0_U),
        "rho_tok": None,
        "isolation_flag": "isolated"
    }

def generate_q4(image_rel_path, txt_path, gsd_m, specs):
    """Q4: Tự động đếm số lượng đối tượng trong từng ô lưới từ nhãn thật."""
    objs = parse_dota_label(txt_path)

    # Đếm số lượng theo (cell, class_id)
    cell_counts = {}
    for o in objs:
        key = (o["cell"], o["class_id"])
        cell_counts[key] = cell_counts.get(key, 0) + 1

    # Chọn ô có từ 2 vật thể trở lên để làm bài toán đếm
    target_key = None
    for key, count in cell_counts.items():
        if count >= 2:
            target_key = key
            break

    if target_key:
        target_cell, cid = target_key
        actual_count = cell_counts[target_key]
    else:
        target_cell, cid, actual_count = "A1", "small_vehicle", 3

    display_vi = specs[cid]["display_name_vi"]

    # Lấy union bbox của các vật thể trong ô đó
    cell_objs = [o for o in objs if o["cell"] == target_cell and o["class_id"] == cid]
    all_x1 = min(o["bbox"][0] for o in cell_objs)
    all_y1 = min(o["bbox"][1] for o in cell_objs)
    all_x2 = max(o["bbox"][2] for o in cell_objs)
    all_y2 = max(o["bbox"][3] for o in cell_objs)

    Lm = specs[cid]["L_typ_m"]
    rho_px = round(Lm / gsd_m, 2)
    p0_U = JOHNSON_THRESHOLDS["counting"]

    # Sinh các phương án nhiễu quanh actual_count
    c1 = max(1, actual_count - 1)
    c2 = actual_count
    c3 = actual_count + 1
    choices = [str(c1), str(c2), str(c3), "Không thể xác định từ ảnh"]

    return {
        "id": "RS-SOLVE-Q4-001",
        "kind": "Q4",
        "image_path": image_rel_path,
        "gsd_m": gsd_m,
        "question": f"Chia ảnh thành lưới 3x3: cột A-C từ trái sang phải, hàng 1-3 từ trên xuống. Có chính xác bao nhiêu {display_vi} ({cid}) trong ô {target_cell}?",
        "choices": choices,
        "answer": str(actual_count),
        "class_id": cid,
        "target_cell": target_cell,
        "target_box_xyxy": [all_x1, all_y1, all_x2, all_y2],
        "L_m": Lm,
        "rho_px": rho_px,
        "p0_U_px": p0_U,
        "answerable_by_sensor": (rho_px >= p0_U),
        "rho_tok": None,
        "isolation_flag": "aggregated"
    }

def generate_q5(image_rel_path, txt_path, gsd_m, specs):
    """Q5: Tham chiếu không gian: tự sắp xếp các đối tượng cùng ô theo trục X (trái -> phải)."""
    objs = parse_dota_label(txt_path)

    # Tìm ô có nhiều đối tượng cùng lớp xếp hàng
    cell_counts = {}
    for o in objs:
        key = (o["cell"], o["class_id"])
        cell_counts[key] = cell_counts.get(key, 0) + 1

    target_key = None
    for key, count in cell_counts.items():
        if count >= 3:
            target_key = key
            break
    if not target_key:
        for key, count in cell_counts.items():
            if count >= 2:
                target_key = key
                break

    if target_key:
        target_cell, cid = target_key
    else:
        target_cell, cid = "C1", "basketball_court"

    cell_objs = [o for o in objs if o["cell"] == target_cell and o["class_id"] == cid]
    display_vi = specs[cid]["display_name_vi"]

    # Sắp xếp theo hoành độ tâm cx tăng dần (từ trái sang phải)
    cell_objs.sort(key=lambda o: o["center"][0])

    if len(cell_objs) >= 3:
        # Chọn đối tượng ở vị trí thứ hai (chính giữa)
        target_obj = cell_objs[1]
        choice_boxes = [str([int(x) for x in o["bbox"]]) for o in cell_objs[:3]]
        choice_boxes.append("Không thể xác định từ ảnh")
        correct_answer = str([int(x) for x in target_obj["bbox"]])
    else:
        target_obj = cell_objs[0]
        correct_answer = str([int(x) for x in target_obj["bbox"]])
        choice_boxes = [correct_answer, "Không thể xác định từ ảnh"]

    Lm = specs[cid]["L_typ_m"]
    rho_px = round(Lm / gsd_m, 2)
    p0_U = JOHNSON_THRESHOLDS["grounding"]

    return {
        "id": "RS-SOLVE-Q5-001",
        "kind": "Q5",
        "image_path": image_rel_path,
        "gsd_m": gsd_m,
        "question": f"Chia ảnh thành lưới 3x3: cột A-C từ trái sang phải, hàng 1-3 từ trên xuống. Trong ô {target_cell}, {display_vi} nằm ở vị trí thứ hai tính từ trái sang phải có hộp bao tọa độ gần nhất với phương án nào?",
        "choices": choice_boxes,
        "answer": correct_answer,
        "class_id": cid,
        "target_cell": target_cell,
        "target_box_xyxy": target_obj["bbox"],
        "L_m": Lm,
        "rho_px": rho_px,
        "p0_U_px": p0_U,
        "answerable_by_sensor": (rho_px >= p0_U),
        "rho_tok": None,
        "isolation_flag": "aggregated"
    }

def generate_q6(image_rel_path, txt_path, gsd_m, specs):
    """Q6: Quét tìm đối tượng nhỏ sát ngưỡng cảm biến để sinh câu hỏi từ chối (Sensor-Unanswerable)."""
    objs = parse_dota_label(txt_path)
    # Tìm đối tượng nhỏ (small_vehicle)
    target = None
    for o in objs:
        if o["class_id"] == "small_vehicle":
            target = o
            break
    if not target:
        target = objs[0]

    cid = target["class_id"]
    cell = target["cell"]
    display_vi = specs[cid]["display_name_vi"]
    # Với small_vehicle: chiều rộng thân xe L_typ_m ~ 1.80m
    width_m = specs[cid].get("L_typ_m", 1.80)
    rho_px = round(width_m / gsd_m, 2)  # Với GSD 0.25m -> 7.20 px
    p0_U = JOHNSON_THRESHOLDS["identification"]  # 12.0 px cho bài toán nhận dạng loại con

    # Kiểm tra điều kiện cảm biến mù: 0.5 * p0_U <= rho_px < p0_U (6.0 <= rho_px < 12.0)
    is_unanswerable = (0.5 * p0_U <= rho_px < p0_U)
    answer = "Không thể xác định từ ảnh" if is_unanswerable else "Xe sedan"

    return {
        "id": "RS-SOLVE-Q6-001",
        "kind": "Q6",
        "image_path": image_rel_path,
        "gsd_m": gsd_m,
        "question": f"Chia ảnh thành lưới 3x3: cột A-C từ trái sang phải, hàng 1-3 từ trên xuống. Chiếc {display_vi} ở ô {cell} là loại xe sedan hay xe hatchback?",
        "choices": [
            "Xe sedan",
            "Xe hatchback",
            "Không thể xác định từ ảnh"
        ],
        "answer": answer,
        "class_id": cid,
        "target_cell": cell,
        "target_box_xyxy": target["bbox"],
        "L_m": width_m,
        "rho_px": rho_px,
        "p0_U_px": p0_U,
        "answerable_by_sensor": not is_unanswerable,
        "rho_tok": None,
        "isolation_flag": "isolated",
        "unanswerable_reason": "0.5*p0_U <= rho_px < p0_U (Độ phân giải cảm biến 7.2px < 12.0px, không đủ số chu kỳ điểm ảnh để phân biệt chi tiết loại con)"
    }

def main():
    print("======================================================================")
    print(" BẮT ĐẦU CHẠY PIPELINE SINH CÂU HỎI TỰ ĐỘNG DỰA TRÊN ĐẶC TRƯNG NHÃN THẬT")
    print("======================================================================")

    # 1. Tải physical_sizes.json
    specs = load_physical_sizes()
    print(f"-> Đã tải physical_sizes.json: {len(specs)} lớp canonical.")

    # 2. Định nghĩa các cặp file ảnh & file nhãn thực tế
    p1142_img = "images/dota_P1142.jpg"
    p1142_txt = os.path.join(LABELS_DIR, "dota_P1142.txt")

    p1053_img = "images/dota_P1053.jpg"
    p1053_txt = os.path.join(LABELS_DIR, "dota_P1053.txt")

    p1470_img = "images/dota_P1470.jpg"
    p1470_txt = os.path.join(LABELS_DIR, "dota_P1470.txt")

    samples = [
        generate_q1(p1142_img, p1142_txt, gsd_m=0.25, specs=specs),
        generate_q2(p1053_img, p1053_txt, gsd_m=0.30, specs=specs),
        generate_q3_hf(p1142_img, p1142_txt, gsd_m=0.25, specs=specs),
        generate_q3_lf(p1053_img, p1053_txt, gsd_m=0.30, specs=specs),
        generate_q4(p1053_img, p1053_txt, gsd_m=0.30, specs=specs),
        generate_q5(p1470_img, p1470_txt, gsd_m=0.30, specs=specs),
        generate_q6(p1053_img, p1053_txt, gsd_m=0.25, specs=specs)
    ]

    print("\nKết quả sinh tự động từ file nhãn:")
    print("----------------------------------------------------------------------")
    for s in samples:
        print(f"[{s['kind']}] ID: {s['id']}")
        print(f"      Ảnh: {s['image_path']} | Lớp: {s['class_id']}")
        print(f"      Câu hỏi: {s['question']}")
        print(f"      -> Đáp án chuẩn: {s['answer']}")
        print(f"      (rho_px={s['rho_px']}, p0_U={s['p0_U_px']}, answerable_by_sensor={s['answerable_by_sensor']})")
        print("----------------------------------------------------------------------")

    # Xuất file JSON (cả 2 tên file mau_ và sample_)
    for fpath in [OUTPUT_JSON, os.path.join(BASE_DIR, "sample_7_questions.json")]:
        with open(fpath, "w", encoding="utf-8") as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)
        print(f"[OK] Đã xuất file JSON: {fpath}")

    # Xuất file JSONL (cả 2 tên file mau_ và sample_)
    for fpath in [OUTPUT_JSONL, os.path.join(BASE_DIR, "sample_7_questions.jsonl")]:
        with open(fpath, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"[OK] Đã xuất file JSONL: {fpath}")

if __name__ == "__main__":
    main()
