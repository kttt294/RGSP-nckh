"""
build_samples.py
----------------
Script tự động sinh 7 câu hỏi mẫu (Q1, Q2, Q3-HF, Q3-LF, Q4, Q5, Q6) cho benchmark RS-Resolve.

Nguyên tắc tính toán toán học & lý thuyết:
1. Tra cứu kích thước vật lý L_m trực tiếp từ physical_sizes.json:
   - L_m được lấy theo L_typ_m hoặc L_min_m tương ứng với critical_dimension_type của lớp canonical.
2. Tính toán độ phân giải cảm biến quang học (chu kỳ điểm ảnh hiệu dụng):
   rho_px = L_m / gsd_m  (hoặc min cạnh OBB trên ảnh)
3. Ngưỡng nhận thức con người p0_U_px (Johnson's Criteria theo Bảng IV bản thảo TGRS):
   - Q1 (Phát hiện tồn tại): p0_U = 6.0 px
   - Q2 (Phủ định khó hình thái): p0_U = 8.0 px
   - Q3-HF (Hướng trục OBB 0-180 độ): p0_U = 12.0 px
   - Q3-LF (Màu sắc/vật liệu tần số thấp): p0_U = 4.0 px
   - Q4 (Đếm theo ô lưới): p0_U = 5.0 px
   - Q5 (Tham chiếu không gian): p0_U = 5.0 px
   - Q6 (Phân biệt loại con sát ngưỡng cảm biến): p0_U = 12.0 px
4. Điều kiện cảm biến trả lời được (Sensor Answerability):
   answerable_by_sensor = (rho_px >= p0_U_px)
5. Cơ chế từ chối của Q6:
   Khi 0.5 * p0_U <= rho_px < p0_U:
   answer = "Không thể xác định từ ảnh" (Bắt buộc VLM phải từ chối thay vì ảo giác).
"""

import os
import json
import math
import sys

sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = r"C:\Users\trang\Desktop\rgsp_nckh\rs-solve"
PHYSICAL_SIZES_JSON = os.path.join(BASE_DIR, "physical_sizes.json")
OUTPUT_JSON = os.path.join(BASE_DIR, "sample_7_questions.json")
OUTPUT_JSONL = os.path.join(BASE_DIR, "sample_7_questions.jsonl")

# 1. Bảng ngưỡng nhận thức con người p0_U (Johnson's Criteria) theo từng tác vụ trong Bảng IV bản thảo TGRS
JOHNSON_THRESHOLDS = {
    "detection": 6.0,        # Q1: Detection
    "hard_negative": 8.0,    # Q2: Hard negative morphology differentiation
    "orientation": 12.0,     # Q3-HF: Long-axis orientation
    "color_lf": 4.0,         # Q3-LF: Low-frequency color / material
    "counting": 5.0,         # Q4: Counting in grid cell
    "grounding": 5.0,        # Q5: Spatial visual grounding
    "identification": 12.0   # Q6: Fine-grained subtype differentiation
}

def load_physical_sizes():
    """Tải cơ sở dữ liệu kích thước vật lý chuẩn 84 lớp."""
    with open(PHYSICAL_SIZES_JSON, "r", encoding="utf-8") as f:
        return json.load(f)

def get_grid_cell(cx, cy, img_w=1024, img_h=1024):
    """Xác định ô lưới 3x3 (Cột A-C, Hàng 1-3)."""
    col = "A" if cx < img_w / 3.0 else ("B" if cx < 2.0 * img_w / 3.0 else "C")
    row = "1" if cy < img_h / 3.0 else ("2" if cy < 2.0 * img_h / 3.0 else "3")
    return f"{col}{row}"

def clip_box(box, img_w=1024, img_h=1024):
    """Cắt bounding box nằm vừa vặn trong khung ảnh."""
    return [
        round(max(0.0, float(box[0])), 1),
        round(max(0.0, float(box[1])), 1),
        round(min(float(img_w), float(box[2])), 1),
        round(min(float(img_h), float(box[3])), 1)
    ]

def compute_resolution(L_m, gsd_m):
    """Tính chu kỳ điểm ảnh hiệu dụng rho_px = L_m / gsd_m."""
    return round(float(L_m) / float(gsd_m), 2)

def build_q1(specs):
    """Q1: Tồn tại (+) máy bay cô lập trong P1142."""
    class_id = "airplane"
    gsd_m = 0.25
    L_m = specs[class_id]["L_typ_m"]  # 34.11m (A320/B737 wingspan)
    rho_px = compute_resolution(L_m, gsd_m)  # 136.44 px
    p0_U = JOHNSON_THRESHOLDS["detection"]   # 6.0 px
    answerable = (rho_px >= p0_U)

    return {
        "id": "RS-SOLVE-Q1-001",
        "kind": "Q1",
        "image_path": "sample_data/images/dota_P1142.jpg",
        "gsd_m": gsd_m,
        "question": "Chia ảnh thành lưới 3×3: cột A–C từ trái sang phải, hàng 1–3 từ trên xuống. Trong toàn ảnh có máy bay cánh bằng (airplane) không?",
        "choices": ["Có", "Không", "Không thể xác định từ ảnh"],
        "answer": "Có",
        "class_id": class_id,
        "target_box_xyxy": clip_box([641.0, 273.0, 1024.0, 707.0]),
        "L_m": L_m,
        "rho_px": rho_px,
        "p0_U_px": p0_U,
        "answerable_by_sensor": answerable,
        "rho_tok": None,
        "isolation_flag": "isolated"
    }

def build_q2(specs):
    """Q2: Tồn tại (-) phủ định khó: hỏi về sân bóng chày trong ảnh chỉ có sân bóng đá."""
    class_id = "baseball_diamond"  # Lớp giả định bị hỏi nhưng vắng mặt
    gsd_m = 0.30
    L_m = specs[class_id]["L_min_m"]  # 27.43m (cự ly base 90ft)
    rho_px = compute_resolution(L_m, gsd_m)  # 91.43 px
    p0_U = JOHNSON_THRESHOLDS["hard_negative"]  # 8.0 px
    # Vì rho_px >= p0_U, cảm biến hoàn toàn nhìn thấy rõ nếu có -> Đáp án chắc chắn là "Không"
    answerable = (rho_px >= p0_U)

    return {
        "id": "RS-SOLVE-Q2-001",
        "kind": "Q2",
        "image_path": "sample_data/images/dota_P1053.jpg",
        "gsd_m": gsd_m,
        "question": "Chia ảnh thành lưới 3×3: cột A–C từ trái sang phải, hàng 1–3 từ trên xuống. Trong toàn ảnh có sân bóng chày (baseball diamond) không?",
        "choices": ["Có", "Không", "Không thể xác định từ ảnh"],
        "answer": "Không",
        "class_id": class_id,
        "target_box_xyxy": None,
        "L_m": L_m,
        "rho_px": rho_px,
        "p0_U_px": p0_U,
        "answerable_by_sensor": answerable,
        "rho_tok": None,
        "isolation_flag": "isolated",
        "confused_present_class": "soccer_ball_field"
    }

def build_q3_hf(specs):
    """Q3-HF: Hướng trục dài thân máy bay (Orientation 0-180 độ)."""
    class_id = "airplane"
    gsd_m = 0.25
    L_m = specs[class_id]["L_typ_m"]  # 34.11m
    rho_px = compute_resolution(L_m, gsd_m)  # 136.44 px
    p0_U = JOHNSON_THRESHOLDS["orientation"]  # 12.0 px
    answerable = (rho_px >= p0_U)

    return {
        "id": "RS-SOLVE-Q3HF-001",
        "kind": "Q3-HF",
        "image_path": "sample_data/images/dota_P1142.jpg",
        "gsd_m": gsd_m,
        "question": "Chia ảnh thành lưới 3×3: cột A–C từ trái sang phải, hàng 1–3 từ trên xuống. Trục dài thân máy bay ở ô C2 gần với hướng nào hơn: Đông Bắc – Tây Nam hay Tây Bắc – Đông Nam?",
        "choices": ["Đông Bắc – Tây Nam", "Tây Bắc – Đông Nam", "Không thể xác định từ ảnh"],
        "answer": "Tây Bắc – Đông Nam",
        "class_id": class_id,
        "target_cell": "C2",
        "target_box_xyxy": clip_box([641.0, 273.0, 1024.0, 707.0]),
        "L_m": L_m,
        "rho_px": rho_px,
        "p0_U_px": p0_U,
        "answerable_by_sensor": answerable,
        "rho_tok": None,
        "isolation_flag": "isolated"
    }

def build_q3_lf(specs):
    """Q3-LF: Màu sắc chủ đạo mặt sân bóng đá cỏ tự nhiên."""
    class_id = "soccer_ball_field"
    gsd_m = 0.30
    L_m = specs[class_id]["L_typ_m"]  # 68.00m (FIFA standard pitch width)
    rho_px = compute_resolution(L_m, gsd_m)  # 226.67 px
    p0_U = JOHNSON_THRESHOLDS["color_lf"]    # 4.0 px
    answerable = (rho_px >= p0_U)

    return {
        "id": "RS-SOLVE-Q3LF-001",
        "kind": "Q3-LF",
        "image_path": "sample_data/images/dota_P1053.jpg",
        "gsd_m": gsd_m,
        "question": "Chia ảnh thành lưới 3×3: cột A–C từ trái sang phải, hàng 1–3 từ trên xuống. Sân bóng đá nằm ở ô A2 có màu sắc mặt sân chủ đạo là màu gì?",
        "choices": ["Màu xanh lá cây", "Màu đỏ đất", "Màu xanh dương", "Không thể xác định từ ảnh"],
        "answer": "Màu xanh lá cây",
        "class_id": class_id,
        "target_cell": "A2",
        "target_box_xyxy": clip_box([136.0, 469.0, 170.0, 513.0]),
        "L_m": L_m,
        "rho_px": rho_px,
        "p0_U_px": p0_U,
        "answerable_by_sensor": answerable,
        "rho_tok": None,
        "isolation_flag": "isolated"
    }

def build_q4(specs):
    """Q4: Đếm số lượng xe cơ giới trong ô C2."""
    class_id = "small_vehicle"
    gsd_m = 0.30
    L_m = specs[class_id]["L_typ_m"]  # 1.80m (car width)
    rho_px = compute_resolution(L_m, gsd_m)  # 6.00 px
    p0_U = JOHNSON_THRESHOLDS["counting"]    # 5.0 px
    answerable = (rho_px >= p0_U)

    return {
        "id": "RS-SOLVE-Q4-001",
        "kind": "Q4",
        "image_path": "sample_data/images/dota_P1053.jpg",
        "gsd_m": gsd_m,
        "question": "Chia ảnh thành lưới 3×3: cột A–C từ trái sang phải, hàng 1–3 từ trên xuống. Có chính xác bao nhiêu xe cơ giới (small vehicle) trong ô C2?",
        "choices": ["1", "2", "3", "Không thể xác định từ ảnh"],
        "answer": "3",
        "class_id": class_id,
        "target_cell": "C2",
        "target_box_xyxy": clip_box([641.0, 510.0, 1011.0, 745.0]),
        "L_m": L_m,
        "rho_px": rho_px,
        "p0_U_px": p0_U,
        "answerable_by_sensor": answerable,
        "rho_tok": None,
        "isolation_flag": "aggregated"
    }

def build_q5(specs):
    """Q5: Tham chiếu không gian chiếc xe ở giữa trong 3 xe ô C2."""
    class_id = "small_vehicle"
    gsd_m = 0.30
    L_m = specs[class_id]["L_typ_m"]  # 1.80m
    rho_px = compute_resolution(L_m, gsd_m)  # 6.00 px
    p0_U = JOHNSON_THRESHOLDS["grounding"]   # 5.0 px
    answerable = (rho_px >= p0_U)

    return {
        "id": "RS-SOLVE-Q5-001",
        "kind": "Q5",
        "image_path": "sample_data/images/dota_P1053.jpg",
        "gsd_m": gsd_m,
        "question": "Chia ảnh thành lưới 3×3: cột A–C từ trái sang phải, hàng 1–3 từ trên xuống. Trong ô C2, chiếc xe cơ giới nằm ở vị trí chính giữa (thứ hai tính từ trái sang phải) có hộp bao tọa độ gần nhất với phương án nào?",
        "choices": [
            "[641, 510, 737, 696]",
            "[779, 519, 874, 704]",
            "[913, 559, 1011, 745]",
            "Không thể xác định từ ảnh"
        ],
        "answer": "[779, 519, 874, 704]",
        "class_id": class_id,
        "target_cell": "C2",
        "target_box_xyxy": clip_box([779.0, 519.0, 874.0, 704.0]),
        "L_m": L_m,
        "rho_px": rho_px,
        "p0_U_px": p0_U,
        "answerable_by_sensor": answerable,
        "rho_tok": None,
        "isolation_flag": "aggregated"
    }

def build_q6(specs):
    """Q6: Không thể trả lời do cảm biến (0.5*p0_U <= rho_px < p0_U)."""
    class_id = "small_vehicle"
    gsd_m = 0.30
    L_m = specs[class_id]["L_typ_m"]  # 1.80m
    rho_px = compute_resolution(L_m, gsd_m)  # 6.00 px
    p0_U = JOHNSON_THRESHOLDS["identification"]  # 12.0 px (để phân biệt xe pickup vs van)
    # 0.5 * 12.0 = 6.0 <= rho_px (6.0) < 12.0 -> Nằm đúng khoảng mù cảm biến!
    answerable = False
    answer = "Không thể xác định từ ảnh"

    return {
        "id": "RS-SOLVE-Q6-001",
        "kind": "Q6",
        "image_path": "sample_data/images/dota_P1053.jpg",
        "gsd_m": gsd_m,
        "question": "Chia ảnh thành lưới 3×3: cột A–C từ trái sang phải, hàng 1–3 từ trên xuống. Chiếc xe cơ giới nhỏ ở góc dưới ô A3 là loại xe bán tải (pickup truck) hay xe tải van (van)?",
        "choices": [
            "Xe bán tải (pickup truck)",
            "Xe tải van (van)",
            "Không thể xác định từ ảnh"
        ],
        "answer": answer,
        "class_id": class_id,
        "target_cell": "A3",
        "target_box_xyxy": clip_box([82.0, 654.0, 180.0, 840.0]),
        "L_m": L_m,
        "rho_px": rho_px,
        "p0_U_px": p0_U,
        "answerable_by_sensor": answerable,
        "rho_tok": None,
        "isolation_flag": "isolated",
        "unanswerable_reason": "0.5*p0_U <= rho_px < p0_U (Độ phân giải cảm biến không đủ số chu kỳ điểm ảnh để phân biệt chi tiết loại con)"
    }

def main():
    print("======================================================================")
    print(" BẮT ĐẦU CHẠY SCRIPT BUILD_SAMPLES.PY (TỰ ĐỘNG TÍNH TOÁN THEO CÔNG THỨC)")
    print("======================================================================")

    # 1. Tải physical_sizes.json
    specs = load_physical_sizes()
    print(f"-> Đã tải thành công physical_sizes.json: {len(specs)} lớp canonical.")

    # 2. Xây dựng 7 câu hỏi mẫu
    builders = [build_q1, build_q2, build_q3_hf, build_q3_lf, build_q4, build_q5, build_q6]
    samples = []

    print("\nChi tiết quá trình tính toán toán học cho từng câu hỏi:")
    print("----------------------------------------------------------------------")
    for b in builders:
        item = b(specs)
        samples.append(item)
        qid = item["id"]
        kind = item["kind"]
        cid = item["class_id"]
        Lm = item["L_m"]
        gsd = item["gsd_m"]
        rho = item["rho_px"]
        p0 = item["p0_U_px"]
        ans_sensor = item["answerable_by_sensor"]
        ans = item["answer"]
        print(f"[{kind}] ID: {qid}")
        print(f"      Lớp: {cid} | L_m = {Lm:.2f}m | GSD = {gsd:.2f}m/px")
        print(f"      Công thức: rho_px = L_m / GSD = {Lm:.2f} / {gsd:.2f} = {rho:.2f} px")
        print(f"      Ngưỡng Johnson p0_U = {p0:.1f} px | Cảm biến giải quyết được? {ans_sensor}")
        print(f"      -> Đáp án chuẩn: {ans}")
        print("----------------------------------------------------------------------")

    # 3. Xuất file JSON (dạng mảng đọc trực quan)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] Đã xuất thành công file JSON: {OUTPUT_JSON}")

    # 4. Xuất file JSONL (chuẩn streaming cho mô hình VLM)
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"[OK] Đã xuất thành công file JSONL: {OUTPUT_JSONL}")
    print("======================================================================\n")

if __name__ == "__main__":
    main()
