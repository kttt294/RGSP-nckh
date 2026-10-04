"""Sinh tự động 400-500 câu hỏi Q3-LF (Màu sắc chủ đạo) cho RS-Solve Benchmark.

Thuật toán tự động 100% (Option 2):
1. Quét các scene trong 5 dataset (DOTA, DIOR, VisDrone, xView, iSAID).
2. Lọc các đối tượng thỏa mãn tiêu chuẩn Johnson & kiểm soát chất lượng RS-Solve:
   - require_unique_target: nhìn thấy rõ, cô lập (cách xa láng giềng >= 4 token), duy nhất trong ô 3x3.
   - Loại bỏ các lớp không đồng nhất màu (harbor, airport, bridge, dam...).
3. Tự động trích xuất màu sắc bằng Computer Vision (HSV + Core-crop):
   - Crop vùng lõi 25% trung tâm của bbox (margin 25% mỗi phía) để loại trừ nền đường/bóng đổ.
   - Lọc bỏ điểm ảnh bóng tối sâu (V < 35) và phản xạ lóa sáng mặt trời (S < 20, V > 248).
   - Phân loại từng pixel vào 6 màu chuẩn: Màu đỏ, Màu xanh dương, Màu xanh lá cây, Màu vàng, Màu trắng, Màu đen / xám.
   - Tính tỷ lệ áp đảo (Dominant Ratio). Chỉ giữ các mẫu có độ tin cậy >= 65% và >= 25 pixel hợp lệ.
4. Cân bằng phân phối màu sắc:
   - Mục tiêu: ~65-75 câu cho mỗi màu trong 6 màu chuẩn.
   - Thêm ~45-50 câu Không thể xác định từ ảnh (rho_px < 4.0 px) theo tiêu chuẩn Johnson.
5. Chuyển ngữ song ngữ Việt - Anh đồng bộ.
6. Hợp nhất trực tiếp vào các file benchmark chính thức.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import numpy as np
from PIL import Image

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

Image.MAX_IMAGE_PIXELS = None

# Thiết lập đường dẫn import
CURRENT_DIR = Path(__file__).resolve().parent
REPO_DIR = CURRENT_DIR.parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from core_generators import (
    Scene, InvalidInput, SkipSample, make_sample, rng_for,
    require_unique_target, in_cell, native_footprint, threshold,
    COLOR_CHOICES, ABSTAIN, GRID_PREFIX
)
from parsers import (
    parse_dota_label, parse_dota_header,
    parse_isaid_label, parse_dior_label,
    parse_xview_label, parse_visdrone_label,
    DOTA_CLASS_MAP, ISAID_NAME_MAP, DIOR_NAME_MAP,
    XVIEW_ID_MAP, VISDRONE_NAME_MAP
)
from build_dataset import (
    load_specs, load_config, load_dataset_scene, discover_scenes
)
from export_english_benchmark import translate_item

# Các lớp không có màu sơn đồng nhất (diện tích cảnh rộng hoặc địa vật hỗn hợp)
EXCLUDED_CLASSES = {
    "harbor", "airport", "bridge", "dam", "ground_track_field",
    "ground-track-field", "overpass", "stadium"
}


def classify_pixel_hsv(h: float, s: float, v: float) -> str | None:
    """Phân loại một pixel theo hệ màu HSV sang 6 màu chuẩn của RS-Solve.
    
    h: [0, 180] (thang đo OpenCV: 0 - 360 độ chia đôi)
    s: [0, 255]
    v: [0, 255]
    """
    # 1. Lọc nhiễu: bóng đổ sâu và điểm chói lóa
    if v < 35:
        return None
    if s < 20 and v > 248:
        return None

    # 2. Vô sắc (Saturation thấp)
    if s < 45:
        if v >= 165:
            return "Màu trắng"
        else:
            return "Màu đen / xám"

    # 3. Giá trị độ sáng quá thấp chuyển về đen / xám
    if v < 55:
        return "Màu đen / xám"

    # 4. Có sắc màu (Chromatic): Kiểm tra cung góc màu Hue
    # Đỏ: 0-10 hoặc 165-180
    if (0 <= h <= 10) or (165 <= h <= 180):
        return "Màu đỏ"
    # Vàng / Cam vàng: 11-35
    elif 11 <= h <= 35:
        return "Màu vàng"
    # Xanh lá cây: 36-85
    elif 36 <= h <= 85:
        return "Màu xanh lá cây"
    # Xanh dương / Cyan: 86-135
    elif 86 <= h <= 135:
        return "Màu xanh dương"

    return None


def extract_dominant_color(img_rgb: Image.Image, bbox: list[float], min_pixels: int = 25, min_confidence: float = 0.65):
    """Trích xuất màu chủ đạo từ vùng bbox với kỹ thuật core-crop và biểu quyết HSV."""
    x1, y1, x2, y2 = [int(round(c)) for c in bbox]
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)

    # Core crop: lấy vùng 25% trung tâm để triệt tiêu viền bóng râm & mặt đường
    mx = int(0.25 * w)
    my = int(0.25 * h)
    cx1 = max(0, x1 + mx)
    cy1 = max(0, y1 + my)
    cx2 = min(img_rgb.width, x2 - mx)
    cy2 = min(img_rgb.height, y2 - my)

    if cx2 <= cx1 or cy2 <= cy1:
        cx1, cy1, cx2, cy2 = max(0, x1), max(0, y1), min(img_rgb.width, x2), min(img_rgb.height, y2)

    crop = img_rgb.crop((cx1, cy1, cx2, cy2))
    arr = np.array(crop, dtype=np.float32) / 255.0
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    v = maxc * 255.0
    deltac = maxc - minc

    s = np.zeros_like(maxc)
    mask_max = maxc > 1e-5
    s[mask_max] = (deltac[mask_max] / maxc[mask_max]) * 255.0

    h_arr = np.zeros_like(maxc)
    mask_delta = deltac > 1e-5

    mask_r = mask_delta & (maxc == r)
    h_arr[mask_r] = (60.0 * ((g[mask_r] - b[mask_r]) / deltac[mask_r])) % 360.0

    mask_g = mask_delta & (maxc == g)
    h_arr[mask_g] = (60.0 * ((b[mask_g] - r[mask_g]) / deltac[mask_g]) + 120.0) % 360.0

    mask_b = mask_delta & (maxc == b)
    h_arr[mask_b] = (60.0 * ((r[mask_b] - g[mask_b]) / deltac[mask_b]) + 240.0) % 360.0

    h_arr = h_arr / 2.0  # Chuyển về thang 0-180 của OpenCV

    h_flat = h_arr.flatten()
    s_flat = s.flatten()
    v_flat = v.flatten()

    votes = {c: 0 for c in COLOR_CHOICES}
    valid_count = 0
    for hi, si, vi in zip(h_flat, s_flat, v_flat):
        color = classify_pixel_hsv(hi, si, vi)
        if color:
            votes[color] += 1
            valid_count += 1

    if valid_count < min_pixels:
        return None, 0.0, votes

    best_color = max(votes, key=votes.get)
    best_ratio = votes[best_color] / valid_count

    if best_ratio < min_confidence:
        return None, best_ratio, votes

    return best_color, best_ratio, votes


def build_q3_lf_sample(scene: Scene, obj: dict, specs: dict, config: dict, truth_color: str, answerable: bool):
    """Tạo mẫu câu hỏi chuẩn Q3-LF."""
    cid = obj["class_id"]
    spec = specs[cid]
    display_vi = spec.get("display_name_vi", cid)
    cell = obj["cell"]
    rho = native_footprint(scene, obj)
    p0 = threshold(config, "Q3-LF")

    rng = rng_for(config, scene, "Q3-LF", obj["obj_idx"])
    options = [truth_color] + rng.sample([c for c in COLOR_CHOICES if c != truth_color], 5)
    rng.shuffle(options)

    question_text = f"{display_vi} ở ô {cell} có màu chủ đạo nào?"
    q = make_sample(
        scene, "Q3-LF", cid, obj["bbox"], rho, p0,
        question_text, options, truth_color, "isolated", cell=cell
    )

    if not answerable:
        q["answerable_by_sensor"] = False
        q["answer"] = ABSTAIN
        q["unanswerable_reason"] = (
            f"Câu gốc Q3-LF có footprint cảm biến rho_px={rho:.2f}px < p0_U_px={p0:.2f}px, "
            f"dưới ngưỡng từ calibration đã khai báo."
        )

    return q


def generate_q3_lf_questions(
    scenes_info: list[tuple[str, Path, Path]],
    specs: dict,
    config: dict,
    target_per_color: int = 70,
    target_unanswerable: int = 50,
    min_confidence: float = 0.65,
    min_pixels: int = 25,
    max_scenes: int | None = None,
    seed: int = 42
):
    """Duyệt các scene và sinh tập câu hỏi Q3-LF cân bằng màu sắc và khả năng giải."""
    print("=" * 80)
    print("QUY TRÌNH TỰ ĐỘNG SINH CÂU HỎI Q3-LF (DOMINANT COLOR BENCHMARK)")
    print("=" * 80)
    print(f"Mục tiêu câu hỏi cho mỗi màu ({len(COLOR_CHOICES)} màu): {target_per_color} câu/màu")
    print(f"Mục tiêu câu không thể xác định (Johnson unanswerable): {target_unanswerable} câu")
    total_target = len(COLOR_CHOICES) * target_per_color + target_unanswerable
    print(f"Tổng số lượng mục tiêu: ~{total_target} câu hỏi")
    print(f"Ngưỡng tin cậy màu sắc (Dominant Ratio): >= {min_confidence:.0%}")
    print(f"Số pixel hợp lệ tối thiểu: >= {min_pixels} px")
    print("=" * 80)

    # Khởi tạo hạn ngạch
    quotas = {c: 0 for c in COLOR_CHOICES}
    unanswerable_count = 0
    generated_items = []
    dataset_counts = Counter()

    # Xáo trộn danh sách scene để lấy đều các dataset
    rng = random.Random(seed)
    shuffled_scenes = list(scenes_info)
    rng.shuffle(shuffled_scenes)

    if max_scenes:
        shuffled_scenes = shuffled_scenes[:max_scenes]

    start_time = time.time()
    scenes_checked = 0

    p0 = threshold(config, "Q3-LF")

    for ds_name, img_path, lbl_path in shuffled_scenes:
        scenes_checked += 1
        meta = {}
        try:
            scene = load_dataset_scene(ds_name, img_path, lbl_path, meta, config)
        except Exception:
            continue

        # Kiểm tra nhanh: nếu không có object nào trong scene thì bỏ qua
        if not scene.objects:
            continue

        # Tìm các ứng viên có tiềm năng
        candidates = []
        for obj in scene.objects:
            cid = obj.get("class_id")
            if not cid or cid not in specs or cid in EXCLUDED_CLASSES:
                continue
            try:
                require_unique_target(scene, obj)
                candidates.append(obj)
            except SkipSample:
                continue

        if not candidates:
            continue

        # Đã tìm thấy ứng viên hợp lệ -> nạp ảnh
        try:
            with Image.open(img_path) as pil_img:
                img_rgb = pil_img.convert("RGB")
                img_w, img_h = img_rgb.size
        except Exception:
            continue

        for obj in candidates:
            # Kiểm tra xem toàn bộ hạn ngạch đã đầy chưa
            all_colors_full = all(quotas[c] >= target_per_color for c in COLOR_CHOICES)
            unans_full = (unanswerable_count >= target_unanswerable)
            if all_colors_full and unans_full:
                break

            rho = native_footprint(scene, obj)

            # Trường hợp 1: Unanswerable (rho < p0)
            if rho < p0:
                if unanswerable_count < target_unanswerable:
                    dummy_color = rng.choice(COLOR_CHOICES)
                    q = build_q3_lf_sample(scene, obj, specs, config, dummy_color, answerable=False)
                    generated_items.append((ds_name, q))
                    unanswerable_count += 1
                    dataset_counts[ds_name] += 1
                continue

            # Trường hợp 2: Answerable (rho >= p0)
            # Yêu cầu kích thước tối thiểu để màu sắc có độ tin cậy quang học cao
            if rho < 14.0 or obj["minor_len_px"] < 14.0:
                continue

            # Trích xuất màu chủ đạo
            color, conf, votes = extract_dominant_color(
                img_rgb, obj["bbox"], min_pixels=min_pixels, min_confidence=min_confidence
            )

            if color is None:
                continue

            # Kiểm tra hạn ngạch của màu này
            if quotas[color] < target_per_color:
                q = build_q3_lf_sample(scene, obj, specs, config, color, answerable=True)
                generated_items.append((ds_name, q))
                quotas[color] += 1
                dataset_counts[ds_name] += 1

        if all(quotas[c] >= target_per_color for c in COLOR_CHOICES) and (unanswerable_count >= target_unanswerable):
            print(f"[!] Đã đạt đủ hạn ngạch toàn bộ sau khi quét {scenes_checked:,} scenes!")
            break

        if scenes_checked % 500 == 0:
            current_total = sum(quotas.values()) + unanswerable_count
            print(f"  -> Đã quét {scenes_checked:,} scenes | Đã tạo: {current_total:,}/{total_target} câu | " +
                  " ".join(f"{c.split()[-1]}:{quotas[c]}" for c in COLOR_CHOICES) +
                  f" Unans:{unanswerable_count}")

    elapsed = time.time() - start_time
    total_created = len(generated_items)
    print("\n" + "=" * 80)
    print(f"KẾT QUẢ SINH DỮ LIỆU Q3-LF TRONG {elapsed:.1f}s:")
    print(f"Tổng số câu hỏi sinh được: {total_created:,} câu")
    print(f"Số scene đã quét: {scenes_checked:,}")
    print("\nPhân bổ theo màu sắc:")
    for c in COLOR_CHOICES:
        print(f"  - {c:16s}: {quotas[c]:3d} câu ({quotas[c]/max(1, total_created):.1%})")
    print(f"  - Không thể xác định: {unanswerable_count:3d} câu ({unanswerable_count/max(1, total_created):.1%})")

    print("\nPhân bổ theo nguồn dataset:")
    for ds, count in dataset_counts.items():
        print(f"  - {ds.upper():10s}: {count:3d} câu ({count/max(1, total_created):.1%})")
    print("=" * 80)

    # Đánh số ID chuẩn hóa
    final_vi = []
    for idx, (ds_name, item) in enumerate(generated_items, start=1):
        item["id"] = f"RS-SOLVE-Q3LF-{idx:05d}"
        final_vi.append(item)

    return final_vi, dataset_counts


def merge_q3_lf_to_benchmark(
    q3_lf_vi_items: list[dict],
    benchmark_dir: Path,
    specs: dict,
    dataset_counts: dict
):
    """Hợp nhất câu hỏi Q3-LF vào tập benchmark hiện có và cập nhật báo cáo."""
    print("\n[+] Đang tiến hành hợp nhất Q3-LF vào các tệp Benchmark chính thức...")

    jsonl_vi_path = benchmark_dir / "rs_solve_full_benchmark.jsonl"
    json_vi_path = benchmark_dir / "rs_solve_full_benchmark.json"
    jsonl_en_path = benchmark_dir / "rs_solve_full_benchmark_en.jsonl"
    json_en_path = benchmark_dir / "rs_solve_full_benchmark_en.json"
    report_path = benchmark_dir / "rs_solve_full_benchmark_build_report.json"

    fallback_vi = REPO_DIR / "kaggle_benchmark_output" / "rs_solve_full_benchmark.jsonl"
    fallback_en = REPO_DIR / "kaggle_benchmark_output" / "rs_solve_full_benchmark_en.jsonl"
    fallback_rep = REPO_DIR / "kaggle_benchmark_output" / "rs_solve_full_benchmark_build_report.json"

    if jsonl_vi_path.exists():
        existing_vi = [json.loads(line) for line in jsonl_vi_path.open(encoding="utf-8")]
    elif fallback_vi.exists():
        print(f"[*] Tìm thấy tệp chuẩn gốc tại {fallback_vi}. Đang nạp làm nền tảng hợp nhất...")
        existing_vi = [json.loads(line) for line in fallback_vi.open(encoding="utf-8")]
    else:
        print(f"[-] Không tìm thấy tệp benchmark nền. Tạo mới từ các câu hỏi Q3-LF vừa sinh.")
        existing_vi = []

    # Kiểm tra xem Q3-LF đã có trước đó chưa để tránh trùng lặp
    existing_without_q3lf = [item for item in existing_vi if item.get("kind") != "Q3-LF"]
    merged_vi = existing_without_q3lf + q3_lf_vi_items

    # 1. Ghi tệp Tiếng Việt (JSONL & JSON)
    print(f"  -> Ghi {jsonl_vi_path.name} ({len(merged_vi):,} câu)...")
    with open(jsonl_vi_path, "w", encoding="utf-8") as f:
        for item in merged_vi:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"  -> Ghi {json_vi_path.name}...")
    with open(json_vi_path, "w", encoding="utf-8") as f:
        json.dump(merged_vi, f, ensure_ascii=False, indent=2)

    # 2. Chuyển ngữ sang Tiếng Anh
    print(f"  -> Đang chuyển ngữ {len(q3_lf_vi_items):,} câu Q3-LF sang Tiếng Anh...")
    q3_lf_en_items = [translate_item(item, specs) for item in q3_lf_vi_items]

    if jsonl_en_path.exists():
        existing_en = [json.loads(line) for line in jsonl_en_path.open(encoding="utf-8")]
    elif fallback_en.exists():
        existing_en = [json.loads(line) for line in fallback_en.open(encoding="utf-8")]
    else:
        existing_en = []

    existing_en_without_q3lf = [item for item in existing_en if item.get("kind") != "Q3-LF"]
    merged_en = existing_en_without_q3lf + q3_lf_en_items

    print(f"  -> Ghi {jsonl_en_path.name} ({len(merged_en):,} câu)...")
    with open(jsonl_en_path, "w", encoding="utf-8") as f:
        for item in merged_en:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"  -> Ghi {json_en_path.name}...")
    with open(json_en_path, "w", encoding="utf-8") as f:
        json.dump(merged_en, f, ensure_ascii=False, indent=2)

    # 3. Cập nhật Báo cáo Build Report
    target_rep_path = report_path if report_path.exists() else fallback_rep
    if target_rep_path.exists():
        print(f"  -> Cập nhật báo cáo kỹ thuật: {report_path.name}...")
        report = json.loads(target_rep_path.read_text(encoding="utf-8"))
        report["counts_by_kind"]["Q3-LF"] = len(q3_lf_vi_items)
        if "selected_by_dataset_and_kind" in report:
            report["selected_by_dataset_and_kind"]["Q3-LF"] = dict(dataset_counts)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n[+] HOÀN THÀNH TẤT CẢ! TẬP BENCHMARK RS-SOLVE ĐÃ ĐẦY ĐỦ 7 LOẠI CÂU HỎI (Q1 - Q6).")


def main():
    parser = argparse.ArgumentParser(description="Tự động sinh câu hỏi Q3-LF (màu sắc) cho Benchmark RS-Solve.")
    parser.add_argument("--root-dir", default=str(REPO_DIR), help="Thư mục chứa dữ liệu ảnh hoặc sample_data")
    parser.add_argument("--benchmark-dir", default=str(REPO_DIR / "kaggle_benchmark_output"), help="Thư mục chứa benchmark")
    parser.add_argument("--target-per-color", type=int, default=70, help="Số lượng câu hỏi cho mỗi màu (mặc định: 70)")
    parser.add_argument("--target-unanswerable", type=int, default=50, help="Số lượng câu unanswerable rho < 4px (mặc định: 50)")
    parser.add_argument("--min-confidence", type=float, default=0.65, help="Độ tin cậy màu sắc tối thiểu (mặc định: 0.65)")
    parser.add_argument("--min-pixels", type=int, default=25, help="Số pixel hợp lệ tối thiểu (mặc định: 25)")
    parser.add_argument("--max-scenes", type=int, default=None, help="Số scene tối đa cần quét")
    parser.add_argument("--seed", type=int, default=42, help="Seed ngẫu nhiên")
    parser.add_argument("--no-merge", action="store_true", help="Chỉ sinh dữ liệu mà không ghi đè vào file benchmark")
    args = parser.parse_args()

    root_dir = Path(args.root_dir)
    bench_dir = Path(args.benchmark_dir)

    config = load_config()
    specs = load_specs()

    scenes = discover_scenes(root_dir)
    if not scenes:
        print(f"[-] Không tìm thấy scene nào trong {root_dir}")
        sys.exit(1)

    # Loại trừ các file trùng lặp do case-insensitive trên Windows
    unique_scenes = []
    seen_paths = set()
    for ds, img_p, lbl_p in scenes:
        p_res = Path(img_p).resolve()
        if p_res not in seen_paths:
            seen_paths.add(p_res)
            unique_scenes.append((ds, img_p, lbl_p))

    print(f"[+] Tìm thấy {len(unique_scenes):,} scene độc nhất sau khi khử trùng lặp.")

    q3_lf_vi, ds_counts = generate_q3_lf_questions(
        unique_scenes, specs, config,
        target_per_color=args.target_per_color,
        target_unanswerable=args.target_unanswerable,
        min_confidence=args.min_confidence,
        min_pixels=args.min_pixels,
        max_scenes=args.max_scenes,
        seed=args.seed
    )

    if not args.no_merge and q3_lf_vi:
        merge_q3_lf_to_benchmark(q3_lf_vi, bench_dir, specs, ds_counts)


if __name__ == "__main__":
    main()
