import sys
sys.stdout.reconfigure(encoding="utf-8")
import argparse
import json
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

GRID_PREFIX_EN = "Divide the image into a 3x3 grid: columns A-C from left to right, rows 1-3 from top to bottom. "

CHOICE_MAP = {
    "Không thể xác định từ ảnh": "Cannot be determined from the image",
    "Có": "Yes",
    "Không": "No",
    "Từ trên trái xuống dưới phải": "From top-left to bottom-right",
    "Từ trên phải xuống dưới trái": "From top-right to bottom-left",
    "Màu đỏ": "Red",
    "Màu xanh dương": "Blue",
    "Màu xanh lá cây": "Green",
    "Màu vàng": "Yellow",
    "Màu trắng": "White",
    "Màu đen / xám": "Black / Gray",
}


def load_specs(specs_path):
    with open(specs_path, encoding="utf-8") as f:
        return json.load(f)


def translate_item(item, specs):
    new_item = dict(item)
    cid = item.get("class_id", "")
    spec = specs.get(cid, {})
    display_en = spec.get("display_name_en", cid)
    cell = item.get("target_cell", "")
    kind = item.get("kind", "")
    q_vi = item.get("question", "")

    if kind in ("Q1", "Q2"):
        new_item["question"] = f"Is there any {display_en} ({cid}) in the entire image?"
    elif kind == "Q3-HF":
        new_item["question"] = GRID_PREFIX_EN + f"Which diagonal of the image is the major axis of the {display_en} in cell {cell} closest to?"
    elif kind == "Q3-LF":
        new_item["question"] = GRID_PREFIX_EN + f"What is the dominant color of the {display_en} in cell {cell}?"
    elif kind == "Q4":
        new_item["question"] = GRID_PREFIX_EN + f"Exactly how many {display_en} are there in cell {cell}? Only count objects with at least 50% of their area inside the cell; if split evenly, follow the cell containing the center."
    elif kind == "Q5":
        new_item["question"] = GRID_PREFIX_EN + f"In cell {cell}, which bounding box most tightly encloses the second {display_en} from left to right by center? [x1,y1,x2,y2] in image pixel coordinates. Only consider objects with at least 50% area inside the cell; split evenly by center."
    elif kind == "Q6":
        if "Có chính xác bao nhiêu" in q_vi:
            new_item["question"] = GRID_PREFIX_EN + f"Exactly how many {display_en} are there in cell {cell}? Only count objects with at least 50% of their area inside the cell; if split evenly, follow the cell containing the center."
        elif "thứ hai theo tâm" in q_vi:
            new_item["question"] = GRID_PREFIX_EN + f"In cell {cell}, which bounding box most tightly encloses the second {display_en} from left to right by center? [x1,y1,x2,y2] in image pixel coordinates. Only consider objects with at least 50% area inside the cell; split evenly by center."
        elif "Trục dài của" in q_vi:
            new_item["question"] = GRID_PREFIX_EN + f"Which diagonal of the image is the major axis of the {display_en} in cell {cell} closest to?"
        else:
            new_item["question"] = q_vi

    new_item["choices"] = [CHOICE_MAP.get(c, c) for c in item.get("choices", [])]
    new_item["answer"] = CHOICE_MAP.get(item.get("answer"), item.get("answer"))

    if "unanswerable_reason" in item and item["unanswerable_reason"]:
        r = item["unanswerable_reason"]
        r = r.replace("Câu gốc", "Base question")
        r = r.replace("pixel tham chiếu", "reference pixels")
        r = r.replace("ngưỡng từ calibration đã khai báo.", "threshold from declared sensor calibration.")
        new_item["unanswerable_reason"] = r

    return new_item


def process_file(in_path, out_jsonl_path, out_json_path, specs, is_benchmark=False):
    print(f"\n[+] Đang xử lý: {in_path.name} -> {out_jsonl_path.name}")
    start_time = time.time()
    count = 0
    all_items = [] if out_json_path else None

    with open(in_path, encoding="utf-8") as fin, open(out_jsonl_path, "w", encoding="utf-8") as fout:
        for line in fin:
            item = json.loads(line)
            en_item = translate_item(item, specs)
            fout.write(json.dumps(en_item, ensure_ascii=False) + "\n")
            if all_items is not None:
                all_items.append(en_item)
            count += 1
            if count % 50000 == 0:
                print(f"    Đã chuyển ngữ {count:,} câu...")

    if out_json_path and all_items is not None:
        print(f"    Đang xuất định dạng JSON: {out_json_path.name}...")
        with open(out_json_path, "w", encoding="utf-8") as f_json:
            json.dump(all_items, f_json, ensure_ascii=False, indent=2)

    elapsed = time.time() - start_time
    print(f"    => Hoàn thành {count:,} câu trong {elapsed:.2f}s!")


def main():
    parser = argparse.ArgumentParser(description="Chuyển ngữ Benchmark RS-Solve sang Tiếng Anh (Bilingual Benchmark)")
    parser.add_argument("--dir", default=str(BASE_DIR / "kaggle_benchmark_output"), help="Thư mục chứa benchmark")
    parser.add_argument("--specs", default=str(BASE_DIR / "rs-solve" / "physical_sizes.json"), help="Đường dẫn physical_sizes.json")
    args = parser.parse_args()

    work_dir = Path(args.dir)
    specs = load_specs(args.specs)

    print("=" * 80)
    print("QUY TRÌNH CHUYỂN NGỮ TẬP BENCHMARK RS-SOLVE SANG TIẾNG ANH (BILINGUAL BENCHMARK)")
    print("=" * 80)

    # 1. Xử lý tập Benchmark chuẩn hóa (30.332 câu) -> sinh cả JSONL và JSON
    bench_in = work_dir / "rs_solve_full_benchmark.jsonl"
    bench_out_jsonl = work_dir / "rs_solve_full_benchmark_en.jsonl"
    bench_out_json = work_dir / "rs_solve_full_benchmark_en.json"

    if bench_in.exists():
        process_file(bench_in, bench_out_jsonl, bench_out_json, specs, is_benchmark=True)
    else:
        print(f"Cảnh báo: Không tìm thấy {bench_in}")

    # 2. Xử lý tập Full Pool (259.251 câu) -> sinh JSONL (streaming line-by-line)
    pool_in = work_dir / "rs_solve_full_benchmark_full_pool.jsonl"
    pool_out_jsonl = work_dir / "rs_solve_full_benchmark_full_pool_en.jsonl"

    if pool_in.exists():
        process_file(pool_in, pool_out_jsonl, None, specs, is_benchmark=False)
    else:
        print(f"Cảnh báo: Không tìm thấy {pool_in}")

    print("\n" + "=" * 80)
    print("HOÀN TẤT XUẤT BẢN TIẾNG ANH CHO CẢ TẬP BENCHMARK CHỌN LỌC VÀ TẬP FULL POOL!")
    print("=" * 80)


if __name__ == "__main__":
    main()
