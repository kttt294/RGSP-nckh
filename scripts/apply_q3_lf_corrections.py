"""Script tự động áp dụng các nhãn Q3-LF đã được người dùng chỉnh sửa từ inspect.html.

Cách sử dụng:
    python scripts/apply_q3_lf_corrections.py --input path/to/q3_lf_audit_result_YYYY-MM-DD.json
"""

from __future__ import annotations
import argparse
import glob
import json
import os
from pathlib import Path
import shutil
import sys

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

CHOICE_MAP_EN = {
    "Màu đỏ": "Red",
    "Màu xanh dương": "Blue",
    "Màu xanh lá cây": "Green",
    "Màu vàng": "Yellow",
    "Màu trắng": "White",
    "Màu đen / xám": "Black / Gray",
    "Không thể xác định từ ảnh": "Cannot be determined from the image",
}


def find_latest_audit_file(search_dir: Path) -> Path | None:
    patterns = [
        str(search_dir / "q3_lf_audit_result_*.json"),
        str(search_dir / "data" / "q3_lf_audit_result_*.json"),
        str(Path.home() / "Downloads" / "q3_lf_audit_result_*.json"),
    ]
    files = []
    for pat in patterns:
        files.extend(glob.glob(pat))
    if not files:
        return None
    files.sort(key=os.path.getmtime, reverse=True)
    return Path(files[0])


def main():
    parser = argparse.ArgumentParser(description="Áp dụng nhãn sửa Q3-LF vào bộ dữ liệu RS-Solve.")
    parser.add_argument("--input", "-i", type=str, default=None,
                        help="Đường dẫn tới file JSON kết quả audit xuất từ inspect.html")
    args = parser.parse_args()

    repo_dir = Path(__file__).resolve().parent.parent
    data_dir = repo_dir / "data"

    if args.input:
        input_path = Path(args.input)
    else:
        input_path = find_latest_audit_file(repo_dir)

    if not input_path or not input_path.exists():
        print("❌ Lỗi: Không tìm thấy file JSON kết quả audit!")
        print("Vui lòng xuất file từ inspect.html hoặc chỉ định: --input <đường dẫn file .json>")
        sys.exit(1)

    print(f"📖 Đang đọc kết quả kiểm tra từ: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        audit_raw = json.load(f)

    items = audit_raw.get("corrections", audit_raw) if isinstance(audit_raw, dict) else audit_raw
    corrections_map = {}
    verified_correct_count = 0

    for it in items:
        qid = it["id"]
        status = it.get("audit_status", it.get("user_verified_status", ""))
        orig = it.get("original_color", it.get("ground_truth_color", ""))
        corr = it.get("corrected_color", orig)
        is_changed = it.get("is_changed", corr != orig)

        if is_changed and corr and corr != orig:
            corrections_map[qid] = {
                "orig": orig,
                "corr": corr,
                "notes": it.get("notes", "")
            }
        elif status == "verified" or status == "correct":
            verified_correct_count += 1

    print(f"🔍 Thống kê kiểm tra:")
    print(f"   - Tổng mẫu trong file audit: {len(items)}")
    print(f"   - Mẫu được xác nhận đúng: {verified_correct_count}")
    print(f"   - Mẫu được sửa đổi màu: {len(corrections_map)}")

    if not corrections_map:
        print("ℹ️ Không có mẫu nào bị sửa đổi màu. Dữ liệu benchmark giữ nguyên!")
        sys.exit(0)

    print(f"\n📝 Danh sách {len(corrections_map)} câu được sửa nhãn:")
    for qid, cinfo in list(corrections_map.items())[:10]:
        print(f"   [{qid}] {cinfo['orig']} ➜ {cinfo['corr']}")
    if len(corrections_map) > 10:
        print(f"   ... và {len(corrections_map) - 10} câu khác.")

    # 1. Update rs_solve_full_benchmark.jsonl
    full_jsonl = data_dir / "rs_solve_full_benchmark.jsonl"
    if full_jsonl.exists():
        backup_path = full_jsonl.with_suffix(".jsonl.bak")
        shutil.copy2(full_jsonl, backup_path)
        print(f"\n💾 Đã sao lưu bản gốc vào: {backup_path.name}")

        updated_records = []
        applied_count = 0
        with open(full_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                qid = rec.get("id")
                if qid in corrections_map:
                    new_c = corrections_map[qid]["corr"]
                    old_c = rec.get("answer")
                    rec["answer"] = new_c
                    if "choices" in rec and new_c not in rec["choices"]:
                        # Replace old answer with new answer in choices
                        choices = [new_c if c == old_c else c for c in rec["choices"]]
                        rec["choices"] = choices
                    applied_count += 1
                updated_records.append(rec)

        with open(full_jsonl, "w", encoding="utf-8") as f:
            for r in updated_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"✅ Đã cập nhật {applied_count} câu vào {full_jsonl.name}")

        # Update full json
        full_json = data_dir / "rs_solve_full_benchmark.json"
        if full_json.exists():
            with open(full_json, "w", encoding="utf-8") as f:
                json.dump(updated_records, f, ensure_ascii=False, indent=2)
            print(f"✅ Đã cập nhật {applied_count} câu vào {full_json.name}")

    # 2. Update rs_solve_full_benchmark_en.jsonl
    en_jsonl = data_dir / "rs_solve_full_benchmark_en.jsonl"
    if en_jsonl.exists():
        updated_en = []
        applied_en_count = 0
        with open(en_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                qid = rec.get("id")
                if qid in corrections_map:
                    new_c_vi = corrections_map[qid]["corr"]
                    old_c_vi = corrections_map[qid]["orig"]
                    new_c_en = CHOICE_MAP_EN.get(new_c_vi, new_c_vi)
                    old_c_en = CHOICE_MAP_EN.get(old_c_vi, old_c_vi)
                    rec["answer"] = new_c_en
                    if "choices" in rec and new_c_en not in rec["choices"]:
                        choices = [new_c_en if c == old_c_en else c for c in rec["choices"]]
                        rec["choices"] = choices
                    applied_en_count += 1
                updated_en.append(rec)

        with open(en_jsonl, "w", encoding="utf-8") as f:
            for r in updated_en:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"✅ Đã cập nhật {applied_en_count} câu vào {en_jsonl.name}")

        en_json = data_dir / "rs_solve_full_benchmark_en.json"
        if en_json.exists():
            with open(en_json, "w", encoding="utf-8") as f:
                json.dump(updated_en, f, ensure_ascii=False, indent=2)

    # 3. Update rs_solve_pilot_1000.jsonl
    pilot_jsonl = data_dir / "rs_solve_pilot_1000.jsonl"
    if pilot_jsonl.exists():
        updated_pilot = []
        applied_pilot_count = 0
        with open(pilot_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                qid = rec.get("id")
                if qid in corrections_map:
                    new_c = corrections_map[qid]["corr"]
                    old_c = rec.get("answer")
                    rec["answer"] = new_c
                    if "choices" in rec and new_c not in rec["choices"]:
                        choices = [new_c if c == old_c else c for c in rec["choices"]]
                        rec["choices"] = choices
                    applied_pilot_count += 1
                updated_pilot.append(rec)

        with open(pilot_jsonl, "w", encoding="utf-8") as f:
            for r in updated_pilot:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"✅ Đã cập nhật {applied_pilot_count} câu vào {pilot_jsonl.name}")

    # Save changelog
    changelog_path = data_dir / "q3_lf_corrections_changelog.json"
    with open(changelog_path, "w", encoding="utf-8") as f:
        json.dump({
            "source_file": str(input_path),
            "total_corrected": len(corrections_map),
            "corrections": corrections_map
        }, f, ensure_ascii=False, indent=2)
    print(f"📄 Đã lưu chi tiết nhật ký thay đổi vào: {changelog_path.name}")
    print("\n🎉 Hoàn tất! Tất cả các file benchmark đã được đồng bộ nhãn mới chuẩn xác 100%.")

if __name__ == "__main__":
    main()
