import sys
sys.stdout.reconfigure(encoding="utf-8")
import json
from pathlib import Path
from collections import Counter

rep_p = Path(r"c:\Users\trang\Desktop\rgsp_nckh\kaggle_final_benchmark\rs_solve_full_benchmark_build_report.json")
jsonl_p = Path(r"c:\Users\trang\Desktop\rgsp_nckh\kaggle_final_benchmark\rs_solve_full_benchmark.jsonl")

rep = json.loads(rep_p.read_text(encoding="utf-8"))

print("=" * 60)
print("BÁO CÁO THẨM ĐỊNH TẬP DỮ LIỆU BENCHMARK RS-SOLVE")
print("=" * 60)
print(f"Tổng số câu hỏi sinh được: {rep.get('total_questions', 0):,} câu")
print(f"Tổng số scene/ảnh đã xử lý: {rep.get('scenes_processed', 0):,} ảnh")
print(f"Tổng số ứng viên bị lọc bởi QA: {rep.get('rejects_count', 0):,} lần")

print("\n--- 1. Phân bổ số lượng câu hỏi theo từng loại (Kind) ---")
for k, v in rep.get("counts_by_kind", {}).items():
    print(f"  {k:8s}: {v:6,} câu")

print("\n--- 2. Phân bổ số lượng ảnh theo từng dataset ---")
ds_counts = Counter()
obj_counts = Counter()
for path, meta in rep.get("scene_provenance", {}).items():
    ds = meta.get("dataset", "unknown")
    ds_counts[ds] += 1
    obj_counts[ds] += meta.get("objects_count", 0)

for ds, count in ds_counts.most_common():
    print(f"  {ds:10s}: {count:5,} ảnh ({obj_counts[ds]:,} objects)")

print("\n--- 3. Kiểm tra tính toàn vẹn trên file JSONL ---")
kinds_in_jsonl = Counter()
abstain_count = 0
answerable_count = 0
classes_in_benchmark = Counter()

with jsonl_p.open(encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        kinds_in_jsonl[item["kind"]] += 1
        if item.get("answer") == "Không thể xác định từ ảnh":
            abstain_count += 1
        else:
            answerable_count += 1
        classes_in_benchmark[item.get("class_id")] += 1

print(f"Tổng số dòng trong JSONL: {sum(kinds_in_jsonl.values()):,}")
print(f"Số câu trả lời được (Answerable): {answerable_count:,} ({answerable_count/sum(kinds_in_jsonl.values())*100:.1f}%)")
print(f"Số câu từ chối vật lý (Abstain/Sensor Limit): {abstain_count:,} ({abstain_count/sum(kinds_in_jsonl.values())*100:.1f}%)")
print(f"Số lượng lớp đối tượng xuất hiện: {len(classes_in_benchmark)} lớp")
print("Top 10 lớp xuất hiện nhiều nhất:")
for c, cnt in classes_in_benchmark.most_common(10):
    print(f"  - {c:20s}: {cnt:5,} câu")

print("\n--- 4. Phân tích các lý do từ chối (Rejection/Skip QA) ---")
reject_reasons = Counter()
for r in rep.get("rejects_sample", []):
    reason = r.get("reason", "unknown")
    reject_reasons[reason] += 1
for r, cnt in reject_reasons.most_common(5):
    print(f"  - {r}")
