import sys
sys.stdout.reconfigure(encoding="utf-8")
import json
import math
from pathlib import Path
from collections import Counter, defaultdict

report_path = Path(r"c:\Users\trang\Desktop\rgsp_nckh\kaggle_benchmark_output\rs_solve_full_benchmark_build_report.json")
jsonl_path = Path(r"c:\Users\trang\Desktop\rgsp_nckh\kaggle_benchmark_output\rs_solve_full_benchmark.jsonl")
pool_path = Path(r"c:\Users\trang\Desktop\rgsp_nckh\kaggle_benchmark_output\rs_solve_full_benchmark_full_pool.jsonl")

print("=" * 80)
print("BÁO CÁO ĐÁNH GIÁ KHOA HỌC TẬP DỮ LIỆU RS-SOLVE BENCHMARK CÂN BẰNG ĐA NGUỒN (RUN V6)")
print("=" * 80)

if not report_path.exists() or not jsonl_path.exists():
    print(f"Lỗi: Không tìm thấy file dữ liệu tại {report_path} hoặc {jsonl_path}")
    sys.exit(1)

report = json.loads(report_path.read_text(encoding="utf-8"))

print(f"\n[1] TỔNG QUAN TẬP DỮ LIỆU (BUILD REPORT)")
print(f"  • Tổng số câu hỏi chọn lọc chuẩn hóa: {report.get('total_questions', 0):,} câu")
if pool_path.exists():
    print(f"  • Tổng số câu hỏi trong FULL POOL     : {report.get('total_pool_questions', 0):,} câu ({pool_path.stat().st_size / (1024*1024):.2f} MB)")
print(f"  • Số scenes/ảnh đã nạp và xử lý     : {report.get('scenes_processed', 0):,} ảnh")
print(f"  • Số lượng rejects (bị lọc bởi QA)     : {report.get('rejects_count', 0):,} lần")

kinds_counter = Counter()
dataset_counter = Counter()
answer_types = Counter()
kind_answer_table = defaultdict(Counter)
rho_px_values = []
rho_px_by_kind = defaultdict(list)
rho_px_bins = Counter()
density_counter = Counter()
johnson_counter = Counter()
class_counter = Counter()
grid_table = defaultdict(Counter)

total_lines = 0
sample_questions = defaultdict(list)

with jsonl_path.open(encoding="utf-8") as f:
    for line in f:
        total_lines += 1
        item = json.loads(line)
        kind = item.get("kind", "unknown")
        kinds_counter[kind] += 1
        
        img_p = item.get("image_path", "")
        ds_name = "unknown"
        if "dota" in img_p.lower():
            ds_name = "DOTA"
        elif "dior" in img_p.lower():
            ds_name = "DIOR"
        elif "visdrone" in img_p.lower():
            ds_name = "VisDrone"
        elif "xview" in img_p.lower():
            ds_name = "xView"
        elif "isaid" in img_p.lower():
            ds_name = "iSAID"
        dataset_counter[ds_name] += 1
        
        ans = item.get("answer", "")
        is_abstain = (ans == "Không thể xác định từ ảnh")
        ans_status = "Abstain (Không thể xác định)" if is_abstain else "Answerable"
        answer_types[ans_status] += 1
        kind_answer_table[kind][ans_status] += 1
        
        rho = item.get("rho_px", None)
        if rho is not None:
            rho_px_values.append(rho)
            rho_px_by_kind[kind].append(rho)
            
            if rho < 2:
                rbin = "< 2"
            elif rho < 4:
                rbin = "2-4"
            elif rho < 8:
                rbin = "4-8"
            elif rho < 16:
                rbin = "8-16"
            else:
                rbin = ">= 16"
            rho_px_bins[rbin] += 1
            
            dens = item.get("isolation_flag", "medium")
            density_counter[dens] += 1
            grid_table[rbin][dens] += 1
            
        j_lvl = item.get("johnson_level")
        if j_lvl:
            johnson_counter[j_lvl] += 1
            
        cls_name = item.get("class_id") or item.get("class_name")
        if cls_name:
            class_counter[cls_name] += 1
            
        if len(sample_questions[kind]) < 2:
            sample_questions[kind].append(item)

print(f"\n[2] PHÂN BỔ THEO LOẠI CÂU HỎI (KIND / QUESTION TYPE)")
print(f"{'Loại câu hỏi (Kind)':<20} | {'Số lượng':<10} | {'Tỷ lệ %':<8} | {'Answerable':<12} | {'Abstain':<10} | {'% Abstain':<10}")
print("-" * 80)
for k, count in sorted(kinds_counter.items(), key=lambda x: x[0]):
    pct = count / total_lines * 100
    ans_cnt = kind_answer_table[k]["Answerable"]
    abs_cnt = kind_answer_table[k]["Abstain (Không thể xác định)"]
    abs_pct = abs_cnt / count * 100 if count > 0 else 0
    print(f"{k:<20} | {count:<10,} | {pct:6.2f}% | {ans_cnt:<12,} | {abs_cnt:<10,} | {abs_pct:8.2f}%")
print("-" * 80)
print(f"{'TỔNG CỘNG':<20} | {total_lines:<10,} | 100.00% | {answer_types['Answerable']:<12,} | {answer_types['Abstain (Không thể xác định)']:<10,} | {answer_types['Abstain (Không thể xác định)']/total_lines*100:8.2f}%")

print(f"\n[3] PHÂN BỔ THEO DATASET NGUỒN")
for ds, count in dataset_counter.most_common():
    print(f"  • {ds:<12}: {count:6,} câu ({count/total_lines*100:5.2f}%)")

print(f"\n[4] THỐNG KÊ PHÂN BỐ VẬT LÝ RHO_PX (FOOTPRINT IN PIXELS)")
if rho_px_values:
    rho_sorted = sorted(rho_px_values)
    n = len(rho_sorted)
    min_v = rho_sorted[0]
    q25 = rho_sorted[int(0.25 * n)]
    median_v = rho_sorted[int(0.50 * n)]
    q75 = rho_sorted[int(0.75 * n)]
    max_v = rho_sorted[-1]
    mean_v = sum(rho_sorted) / n
    print(f"  • Số mẫu có rho_px   : {n:,} / {total_lines:,}")
    print(f"  • Min - Max           : {min_v:.2f} px - {max_v:.2f} px")
    print(f"  • Q25 - Median - Q75  : {q25:.2f} px | {median_v:.2f} px | {q75:.2f} px")
    print(f"  • Mean ± Std          : {mean_v:.2f} px")

print(f"\n[5] MA TRẬN PHÂN TẦNG THEO BẢNG III-b (RHO_PX x MẬT ĐỘ / ISOLATION)")
print(f"{'rho_px Bin':<12} | {'Thưa (isolated)':<16} | {'Trung bình':<12} | {'Dày (aggregated)':<18} | {'Tổng':<10}")
print("-" * 75)
bin_order = ["< 2", "2-4", "4-8", "8-16", ">= 16"]
for b in bin_order:
    iso = grid_table[b]["isolated"]
    med = grid_table[b]["medium"]
    agg = grid_table[b]["aggregated"]
    tot = iso + med + agg
    print(f"{b:<12} | {iso:<16,} | {med:<12,} | {agg:<18,} | {tot:<10,}")
print("-" * 75)

print(f"\n[6] PHÂN LOẠI TIÊU CHÍ JOHNSON (Q3-HF)")
for jlvl, jcnt in johnson_counter.most_common():
    print(f"  • {jlvl:<20}: {jcnt:6,} câu")

print(f"\n[7] ĐA DẠNG LỚP ĐỐI TƯỢNG (CLASSES)")
print(f"  • Tổng số lớp đối tượng: {len(class_counter)} lớp")
print(f"  • Top 10 lớp phổ biến nhất:")
for c, cnt in class_counter.most_common(10):
    print(f"     - {c:<25}: {cnt:5,} câu ({cnt/total_lines*100:5.2f}%)")

print(f"\n[8] PHÂN TÍCH LÝ DO REJECTS TRONG BUILD REPORT")
reject_reasons = Counter()
for r in report.get("rejects_sample", []):
    reject_reasons[r.get("reason", "unknown")] += 1
for reason, count in reject_reasons.most_common(10):
    print(f"  • {reason:<45}: {count} mẫu trong sample")
