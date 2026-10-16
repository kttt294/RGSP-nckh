import json
import sys
from collections import defaultdict

models = {
    "GeoChat-7B": "results/geochat_full/preds_geochat.jsonl",
    "LLaVA-1.5-7B": "results/llava_1_5_full/preds_llava_1_5.jsonl",
    "Qwen3-VL-8B": "results/qwen3_full/preds_qwen3.jsonl",
    "InternVL3.5-8B": "results/internvl3_5_full/preds_internvl3_5.jsonl",
    "LLaVA-NeXT-7B": "results/llava_next_full/preds_llava_next.jsonl",
}

tasks_order = [
    "Q1: Hien dien",
    "Q2: Dem so luong",
    "Q3: Mau sac/Thuoc tinh",
    "Q4: Huong xoay / Dien tich",
    "Q5: Nhan dang loai con",
    "Q6: Dinh vi khong gian"
]

def map_task(qid):
    if "-Q1-" in qid: return "Q1: Hien dien"
    elif "-Q2-" in qid: return "Q2: Dem so luong"
    elif "-Q3-" in qid or "-Q3HF-" in qid or "-Q3LF-" in qid: return "Q3: Mau sac/Thuoc tinh"
    elif "-Q4-" in qid: return "Q4: Huong xoay / Dien tich"
    elif "-Q5-" in qid: return "Q5: Nhan dang loai con"
    elif "-Q6-" in qid: return "Q6: Dinh vi khong gian"
    return "Khac"

results = {}

for m_name, path in models.items():
    try:
        lines = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
        total = len(lines)
        corr = sum(1 for x in lines if x.get("is_correct"))
        acc = corr / total * 100 if total > 0 else 0
        
        task_accs = {}
        t_counts = defaultdict(lambda: {"total": 0, "corr": 0})
        for x in lines:
            t = map_task(x.get("id", ""))
            t_counts[t]["total"] += 1
            if x.get("is_correct"):
                t_counts[t]["corr"] += 1
        
        for t in tasks_order:
            cnt = t_counts[t]
            t_acc = cnt["corr"] / cnt["total"] * 100 if cnt["total"] > 0 else 0
            task_accs[t] = (cnt["corr"], cnt["total"], t_acc)
            
        results[m_name] = {
            "total": total,
            "corr": corr,
            "acc": acc,
            "tasks": task_accs
        }
    except Exception as e:
        print(f"Loi voi {m_name}: {e}")

print("=" * 105)
print(f"{'MO HINH':<18} | {'TONG ACC':<10} | {'Q1 (Hien dien)':<15} | {'Q2 (Dem)':<10} | {'Q3 (Mau/HF)':<12} | {'Q4 (OBB)':<10} | {'Q5 (Loai)':<10} | {'Q6 (Vi tri)':<12}")
print("=" * 105)
for m_name, r in results.items():
    t = r["tasks"]
    q1 = f"{t['Q1: Hien dien'][2]:.1f}%"
    q2 = f"{t['Q2: Dem so luong'][2]:.1f}%"
    q3 = f"{t['Q3: Mau sac/Thuoc tinh'][2]:.1f}%"
    q4 = f"{t['Q4: Huong xoay / Dien tich'][2]:.1f}%"
    q5 = f"{t['Q5: Nhan dang loai con'][2]:.1f}%"
    q6 = f"{t['Q6: Dinh vi khong gian'][2]:.1f}%"
    print(f"{m_name:<18} | {r['acc']:>6.2f}%    | {q1:>15} | {q2:>10} | {q3:>12} | {q4:>10} | {q5:>10} | {q6:>12}")
print("=" * 105)
