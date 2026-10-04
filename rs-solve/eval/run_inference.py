"""Bộ điều khiển chạy suy luận (Inference CLI Runner) cho các mô hình MLLM trên Benchmark RS-Solve.

Hỗ trợ chạy trên cả môi trường GPU (Kaggle / Local) và CPU / Mock.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import time

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
from collections import defaultdict
from typing import Dict, List, Optional
from PIL import Image

CURRENT_DIR = Path(__file__).resolve().parent
REPO_DIR = CURRENT_DIR.parent.parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
if str(REPO_DIR / "rs-solve") not in sys.path:
    sys.path.insert(0, str(REPO_DIR / "rs-solve"))

from model_adapters import get_model_adapter, BaseModelAdapter


_IMAGE_CACHE: Dict[str, Optional[Path]] = {}


def resolve_image_path(rel_path: str, image_root: Path) -> Optional[Path]:
    """Tìm đường dẫn ảnh thực tế trên ổ đĩa từ đường dẫn tương đối (tối ưu tốc độ cao)."""
    if not rel_path:
        return None
    if rel_path in _IMAGE_CACHE:
        return _IMAGE_CACHE[rel_path]

    # 1. Kiểm tra trực tiếp
    p_direct = Path(rel_path)
    if p_direct.is_file():
        _IMAGE_CACHE[rel_path] = p_direct
        return p_direct

    # 2. image_root / rel_path
    p1 = image_root / rel_path
    if p1.is_file():
        _IMAGE_CACHE[rel_path] = p1
        return p1

    # 3. REPO_DIR / rel_path
    p2 = REPO_DIR / rel_path
    if p2.is_file():
        _IMAGE_CACHE[rel_path] = p2
        return p2

    # 4. Biến thể tiền tố Kaggle: loại bỏ các tiền tố trùng lặp ('kaggle', 'input', 'datasets')
    parts = [p for p in rel_path.replace("\\", "/").split("/") if p and p not in ("kaggle", "input", "datasets")]
    if parts:
        p3 = image_root.joinpath(*parts)
        if p3.is_file():
            _IMAGE_CACHE[rel_path] = p3
            return p3

        # Kiểm tra trong từng thư mục con của image_root (/kaggle/input/<dataset>/...)
        if image_root.exists():
            for sub in image_root.iterdir():
                if sub.is_dir():
                    cand = sub.joinpath(*parts)
                    if cand.is_file():
                        _IMAGE_CACHE[rel_path] = cand
                        return cand

    _IMAGE_CACHE[rel_path] = None
    return None


def run_evaluation(
    model_adapter: BaseModelAdapter,
    benchmark_file: Path,
    image_root: Path,
    output_file: Path,
    max_samples: Optional[int] = None,
    resume: bool = True,
    stratified: bool = False
):
    print("=" * 80)
    print(f"BẮT ĐẦU CHẠY SUY LUẬN BENCHMARK CHO MÔ HÌNH: {model_adapter.model_name}")
    print(f"File Benchmark : {benchmark_file.name}")
    print(f"Thư mục ảnh    : {image_root}")
    print(f"File xuất      : {output_file.name}")
    if stratified:
        print(f"Chế độ lấy mẫu : Stratified sampling (cân bằng các loại câu hỏi)")
    print("=" * 80)

    # Đọc các mẫu đã chạy trước đó nếu bật chế độ resume
    completed_ids = set()
    if resume and output_file.exists():
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    completed_ids.add(data.get("id"))
                except Exception:
                    pass
        print(f"[+] Tìm thấy {len(completed_ids):,} mẫu đã hoàn thành trước đó. Sẽ chạy tiếp tục.")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    out_f = open(output_file, "a" if resume else "w", encoding="utf-8")

    # Đọc danh sách items cần chạy
    all_items = []
    with open(benchmark_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line)
                if item.get("id") not in completed_ids:
                    all_items.append(item)
            except Exception:
                pass

    if stratified:
        import random
        random.seed(42)
        by_kind = defaultdict(list)
        for it in all_items:
            by_kind[it.get("kind", "unknown")].append(it)
        for k in by_kind:
            random.shuffle(by_kind[k])

        # Xen kẽ đều đặn các loại câu hỏi (Q1, Q2, Q3-LF, Q3-HF, Q4, Q5, Q6)
        interleaved = []
        max_kind_len = max(len(v) for v in by_kind.values()) if by_kind else 0
        sorted_kinds = sorted(by_kind.keys())
        for step in range(max_kind_len):
            for k in sorted_kinds:
                if step < len(by_kind[k]):
                    interleaved.append(by_kind[k][step])

        if max_samples and max_samples < len(interleaved):
            all_items = interleaved[:max_samples]
        else:
            all_items = interleaved
    elif max_samples and max_samples < len(all_items):
        all_items = all_items[:max_samples]

    total_processed = len(completed_ids)
    total_correct = 0
    start_time = time.time()

    for idx, item in enumerate(all_items, 1):
        qid = item.get("id")
        img_rel = item.get("image_path", "")
        img_real = resolve_image_path(img_rel, image_root)

        # Nếu không tìm thấy ảnh gốc thì dùng ảnh mẫu trống để tránh dừng tiến trình
        if img_real is None or not img_real.is_file():
            # Tạo ảnh dummy 512x512
            img_real = output_file.parent / "dummy.jpg"
            if not img_real.exists():
                Image.new("RGB", (512, 512), color=(128, 128, 128)).save(img_real)

        try:
            with Image.open(img_real) as im:
                img_w, img_h = im.size
        except Exception:
            img_w, img_h = 512, 512

        rho_px = float(item.get("rho_px", 10.0))
        rho_tok = model_adapter.compute_rho_tok(rho_px, img_w, img_h)

        question = item.get("question", "")
        choices = item.get("choices", [])
        gt_answer = item.get("answer", "")

        # Gọi mô hình suy luận
        pred_res = model_adapter.predict(img_real, question, choices)
        pred_choice = pred_res.get("predicted_choice", "")

        is_correct = (pred_choice.strip().lower() == gt_answer.strip().lower())
        is_abstain = pred_choice in ("Cannot be determined from the image", "Không thể xác định từ ảnh")

        if is_correct:
            total_correct += 1

        record = {
            "id": qid,
            "kind": item.get("kind", ""),
            "class_id": item.get("class_id", ""),
            "rho_px": rho_px,
            "p0_U_px": float(item.get("p0_U_px", 4.0)),
            "rho_tok": rho_tok,
            "answerable_by_sensor": item.get("answerable_by_sensor", True),
            "ground_truth_answer": gt_answer,
            "predicted_answer": pred_choice,
            "predicted_letter": pred_res.get("predicted_letter"),
            "confidence": pred_res.get("confidence", 0.5),
            "raw_response": pred_res.get("raw_response", ""),
            "is_correct": is_correct,
            "is_abstain": is_abstain
        }

        out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
        out_f.flush()

        total_processed += 1
        if total_processed % 50 == 0:
            elapsed = time.time() - start_time
            speed = total_processed / max(1e-3, elapsed)
            print(f"  -> Đã suy luận: {total_processed:,} câu | Tốc độ: {speed:.1f} câu/s | Tạm tính Acc: {total_correct/max(1, total_processed):.1%}")

    out_f.close()
    elapsed = time.time() - start_time
    print("=" * 80)
    print(f"HOÀN THÀNH SUY LUẬN {total_processed:,} CÂU TRONG {elapsed:.1f}s!")
    print(f"Kết quả lưu tại: {output_file}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Chạy dự đoán MLLM trên Benchmark RS-Solve")
    parser.add_argument("--model", default="mock", help="Mã mô hình: qwen2.5-vl, internvl, llava-1.5, geochat, gpt-4o, gemini-1.5-pro, mock")
    parser.add_argument("--benchmark-file", default=str(REPO_DIR / "kaggle_benchmark_output" / "rs_solve_full_benchmark_en.jsonl"), help="File benchmark jsonl (khuyên dùng bản _en)")
    parser.add_argument("--image-root", default=str(REPO_DIR), help="Thư mục chứa ảnh gốc")
    parser.add_argument("--output-file", default=None, help="File lưu kết quả dự đoán jsonl")
    parser.add_argument("--max-samples", type=int, default=None, help="Số lượng mẫu tối đa cần chạy (ví dụ 1000 cho pilot test)")
    parser.add_argument("--stratified", action="store_true", help="Lấy mẫu phân tầng cân bằng các loại câu hỏi (Q1-Q6)")
    parser.add_argument("--no-resume", action="store_true", help="Chạy lại từ đầu, không đọc file cũ")
    parser.add_argument("--device", default="cuda", help="Thiết bị: cuda hoặc cpu")
    args = parser.parse_args()

    bench_path = Path(args.benchmark_file)
    if not bench_path.exists():
        print(f"[-] Không tìm thấy file benchmark tại {bench_path}")
        sys.exit(1)

    model_clean = args.model.lower().replace("-", "_").replace(".", "_")
    if args.output_file is None:
        out_path = REPO_DIR / "eval_outputs" / f"preds_{model_clean}.jsonl"
    else:
        out_path = Path(args.output_file)

    adapter = get_model_adapter(args.model, device=args.device)

    run_evaluation(
        model_adapter=adapter,
        benchmark_file=bench_path,
        image_root=Path(args.image_root),
        output_file=out_path,
        max_samples=args.max_samples,
        resume=(not args.no_resume),
        stratified=args.stratified
    )


if __name__ == "__main__":
    main()
