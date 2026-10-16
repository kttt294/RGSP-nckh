# RGSP: Resolution-Grounded Selective Prediction for Remote Sensing MLLMs
> **Đề tài Nghiên cứu Khoa học & Bài báo IEEE TGRS**  
> Dự án: Benchmark RS-Solve & Cơ chế Dự đoán Chọn lọc Ràng buộc theo Phân giải Cảm biến (RGSP).

---

## 📂 Cấu Trúc Repository Chuẩn Hóa

Repository được tinh gọn và tổ chức chuyên nghiệp, tách biệt rõ ràng giữa các tài liệu khoa học báo cáo hội đồng, dữ liệu benchmark chính thức, kết quả thực nghiệm và các công cụ thực thi:

```text
rgsp_nckh/
├── docs/                                  # 1. Hồ sơ Khoa học & Thuyết minh Đề tài
│   ├── Ke_hoach_nghien_cuu_RGSP_RS-MLLM.docx  # Thuyết minh kế hoạch nghiên cứu chính thức
│   └── RGSP_Ban_thao_TGRS_VI.md           # Bản thảo bài báo khoa học chuẩn IEEE TGRS (Song ngữ/VI)
│
├── report_full.md                         # 2. Báo cáo Khoa học Tổng kết Đầy đủ
│                                          # Chứa toàn bộ Bảng số liệu phân tích: Bảng 5.1,
│                                          # Bảng VI (Ma trận rho_px x rho_tok), Bảng VII (H1),
│                                          # Bảng VIII (H2 - Ngưỡng phân giải), Bảng IX (H3 Johnson)
│
├── data/                                  # 3. Bộ Dữ liệu Benchmark Chính thức (RS-Solve)
│   ├── rs_solve_full_benchmark.jsonl      # Bản Full 30.771 câu hỏi (Tiếng Việt)
│   ├── rs_solve_full_benchmark_en.jsonl   # Bản Full 30.771 câu hỏi (Tiếng Anh)
│   ├── rs_solve_pilot_1000.jsonl          # Tập Pilot 1.000 câu chuẩn hóa để đo nhanh đa mô hình
│   ├── rs_solve_full_benchmark.json       # Định dạng JSON đầy đủ
│   ├── rs_solve_full_benchmark_en.json    # Định dạng JSON đầy đủ (English)
│   ├── rs_solve_full_benchmark_build_report.json # Báo cáo kiểm định cân bằng & chất lượng sinh
│   ├── q3_lf_pilot_verification_list.csv  # Bảng kê 143 câu Q3-LF Pilot phục vụ kiểm định thủ công
│   └── q3_lf_full_verification_list.csv   # Bảng kê 439 câu Q3-LF Full phục vụ kiểm định thủ công
│
├── results/                               # 4. Kết quả Thực nghiệm & Đo đạc Đã Hoàn thiện (30.771 câu x 5 Mô hình)
│   ├── geochat_full/                      # Kết quả Full của mô hình GeoChat-7B (CLIP-ViT@504)
│   ├── internvl3_5_full/                  # Kết quả Full của mô hình InternVL3.5-8B (InternViT)
│   ├── llava_1_5_full/                    # Kết quả Full của mô hình LLaVA-1.5-7B (CLIP-ViT@336)
│   ├── llava_next_full/                   # Kết quả Full của mô hình LLaVA-NeXT-7B (AnyRes Grid)
│   ├── qwen3_full/                        # Kết quả Full của mô hình Qwen3-VL-8B (SigLIP2)
│   └── cross_model_comparison_report.md   # BẢNG TỔNG HỢP SO SÁNH ĐA MÔ HÌNH (TABLES VI - X)
│
├── build_benchmark/                       # 5. Bộ Sinh Benchmark Tự động (Core Synthesizer)
│   ├── build_dataset.py                   # Engine sinh dữ liệu theo tiêu chuẩn vật lý Johnson
│   ├── core_generators.py                 # Hàm tạo câu hỏi Q1 - Q6, kiểm tra cô lập không gian & tính rho_px
│   ├── physical_sizes.csv & .json         # Kích thước vật lý thực địa L_m của từng lớp địa vật
│   ├── export_english_benchmark.py        # Xuất benchmark chuẩn tiếng Anh
│   ├── generate_q3_lf_automated.py        # Thuật toán trích xuất thuộc tính màu sắc Q3-LF
│   ├── parsers/                           # Parser nhãn chuẩn hóa cho 5 dataset: DIOR, DOTA, VisDrone, xView, iSAID
│   ├── sample_data/                       # Dữ liệu ảnh và nhãn mẫu phục vụ unit test
│   └── tests/                             # Bộ kiểm thử tự động (Unit tests)
│
└── scripts/                               # 6. Bộ Công cụ Kỹ thuật & Suy luận (Inference & Evaluation)
    ├── run_inference.py                   # Script chạy suy luận MLLM trên GPU / CPU
    ├── model_adapters.py                  # Bộ điều hợp 5 kiến trúc: GeoChat, Qwen3-VL, LLaVA-1.5, LLaVA-NeXT, InternVL3.5
    ├── compute_metrics.py                 # Engine tính toán chỉ số khoa học RGSP (Bảng VI, VII, VIII, IX, X)
    └── compare_all_5_models.py            # Tổng hợp nhanh độ chính xác 5 mô hình theo các tác vụ Q1 - Q6
```

---

## 🚀 Hướng Dẫn Sử Dụng Nhanh

### 1. Sinh Benchmark RS-Solve / RS-Resolve từ ảnh gốc
```bash
python build_benchmark/build_dataset.py --dataset all
```

### 2. Chạy Suy luận MLLM trên Benchmark
```bash
python scripts/run_inference.py \
    --model qwen \
    --benchmark-file data/rs_solve_full_benchmark_en.jsonl \
    --image-root /path/to/images \
    --output-file results/qwen3_full/preds_qwen3.jsonl
```

### 3. Tính toán Toàn bộ Bảng Chỉ số Khoa học RGSP (Tables VI - X)
```bash
# Đo lường và xuất báo cáo cho từng mô hình hoặc so sánh chéo 5 mô hình
python scripts/compute_metrics.py \
    --preds-file results/geochat_full/preds_geochat.jsonl \
                 results/llava_1_5_full/preds_llava_1_5.jsonl \
                 results/qwen3_full/preds_qwen3.jsonl \
                 results/internvl3_5_full/preds_internvl3_5.jsonl \
                 results/llava_next_full/preds_llava_next.jsonl \
    --output-dir results
```

### 4. Xem Tổng kết Nhanh Độ chính xác Tác vụ Q1 - Q6
```bash
python scripts/compare_all_5_models.py
```
