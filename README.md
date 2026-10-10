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
├── results/                               # 4. Kết quả Thực nghiệm & Đo đạc Đã Hoàn thiện
│   └── llava_1_5_full/                    # Kết quả Full 30.771 câu của mô hình LLaVA-1.5-7B
│       ├── preds_llava_1_5.jsonl          # Dự đoán chi tiết từng câu
│       ├── eval_report_preds_llava_1_5.md # Báo cáo số liệu độ chính xác & kiểm định giả thuyết
│       └── eval_report_preds_llava_1_5.json
│                                          # (Các kết quả mới sẽ được lưu vào đây khi chạy trên Server GPU)
│
├── rs-solve/                              # 5. Bộ Sinh Benchmark Tự động (Core Synthesizer)
│   ├── build_dataset.py                   # Engine sinh dữ liệu theo tiêu chuẩn vật lý Johnson
│   ├── core_generators.py                 # Hàm tạo câu hỏi Q1 - Q6, kiểm tra cô lập không gian
│   ├── physical_sizes.json                # Kích thước vật lý thực địa L_m của từng lớp địa vật
│   ├── parsers/                           # Parser nhãn chuẩn hóa cho 5 dataset: DIOR, DOTA, VisDrone, xView, iSAID
│   ├── sample_data/                       # Dữ liệu ảnh và nhãn mẫu phục vụ unit test
│   └── tests/                             # Bộ kiểm thử tự động (Unit tests)
│
└── scripts/                               # 6. Bộ Công cụ Kỹ thuật & Suy luận (Inference & Audit)
    ├── run_inference.py                   # Script chạy suy luận MLLM trên GPU (Hỗ trợ 8-bit)
    ├── model_adapters.py                  # Bộ điều hợp mô hình: GeoChat, Qwen2.5-VL, LLaVA-1.5, LLaVA-NeXT, InternVL2
    ├── compute_metrics.py                 # Engine tính toán chỉ số khoa học, S-curve và điểm chuyển pha
    ├── download_and_inspect_q3_lf.py      # Tải ảnh lẻ qua Kaggle API & crop đối tượng kiểm định nhãn
    ├── inspect.html                       # Giao diện web tương tác duyệt nhãn trực quan
    └── legacy_kaggle/                     # Lưu trữ các notebook Kaggle thử nghiệm ban đầu
```

---

## 🚀 Hướng Dẫn Sử Dụng Nhanh

### 1. Kiểm tra / Duyệt thủ công nhãn $Q_3\text{-LF}$
Mở trực tiếp file giao diện web:
```bash
# Mở bằng trình duyệt
start scripts/inspect.html
```

### 2. Chạy Suy luận MLLM (Local hoặc Remote Server GPU)
```bash
# Chạy suy luận cho Qwen2.5-VL trên tập Pilot 1.000 câu (hoặc tập Full)
python scripts/run_inference.py \
    --model qwen2.5-vl \
    --benchmark-file data/rs_solve_pilot_1000.jsonl \
    --image-root /path/to/images \
    --output-file results/preds_qwen2_5_vl.jsonl
```

### 3. Tính toán Toàn bộ Bảng Chỉ số Khoa học (Tables VI - X)
```bash
python scripts/compute_metrics.py \
    --preds-file results/pilot_eval/preds_qwen2_5_vl.jsonl \
    --output-dir results/pilot_eval \
    --p-patch 32
```
