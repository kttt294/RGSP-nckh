# Báo cáo Phát triển & Tái cấu trúc Mã nguồn RS-Solve

**Đề tài:** Resolvability-Gated Selective Prediction for Remote-Sensing MLLMs (RGSP / RS-SOLVE)  
**Thời gian cập nhật:** 2026-09-29  
**Người thực hiện:** Antigravity Pair Programmer  

---

## 1. Mục tiêu & Tổng quan Kiến trúc

Nhằm giải quyết hạn chế dung lượng ổ cứng cục bộ (không thể tải toàn bộ hàng chục GB của 5 tập dữ liệu) đồng thời chuẩn hóa quy trình sinh benchmark câu hỏi trắc nghiệm kiểm soát vật lý theo tiêu chuẩn Johnson, hệ thống mã nguồn đã được tái cấu trúc từ dạng đơn thể (`build_samples.py`) sang mô hình **Đa tầng - Mô-đun hóa (Layered Modular Architecture)**:

```
rs-solve/
├── core_generators.py          # Trục lõi: Scene, Johnson threshold, footprint, Q1-Q6
├── build_dataset.py            # Master CLI Runner: tự phát hiện scene, sinh benchmark đa dataset
├── build_samples.py            # Cầu nối tương thích ngược (Backward Compatibility)
├── test_label_parser.py        # 34 bài kiểm thử đơn vị hồi quy kế thừa
├── test_parsers.py             # 5 bài kiểm thử đơn vị cho 5 bộ đọc chuyên biệt
├── physical_sizes.json         # Từ điển kích thước vật lý chuẩn hóa (80+ lớp, FAA/NATO/STANAG)
├── build_config.json           # Cấu hình ngưỡng hiệu chuẩn tâm vật lý RS-CALIB-001
├── parsers/                    # Gói chứa các parser chuyên biệt cho 5 bộ dữ liệu
│   ├── __init__.py             # Xuất các hàm đọc và bảng ánh xạ canonical
│   ├── parse_dota.py           # Bộ đọc DOTA v1.0/v1.5/v2.0 (OBB & YOLO-OBB)
│   ├── parse_isaid.py          # Bộ đọc iSAID (COCO Instance Segmentation Polygons)
│   ├── parse_dior.py           # Bộ đọc DIOR (Pascal VOC XML)
│   ├── parse_xview.py          # Bộ đọc xView (GeoJSON DIUx & HF 0-indexed)
│   └── parse_visdrone.py       # Bộ đọc VisDrone (TXT 8 cột AISKYEYE)
└── sample_data/                # Mẫu dữ liệu thực nghiệm siêu nhẹ (Tổng dung lượng: 2.66 MB)
    ├── dota/                   # 3 ảnh + 3 nhãn YOLO-OBB/DOTA
    ├── visdrone/               # 2 ảnh + 2 nhãn TXT 8 cột chính thức
    ├── dior/                   # 2 ảnh + 2 nhãn Pascal VOC XML
    ├── xview/                  # 2 ảnh + 1 nhãn GeoJSON
    └── isaid/                  # 1 ảnh + 1 nhãn COCO JSON phân đoạn
```

---

## 2. Bảng Theo dõi Triển khai Thành phần

| Thành phần | Đường dẫn | Trạng thái | Chi tiết kỹ thuật & Độ tương thích |
|---|---|:---:|---|
| **Dữ liệu mẫu (Sample Data)** | `sample_data/` | **HOÀN THÀNH** | Trích xuất từ remote HuggingFace/GitHub bằng range request, chỉ tốn **2.66 MB** cho 19 file. |
| **Trục lõi (Core Engine)** | `core_generators.py` | **HOÀN THÀNH** | `Scene`, `obb_metrics`, `polygon_area`, Sutherland-Hodgman clipping, Q1 -> Q6. Chuẩn hóa tiền tố lưới 3x3 (chỉ thêm ở Q3..Q6, bỏ ở Q1/Q2). |
| **Parser DOTA** | `parsers/parse_dota.py` | **HOÀN THÀNH** | Tự động phát hiện (Auto-detect) format YOLO-OBB (9 cột) hoặc DOTA gốc (10 cột), đọc header GSD. |
| **Parser iSAID** | `parsers/parse_isaid.py` | **HOÀN THÀNH** | Đọc đa giác phân đoạn COCO JSON, tính OBB bao nhỏ nhất, ánh xạ 15 lớp canonical. |
| **Parser DIOR** | `parsers/parse_dior.py` | **HOÀN THÀNH** | Phân tích Pascal VOC XML, chuyển đổi HBB sang OBB, ánh xạ 20 lớp DIOR. |
| **Parser xView** | `parsers/parse_xview.py` | **HOÀN THÀNH** | Hỗ trợ 2 chuẩn mã hóa: mã DIUx Challenge (11..94) và chỉ mục HuggingFace (0..59) khớp 100% với `physical_sizes.json`. |
| **Parser VisDrone** | `parsers/parse_visdrone.py` | **HOÀN THÀNH** | Đọc TXT 8 cột, lọc `score == 0`, đánh dấu `occlusion == 2` thành difficult, ánh xạ 10 lớp xe/người. |
| **Bộ điều khiển (CLI Runner)** | `build_dataset.py` | **HOÀN THÀNH** | Hỗ trợ `--dataset all\|dota\|visdrone\|dior\|xview\|isaid`, `--per-kind`, tự động dò tìm ảnh trong `sample_data/`. |
| **Kiểm thử mô-đun** | `test_parsers.py` | **HOÀN THÀNH** | 5/5 bài unit test cho 5 parser đều PASS tuyệt đối. |
| **Kiểm thử hồi quy** | `test_label_parser.py` | **HOÀN THÀNH** | 34/34 bài test kế thừa của `build_samples.py` tiếp tục PASS 100%. |

---

## 3. Chi tiết Kỹ thuật Các Bộ Đọc Dữ liệu (Parsers)

### 3.1. DOTA (`parsers/parse_dota.py`)
- **Đặc thù:** Tọa độ xoay 8 điểm $(x_1, y_1, \dots, x_4, y_4)$.
- **Cơ chế:** Tự động nhận diện dị bản YOLO-OBB (tọa độ chuẩn hóa $[0, 1]$, cột đầu là mã số lớp) hoặc DOTA gốc (tọa độ pixel tuyệt đối, cột 9 là tên lớp, cột 10 là cờ `difficult`).
- **GSD:** Tự động trích xuất thông số cảm biến từ dòng metadata header (`gsd:0.3`).

### 3.2. iSAID (`parsers/parse_isaid.py`)
- **Đặc thù:** Mặt nạ phân đoạn đa giác (COCO segmentation polygons).
- **Cơ chế:** Tìm kiếm đa giác theo `image_id` hoặc tên file, tính toán tâm hình học, diện tích bằng định lý Shoelace, và ma trận hiệp phương sai để xác định góc xoay và trục chính/phụ của vật thể.

### 3.3. DIOR (`parsers/parse_dior.py`)
- **Đặc thù:** Hộp bao ngang HBB trong Pascal VOC XML (`<bndbox><xmin>...`).
- **Cơ chế:** Chuyển đổi HBB sang 4 đỉnh OBB hình chữ nhật tiêu chuẩn góc $0^\circ$. Chuẩn hóa tên lớp Pascal VOC sang canonical (`golffield` $\rightarrow$ `golf_course`, `trainstation` $\rightarrow$ `train_station`, `storagetank` $\rightarrow$ `storage_tank`).
- **Thư mục ảnh:** Chuẩn Pascal VOC gốc quy ước tên `JPEGImages/` và `Annotations/`. Bộ quét `build_dataset.py` được thiết kế linh hoạt hỗ trợ cả 2 trường hợp: nhận diện tự động thư mục chuẩn hóa `images/` hoặc thư mục nguyên bản `JPEGImages/`.

### 3.4. xView (`parsers/parse_xview.py`)
- **Đặc thù:** GeoJSON FeatureCollection với trường `bounds_imcoords` dạng chuỗi `"xmin,ymin,xmax,ymax"`.
- **Cơ chế:** Hỗ trợ đa dạng nguồn dữ liệu: vừa hỗ trợ mã lớp 2 chữ số của cuộc thi DIUx xView gốc (11: Fixed-wing Aircraft, 18: Small Car, 29: Bus, ...), vừa hỗ trợ chỉ mục 0-indexed (0..59) khi người dùng tải từ Hugging Face (`HichTala/xview`).
- **GSD:** Cố định ở mức 0.3 m/pixel (ảnh cảm biến WorldView-3).

### 3.5. VisDrone (`parsers/parse_visdrone.py`)
- **Đặc thù:** Định dạng CSV/TXT 8 cột từ flycam AISKYEYE.
- **Cơ chế:** Lọc bỏ đối tượng ngoài vùng đánh giá (`score == 0` hoặc `category == 0`), chuyển đổi trạng thái che khuất `occlusion == 2` (>50%) thành `difficult = True`. Ánh xạ các đối tượng vi mô (`pedestrian`, `people`, `bicycle`, `motorcycle`, `awning_tricycle`).
- **GSD:** Ước tính từ độ cao bay trung bình của UAV (0.1 m/pixel).

---

## 4. Chuẩn hóa Câu hỏi & Quy ước Lưới Không gian 3x3

Theo kết quả thống nhất thiết kế:
1. **Câu hỏi toàn cục (Q1 - Phát hiện sự hiện diện, Q2 - Phủ định gây nhầm lẫn):**
   - Loại bỏ hoàn toàn tiền tố *"Chia ảnh thành lưới 3x3..."*.
   - Câu hỏi trực tiếp, ngắn gọn:
     - Ví dụ Q1: *"Trong toàn ảnh có Bể bơi ngoài trời (swimming_pool) không?"*
     - Ví dụ Q2: *"Trong toàn ảnh có Sân bóng rổ (basketball_court) không?"* (khi trong ảnh chỉ có sân tennis).
2. **Câu hỏi cục bộ theo ô (Q3-HF - Hướng trục xoay, Q3-LF - Màu sắc, Q4 - Đếm số lượng, Q5 - Định vị tọa độ, Q6 - Thử thách suy biến phân giải):**
   - Giữ nguyên tiền tố quy ước lưới: *"Chia ảnh thành lưới 3x3: cột A-C từ trái sang phải, hàng 1-3 từ trên xuống. ..."* để làm hệ quy chiếu tọa độ cho câu hỏi.

3. **Chuẩn hóa đường dẫn tương đối (Relative POSIX Path):**
   - Thay vì lưu đường dẫn tuyệt đối phụ thuộc vào cấu trúc thư mục của máy tính cá nhân (ví dụ `C:\Users\...\sample_data\...`), toàn bộ trường `image_path` trong câu hỏi và báo cáo `scene_provenance` được tự động chuyển thành **đường dẫn tương đối với định dạng POSIX (dấu gạch chéo `/`)**:
     - Ví dụ: `sample_data/dota/images/dota_P1053.jpg`, `sample_data/visdrone/images/0000001_03499_d_0000006.jpg`.
   - Điều này đảm bảo tính khả chuyển 100% khi đưa mã nguồn và tập dữ liệu lên các nền tảng đám mây như Kaggle, Google Colab hoặc máy chủ Linux mà không bị lỗi đường dẫn.

---

## 5. Kết quả Kiểm thử & Sinh Thử nghiệm

### 5.1. Kiểm thử Đơn vị (Unit Tests)
1. **Kiểm thử 5 Parser (`python -X utf8 test_parsers.py`):**
   ```
   Ran 5 tests in 0.028s
   OK
   ```
2. **Kiểm thử Hồi quy 34 bài test (`python -X utf8 -m unittest discover -s . -p test_label_parser.py`):**
   ```
   Ran 34 tests in 0.265s
   OK
   ```

### 5.2. Chạy Thực nghiệm Sinh Benchmark (`build_dataset.py`)
- **Chạy toàn bộ 5 Dataset kết hợp (`--dataset all --per-kind 1`):**
  - **Kết quả:** Đã tạo đủ bộ **7/7 câu hỏi mẫu** (Q1: 1, Q2: 1, Q3-HF: 1, Q3-LF: 1, Q4: 1, Q5: 1, Q6: 1).
  - **Tệp xuất ra:** `mau_all_5datasets.json`, `mau_all_5datasets.jsonl`, `mau_all_5datasets_build_report.json`.
- **Chạy kiểm tra độc lập từng dataset:**
  - `dota`: 7 câu hỏi (Q1..Q6 đầy đủ)
  - `isaid`: 5 câu hỏi (Q1, Q2, Q4, Q5, Q6)
  - `dior`: 4 câu hỏi (Q1, Q4, Q5, Q6)
  - `visdrone`: 4 câu hỏi (Q1, Q4, Q5, Q6)
  - `xview`: 3 câu hỏi (Q1, Q4, Q5)

---

## 6. Hướng dẫn Sử dụng (CLI Usage)

```bash
# 1. Chạy trên toàn bộ dữ liệu mẫu cả 5 dataset:
python -X utf8 build_dataset.py --dataset all --per-kind 1 --output-prefix mau_all_5datasets

# 2. Chạy riêng cho từng bộ dữ liệu:
python -X utf8 build_dataset.py --dataset dota --per-kind 2 --output-prefix mau_dota
python -X utf8 build_dataset.py --dataset visdrone --per-kind 2 --output-prefix mau_visdrone
python -X utf8 build_dataset.py --dataset dior --per-kind 2 --output-prefix mau_dior
python -X utf8 build_dataset.py --dataset xview --per-kind 2 --output-prefix mau_xview
python -X utf8 build_dataset.py --dataset isaid --per-kind 2 --output-prefix mau_isaid

# 3. Khi chạy trên Kaggle / Máy chủ với dữ liệu đầy đủ:
python -X utf8 build_dataset.py --dataset all --data-dir /path/to/full_datasets --per-kind 50 --output-prefix rs_solve_full_benchmark
```
