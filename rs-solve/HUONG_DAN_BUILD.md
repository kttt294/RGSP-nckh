# Hướng dẫn sinh tập câu hỏi RS-Solve

Tài liệu hướng dẫn cấu hình và chạy sinh tập dữ liệu câu hỏi trắc nghiệm RS-Solve theo chuẩn khoa học IEEE TGRS.

---

## 1. Cấu trúc cấu hình tối giản (`build_config.json`)

Toàn bộ các thông số kỹ thuật mà script có thể tự đọc hoặc có giá trị mặc định chuẩn (`label_path`, `transform_type`, `scale`, `complete_classes`, `color_palette`, `class_prior_approvals`) **đều đã được tự động hóa hoàn toàn**.

Dev chỉ cần điền **2 nhóm thông tin chính**:
1. **`calibration`**: Ngưỡng điểm ảnh $p_0$ thu được từ thực nghiệm hiệu chỉnh (detection, hard_negative, orientation, color_lf, counting, grounding).
2. **`GSD`**: Độ phân giải mặt đất (m/px). Nếu toàn dataset dùng chung thì khai báo `default_gsd_m`; nếu khác nhau thì khai báo theo từng ảnh trong `images`.

### Ví dụ cấu hình chuẩn:

```json
{
  "seed": 42,
  "calibration": {
    "id": "RS-CALIB-001",
    "status": "calibrated",
    "source": "Thực nghiệm hiệu chỉnh tâm vật lý RS-Solve",
    "thresholds_px": {
      "detection": 6.0,
      "hard_negative": 8.0,
      "orientation": 12.0,
      "color_lf": 4.0,
      "counting": 25.0,
      "grounding": 5.0
    }
  },
  "default_gsd_m": 0.25,
  "default_gsd_source": "DOTA-v2.0 metadata table",
  "images": {
    "images/dota_P1053.jpg": {
      "gsd_m": 0.3,
      "gsd_source": "DOTA metadata table",
      "color_reviews": {
        "4": {
          "status": "verified",
          "value": "Màu xanh dương",
          "source": "Kiểm tra trực quan ảnh tham chiếu mịn"
        }
      },
      "isolation_reviews": {
        "0": {
          "nearest_same_class_tokens": 14.8,
          "reference_view": "ViT-14",
          "source": "Khoảng cách đo được 207px / 14px = 14.8 tokens"
        }
      }
    },
    "images/dota_P1142.jpg": {
      "gsd_m": 0.25,
      "gsd_source": "DOTA metadata table"
    },
    "images/dota_P1470.jpg": {
      "gsd_m": 0.3,
      "gsd_source": "DOTA metadata table"
    }
  }
}
```

*Lưu ý:*
- Nếu để `"images": {}`, script sẽ **tự động quét toàn bộ ảnh** trong thư mục `images/`.
- `color_reviews`: Chỉ cần khai báo khi muốn sinh câu hỏi màu sắc Q3-LF trên ảnh không có sẵn nhãn thuộc tính màu.
- `isolation_reviews`: Khai báo khoảng cách láng giềng theo token khi muốn xác nhận vật thể đơn lẻ trong ảnh có nhiều cá thể cùng lớp.

---

## 2. Các cơ chế tự động hóa trong code

| Thông số | Cơ chế tự động hóa |
|---|---|
| **Quét ảnh (`images`)** | Tự động quét file `.jpg`, `.png`, `.tif` trong thư mục `images/`. |
| **Tìm nhãn (`label_path`)** | Tự động ánh xạ `images/{tên_ảnh}.jpg` sang `labels/{tên_ảnh}.txt`. |
| **Định dạng nhãn** | Tự động nhận diện `yolo_obb` (chuẩn hóa) hoặc `dota` (8 tọa độ pixel). |
| **Lịch sử ảnh** | Mặc định ảnh nguyên bản của dataset (`transform_type = "identity"`, `scale = 1.0`). |
| **Taxonomy & Chú giải** | Tự động nạp 18 lớp canonical DOTA và trạng thái chú giải toàn diện (`exhaustive annotation`). |
| **Kích thước Q2 ($L_{\min}$)** | Tự động đọc và thẩm định trực tiếp từ `physical_sizes.json`. |
| **Bảng màu mặc định** | 6 nhóm màu chuẩn: Đỏ, Xanh lá, Xanh dương, Vàng, Trắng, Đen/Xám. |

---

## 3. Chạy chương trình sinh câu hỏi

Trong PowerShell, chạy từ thư mục `rs-solve` hoặc root dự án:

```powershell
# Sinh bộ 7 mẫu câu hỏi mặc định:
python "C:\Users\trang\Desktop\rgsp_nckh\rs-solve\build_samples.py"

# Sinh quy mô lớn (ví dụ 5 mẫu mỗi loại = 35 câu):
python "C:\Users\trang\Desktop\rgsp_nckh\rs-solve\build_samples.py" --per-kind 5 --output-prefix mau_35_questions --allow-partial
```

*Các tham số hữu ích:*
- `--per-kind N`: Số câu hỏi tối đa cho mỗi nhóm câu hỏi (mặc định: 1).
- `--output-prefix <name>`: Tiền tố đặt tên cho file đầu ra (mặc định: `mau_7_questions`).
- `--allow-partial`: Cho phép xuất kết quả ngay cả khi một số nhóm câu chưa đạt đủ số lượng chỉ tiêu.

---

## 4. Định dạng đầu ra

Sau khi chạy, chương trình xuất ra 3 file:
1. `<prefix>.json`: Danh sách câu hỏi dạng JSON mảng có thụt lề, dùng để kiểm tra trực quan.
2. `<prefix>.jsonl`: Định dạng JSON Lines chuẩn để nạp trực tiếp vào pipeline kiểm thử mô hình VLM/MLLM.
3. `<prefix>_build_report.json`: Báo cáo thẩm định:
   - `selected_counts`: Số câu hỏi hợp lệ đã sinh theo từng loại (Q1–Q6).
   - `image_provenance`: Nguồn gốc, kích thước, GSD và phép biến đổi của từng ảnh.
   - `rejection_counts`: Thống kê các lý do loại bỏ ứng viên (quá gần ngưỡng, mơ hồ hướng, thiếu cách ly...).

---

## 5. Chạy kiểm thử tự động (Unit Tests)

```powershell
python -X utf8 -m unittest discover -s "C:\Users\trang\Desktop\rgsp_nckh\rs-solve" -p test_label_parser.py -v
```

Bộ kiểm thử bao gồm 34 test cases kiểm tra tính toàn vẹn hình học OBB, kiểm soát GSD, logic Johnson footprint, phân vùng lưới 3x3 và tính nhất quán đầu ra.
