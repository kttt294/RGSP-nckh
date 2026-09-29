

MÔ TẢ CÁC TRƯỜNG DỮ LIỆU TRONG MỘT MẪU CÂU HỎI RS-SOLVE

Mẫu câu hỏi:
{
  "id": "RS-SOLVE-Q1-001",
  "kind": "Q1",
  "image_path": "images/dota_P1053.jpg",
  "gsd_m": 0.3,
  "question": "Trong toàn ảnh có Bể bơi ngoài trời (swimming_pool) không?",
  "choices": [
    "Có",
    "Không",
    "Không thể xác định từ ảnh"
  ],
  "answer": "Có",
  "class_id": "swimming_pool",
  "target_box_xyxy": [
    511.0,
    166.001,
    706.0,
    306.0
  ],
  "L_m": 39.9552772065217,
  "rho_px": 133.18425735507233,
  "p0_U_px": 6.0,
  "answerable_by_sensor": true,
  "rho_tok": null,
  "isolation_flag": "isolated"
}

---

MÔ TẢ CHI TIẾT TỪNG TRƯỜNG:

1. id
   Mã định danh duy nhất của câu hỏi. Dùng để đối chiếu và đánh giá kết quả của từng mô hình.
2. kind
   Loại tác vụ nhận thức của câu hỏi. Bao gồm:

- Q1: Phát hiện sự hiện diện của đối tượng.
- Q2: Phủ định gây nhầm lẫn (hỏi đối tượng vắng mặt khi trong ảnh có đối tượng gần giống).
- Q3-HF: Hướng trục xoay OBB của đối tượng.
- Q3-LF: Màu sắc đặc trưng của đối tượng.
- Q4: Đếm số lượng đối tượng.
- Q5: Định vị ô lưới 3x3 chứa đối tượng.
- Q6: Thử thách từ chối do mờ hoặc dưới ngưỡng phân giải.

3. image_path
   Đường dẫn tương đối đến tệp ảnh viễn thám đầu vào.
4. question
   Nội dung câu hỏi truyền cho mô hình AI. Với các câu hỏi định vị ô (Q3, Q4, Q5, Q6), câu hỏi sẽ kèm theo quy ước lưới 3x3 (cột A-C, hàng 1-3). Với các câu hỏi toàn ảnh (Q1, Q2), câu hỏi hỏi trực tiếp đối tượng mà không gắn tiền tố lưới.
5. choices
   Danh sách các phương án lựa chọn. Luôn bao gồm các đáp án khả dĩ và phương án từ chối "Không thể xác định từ ảnh".
6. answer
   Đáp án đúng (Ground Truth) dùng để chấm điểm AI.
7. class_id
   Tên lớp đối tượng theo chuẩn tiếng Anh (ví dụ: swimming_pool, small_vehicle).
8. target_box_xyxy
   Tọa độ hộp bao của đối tượng mục tiêu theo dạng [x1, y1, x2, y2] trên tọa độ pixel. Mang giá trị null đối với câu hỏi phủ định Q2.
9. gsd_m
   Độ phân giải mặt đất (Ground Sampling Distance) tính bằng mét/pixel, đọc tự động từ thông tin cảm biến của ảnh.
10. L_m
    Kích thước vật lý thực tế ngoài đời của vật thể tính bằng mét, tính bằng công thức: rho_px nhân với gsd_m.
11. rho_px
    Kích thước footprint cảm biến, đo bằng độ dài cạnh ngắn của đối tượng theo số pixel trên ảnh gốc.
12. p0_U_px
    Ngưỡng nhận thức tối thiểu theo tiêu chuẩn Johnson, tính bằng số pixel cần thiết để nhận biết đối tượng.
13. answerable_by_sensor
    Khả năng cảm biến có thể giải quyết được hay không:

- true: rho_px lớn hơn hoặc bằng p0_U_px (ảnh đủ độ nét để trả lời).
- false: rho_px nhỏ hơn p0_U_px (ảnh bị mờ hoặc vỡ hạt, bắt buộc phải chọn "Không thể xác định từ ảnh").

14. rho_tok
    Số lượng token mà đối tượng chiếm trong Vision Transformer. Mặc định là null khi sinh bộ dữ liệu, được đo động khi đưa vào từng mô hình cụ thể.
15. isolation_flag
    Trạng thái cách ly không gian của đối tượng:

- isolated: đứng riêng lẻ, không bị chen chúc.
- aggregated: nằm trong cụm đông đúc.
- not_applicable: áp dụng cho câu hỏi Q2 vì đối tượng không tồn tại trong ảnh.

---

CÁC TRƯỜNG BỔ SUNG THEO TỪNG LOẠI CÂU HỎI:

- target_cell: Tên ô lưới 3x3 chứa tâm đối tượng (ví dụ: A1, B2, C3).
- confused_present_class: Tên lớp đối tượng thực tế có trong ảnh gây nhầm lẫn với lớp được hỏi (ở câu hỏi Q2).
- unanswerable_reason: Nguyên nhân vật lý khiến câu hỏi không thể trả lời được từ ảnh (ở câu hỏi Q6).




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

| Thông số                                | Cơ chế tự động hóa                                                                                    |
| ----------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| **Quét ảnh (`images`)**         | Tự động quét file`.jpg`, `.png`, `.tif` trong thư mục `images/`.                              |
| **Tìm nhãn (`label_path`)**     | Tự động ánh xạ`images/{tên_ảnh}.jpg` sang `labels/{tên_ảnh}.txt`.                              |
| **Định dạng nhãn**              | Tự động nhận diện`yolo_obb` (chuẩn hóa) hoặc `dota` (8 tọa độ pixel).                        |
| **Lịch sử ảnh**                  | Mặc định ảnh nguyên bản của dataset (`transform_type = "identity"`, `scale = 1.0`).              |
| **Taxonomy & Chú giải**           | Tự động nạp 18 lớp canonical DOTA và trạng thái chú giải toàn diện (`exhaustive annotation`). |
| **Kích thước Q2 ($L_{\min}$)** | Tự động đọc và thẩm định trực tiếp từ`physical_sizes.json`.                                   |
| **Bảng màu mặc định**          | 6 nhóm màu chuẩn: Đỏ, Xanh lá, Xanh dương, Vàng, Trắng, Đen/Xám.                                |

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
