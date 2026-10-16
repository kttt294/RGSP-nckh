

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

---

## 6. Cơ chế chọn đối tượng và ảnh tạo câu hỏi (Q1 đến Q6)

Quá trình chọn đối tượng (object) và ảnh trong RS-Solve hoạt động theo nguyên lý **Quét toàn diện & Lọc đa tầng (Exhaustive Scan with Multi-stage Filtering)**. Hệ thống tự động bóc tách toàn bộ đối tượng trong ảnh, phân chia vào lưới không gian $3 \times 3$ (từ ô `A1` đến `C3`), sau đó lần lượt duyệt qua 6 bộ lọc chuyên biệt (`generate_q1` đến `generate_q5` và probe `generate_q6`).

Chỉ những đối tượng thỏa mãn 100% tiêu chí hình học, không gian và siêu dữ liệu kiểm soát chất lượng mới được chuyển thành mẫu câu hỏi; mọi trường hợp không đạt đều bị loại bỏ an toàn (`raise SkipSample`).

### 6.1. Quy tắc chi tiết cho từng loại câu hỏi

1. **Q1: Phát hiện sự hiện diện (Detection)**
   - *Phạm vi:* Toàn ảnh.
   - *Cơ chế chọn:* Với mỗi lớp xuất hiện trong ảnh, chọn **1 đối tượng có kích thước lớn nhất** (`minor_len_px` cực đại) làm đại diện.
   - *Điều kiện lọc:*
     - Phần nhìn thấy $\ge 50\%$ diện tích và không gắn cờ `difficult`.
     - Tính cô lập (`require_isolated`): Số cá thể cùng lớp trong ảnh $\le 3$, hoặc khoảng cách tới vật thể láng giềng gần nhất $\ge 4$ ViT tokens (tương đương $\approx 56\text{ px}$).

2. **Q2: Phủ định gây nhầm lẫn (Hard Negative Existence)**
   - *Phạm vi:* Toàn ảnh.
   - *Cơ chế chọn:* Quét danh mục các cặp lớp dễ gây nhầm lẫn thị giác (`CONFUSING_PAIRS`): *(airplane, helicopter)*, *(bridge, dam)*, *(baseball_diamond, ground_track_field)*, *(basketball_court, tennis_court)*...
   - *Điều kiện lọc:*
     - Trong ảnh **CÓ MẶT lớp gây nhầm lẫn ($P$)** nhưng **HOÀN TOÀN VẮNG MẶT lớp được hỏi ($A$)**.
     - Lớp vắng mặt $A$ phải được chứng minh đầy đủ thông qua trạng thái gán nhãn toàn diện (`require_complete`) hoặc biên bản kiểm duyệt vắng mặt (`absence_reviews`).
     - Có thông số kích thước vật lý tiên nghiệm $L_{\min}$ hợp lệ trong `physical_sizes.json`.

3. **Q3-HF: Hướng trục xoay (High-Frequency Orientation)**
   - *Phạm vi:* Từng ô lưới $3 \times 3$.
   - *Cơ chế chọn:* Quét từng đối tượng đơn lẻ trong ô.
   - *Điều kiện lọc:*
     - **Duy nhất trong ô (`require_unique_target`):** Trong ô đó chỉ có đúng 1 đối tượng thuộc lớp này.
     - **Có độ thuôn dài rõ nét:** Tỷ lệ cạnh ngắn / cạnh dài $\le 0.8$ (loại bỏ vật thể tròn hoặc gần vuông vì không thể xác định trục).
     - **Góc chéo an toàn:** Góc xoay OBB phải nằm trong khoảng $25^\circ \le \theta \le 65^\circ$ hoặc $115^\circ \le \theta \le 155^\circ$ (loại bỏ vùng đệm $\pm 25^\circ$ quanh trục ngang $0^\circ$ và trục dọc $90^\circ$ để tránh nhập nhằng ranh giới).

4. **Q3-LF: Màu sắc chủ đạo (Low-Frequency Color)**
   - *Phạm vi:* Từng ô lưới $3 \times 3$.
   - *Cơ chế chọn:* Quét từng đối tượng đơn lẻ trong ô.
   - *Điều kiện lọc:*
     - Là mục tiêu duy nhất thuộc lớp đó trong ô.
     - **Bắt buộc có xác nhận của con người:** Phải có bản ghi `color_reviews` trong metadata đạt trạng thái `"verified"` hoặc `"reviewed"` với nguồn trích xuất minh bạch. Không bao giờ tự động đoán màu nếu chưa có kiểm duyệt.

5. **Q4: Đếm số lượng trong ô (Counting)**
   - *Phạm vi:* Từng ô lưới $3 \times 3$.
   - *Cơ chế chọn:* Gom nhóm các đối tượng theo cặp `(ô lưới, lớp)`.
   - *Điều kiện lọc:*
     - Trong ô lưới phải có **tối thiểu 2 đối tượng** cùng lớp (`len(objects) >= 2`).
     - Lớp đối tượng này phải được đánh dấu đã gán nhãn đầy đủ trên toàn ảnh (`require_complete`).
     - Từng đối tượng phải nhìn rõ và có ít nhất $50\%$ diện tích nằm trong ô lưới (`in_cell`).

6. **Q5: Định vị tọa độ hộp bao (Grounding)**
   - *Phạm vi:* Từng ô lưới $3 \times 3$.
   - *Cơ chế chọn:* Gom nhóm các đối tượng theo cặp `(ô lưới, lớp)`.
   - *Điều kiện lọc:*
     - Phải có **ít nhất 2 đối tượng** cùng lớp trong ô để hỏi đối tượng thứ hai theo thứ tự từ trái sang phải.
     - Tâm hoành độ $X$ giữa các đối tượng liền kề phải cách nhau $> 1.0\text{ px}$ (tránh mơ hồ trật tự không gian).
     - Không gian ô lưới phải đủ rộng để thuật toán sinh ngẫu nhiên được ít nhất 2 hộp bao giả (distractor boxes) thỏa mãn $\text{IoU} \le 0.1$.

7. **Q6: Thử thách từ chối (Abstention Probe / Hard Sensor Boundary)**
   - *Phạm vi:* Trích xuất từ kho câu hỏi Q3-HF, Q4, Q5 đã được sinh.
   - *Cơ chế chọn:* Lọc các câu hỏi có kích thước điểm ảnh của đối tượng rơi vào đúng **vùng biên giới hạn phân giải của cảm biến**:
     $$0.5 \times p_0 \le \rho_{\text{px}} < p_0$$
   - *Ý nghĩa:* Đối tượng lúc này quá mờ dưới ngưỡng cảm biến Johnson $p_0$, Ground Truth bắt buộc là `ABSTAIN` ("Không thể xác định từ ảnh") nhằm kiểm tra khả năng tự nhận thức giới hạn và phát hiện ảo giác (hallucination) của MLLM.

### 6.2. Bảng tổng hợp tiêu chí chọn lọc

| Dạng câu hỏi | Phạm vi khảo sát | Điều kiện số lượng obj | Điều kiện hình học / Không gian đặc thù |
| :--- | :--- | :--- | :--- |
| **Q1 (Detection)** | Toàn ảnh | 1 obj lớn nhất / class | Cô lập ($\ge 4$ ViT tokens), nhìn rõ $\ge 50\%$ |
| **Q2 (Hard Negative)** | Toàn ảnh | Không có obj câu hỏi | Thuộc cặp nhầm lẫn, ảnh có $P$ và vắng mặt $A$ |
| **Q3-HF (Orientation)**| Từng ô $3 \times 3$ | Duy nhất 1 obj / class / ô | Tỷ lệ $w/h \le 0.8$; góc xoay $45^\circ \pm 20^\circ$ hoặc $135^\circ \pm 20^\circ$ |
| **Q3-LF (Color)** | Từng ô $3 \times 3$ | Duy nhất 1 obj / class / ô | Bắt buộc có Human Review trong metadata |
| **Q4 (Counting)** | Từng ô $3 \times 3$ | $\ge 2$ objs cùng lớp / ô | Lớp gán nhãn đầy đủ; mỗi obj nằm trong ô $\ge 50\%$ |
| **Q5 (Grounding)** | Từng ô $3 \times 3$ | $\ge 2$ objs cùng lớp / ô | Tâm X cách nhau $> 1\text{ px}$; sinh đủ 2 hộp distractor |
| **Q6 (Abstention)** | Kế thừa Q3/Q4/Q5| 1 câu hỏi tương ứng | Rơi vào vùng mờ vật lý ($0.5 p_0 \le \rho_{\text{px}} < p_0$) |

