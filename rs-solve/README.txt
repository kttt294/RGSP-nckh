MÔ TẢ CÁC TRƯỜNG DỮ LIỆU TRONG MỘT MẪU CÂU HỎI RS-SOLVE

Mẫu câu hỏi:
{
  "id": "RS-SOLVE-Q1-001",
  "kind": "Q1",
  "image_path": "images/dota_P1053.jpg",
  "gsd_m": 0.3,
  "question": "Chia ảnh thành lưới 3x3: cột A-C từ trái sang phải, hàng 1-3 từ trên xuống. Trong toàn ảnh có Bể bơi ngoài trời (swimming_pool) không?",
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

--------------------------------------------------
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
Nội dung câu hỏi truyền cho mô hình AI, luôn kèm theo quy ước lưới 3x3 (cột A-C, hàng 1-3).

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

--------------------------------------------------
CÁC TRƯỜNG BỔ SUNG THEO TỪNG LOẠI CÂU HỎI:

- target_cell: Tên ô lưới 3x3 chứa tâm đối tượng (ví dụ: A1, B2, C3).
- confused_present_class: Tên lớp đối tượng thực tế có trong ảnh gây nhầm lẫn với lớp được hỏi (ở câu hỏi Q2).
- unanswerable_reason: Nguyên nhân vật lý khiến câu hỏi không thể trả lời được từ ảnh (ở câu hỏi Q6).
