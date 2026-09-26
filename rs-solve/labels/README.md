# HƯỚNG DẪN CẤU TRÚC DỮ LIỆU NHÃN (LABELS README)

**Thư mục:** `rs-solve/labels/`  
**Dự án:** RGSP Benchmark (RS-Resolve / RS-Solve)  
**Tiêu chuẩn định dạng:** DOTA v1.0 Oriented Bounding Box (OBB - Normalized YOLO format)

---

## 1. Mục đích và Vai trò trong Pipeline

Thư mục này chứa các file chú thích hình học (.txt) tương ứng với các ảnh viễn thám thực tế trong thư mục `../images/`.  
Các file nhãn này được nạp trực tiếp vào pipeline [`build_samples.py`](../build_samples.py) để:

1. **Phân tích hình học OBB:** Tính tâm vật thể $(c_x, c_y)$, xác định ô lưới 3x3 ($A1 \dots C3$), và tính góc xoay trục dài $\theta \pmod{180^\circ}$.
2. **Đối chiếu kích thước vật lý:** Liên kết mã lớp sang [`physical_sizes.json`](../physical_sizes.json) để tính $\rho_{px} = \frac{L_m}{\text{GSD}}$.
3. **Sinh câu hỏi & đáp án hoàn toàn tự động:** Quyết định sự tồn tại, lớp đối ngẫu gây nhiễu, số lượng đếm thực tế, và thứ tự không gian.

---

## 2. Cấu trúc Cú pháp của từng Dòng trong File Nhãn (.txt)

Mỗi dòng trong file nhãn đại diện cho một đối tượng (instance) trên ảnh, bao gồm **9 trường số** phân tách nhau bằng dấu cách (space):

```text
<class_index> <x1> <y1> <x2> <y2> <x3> <y3> <x4> <y4>
```

### Ý nghĩa chi tiết từng trường:

| Trường        | Kiểu dữ liệu      | Khoảng giá trị | Ý nghĩa                                                 |
| :------------ | :---------------- | :------------- | :------------------------------------------------------ |
| `class_index` | Số nguyên (`int`) | $0 \dots 14$   | Mã định danh lớp đối tượng theo chuẩn DOTA v1.0.        |
| `x1, y1`      | Số thực (`float`) | $[0.0, 1.0]$   | Tọa độ chuẩn hóa của **đỉnh thứ nhất** ($P_1$) của OBB. |
| `x2, y2`      | Số thực (`float`) | $[0.0, 1.0]$   | Tọa độ chuẩn hóa của **đỉnh thứ hai** ($P_2$) của OBB.  |
| `x3, y3`      | Số thực (`float`) | $[0.0, 1.0]$   | Tọa độ chuẩn hóa của **đỉnh thứ ba** ($P_3$) của OBB.   |
| `x4, y4`      | Số thực (`float`) | $[0.0, 1.0]$   | Tọa độ chuẩn hóa của **đỉnh thứ tư** ($P_4$) của OBB.   |

> **Quy ước tọa độ chuẩn hóa:**
>
> - Tọa độ pixel thực tế: $X_i = x_i \times W$, $Y_i = y_i \times H$ (với $W, H$ là chiều rộng và chiều cao ảnh, mặc định $1024 \times 1024$).
> - Thứ tự 4 đỉnh được đánh số xoay vòng tuần tự quanh chu vi OBB (thường theo chiều kim đồng hồ).

---

## 3. Bảng Ánh xạ Toàn bộ 18 Lớp Chuẩn DOTA-v2.0 sang Canonical Class ID (`physical_sizes.json`)

DOTA-v2.0 mở rộng từ DOTA-v1.0 (15 lớp) và DOTA-v1.5 (16 lớp) lên đầy đủ **18 lớp đối tượng**:

| `class_index` | Tên gốc DOTA-v2.0    | `class_id` chuẩn RS-Solve | Tên tiếng Việt (`display_name_vi`) | Ghi chú phiên bản    |
| :-----------: | :------------------- | :------------------------ | :--------------------------------- | :------------------- |
|     **0**     | `plane`              | `airplane`                | Máy bay cánh bằng                  | Có từ DOTA-v1.0      |
|     **1**     | `ship`               | `ship`                    | Tàu thủy / Thuyền                  | Có từ DOTA-v1.0      |
|     **2**     | `storage-tank`       | `storage_tank`            | Bồn chứa hình trụ tròn             | Có từ DOTA-v1.0      |
|     **3**     | `baseball-diamond`   | `baseball_diamond`        | Sân bóng chày                      | Có từ DOTA-v1.0      |
|     **4**     | `tennis-court`       | `tennis_court`            | Sân quần vợt                       | Có từ DOTA-v1.0      |
|     **5**     | `basketball-court`   | `basketball_court`        | Sân bóng rổ                        | Có từ DOTA-v1.0      |
|     **6**     | `ground-track-field` | `ground_track_field`      | Sân vận động điền kinh             | Có từ DOTA-v1.0      |
|     **7**     | `harbor`             | `harbor`                  | Cảng / Bến tàu                     | Có từ DOTA-v1.0      |
|     **8**     | `bridge`             | `bridge`                  | Cây cầu                            | Có từ DOTA-v1.0      |
|     **9**     | `large-vehicle`      | `large_vehicle`           | Xe tải lớn / Xe buýt               | Có từ DOTA-v1.0      |
|    **10**     | `small-vehicle`      | `small_vehicle`           | Xe con / Ô tô du lịch              | Có từ DOTA-v1.0      |
|    **11**     | `helicopter`         | `helicopter`              | Máy bay trực thăng                 | Có từ DOTA-v1.0      |
|    **12**     | `roundabout`         | `roundabout`              | Vòng xuyến giao thông              | Có từ DOTA-v1.0      |
|    **13**     | `soccer-ball-field`  | `soccer_ball_field`       | Sân bóng đá                        | Có từ DOTA-v1.0      |
|    **14**     | `swimming-pool`      | `swimming_pool`           | Bể bơi ngoài trời                  | Có từ DOTA-v1.0      |
|    **15**     | `container-crane`    | `container_crane`         | Cần cẩu container bến cảng         | Bổ sung từ DOTA-v1.5 |
|    **16**     | `airport`            | `airport`                 | Toàn cảnh sân bay                  | Bổ sung từ DOTA-v2.0 |
|    **17**     | `helipad`            | `helipad`                 | Bãi đáp trực thăng                 | Bổ sung từ DOTA-v2.0 |

---

## 4. Chi tiết Nội dung từng File Nhãn Cụ thể

### 4.1. File `dota_P1053.txt`

- **Ảnh tương ứng:** `../images/dota_P1053.jpg` (kích thước $1024 \times 1024$, GSD $\approx 0.30\,\text{m}$).
- **Nội dung ảnh:** Khu liên hợp thể thao và bãi đỗ xe gồm:
  - 8 sân quần vợt (tennis courts).
  - 1 khu bể bơi ngoài trời (swimming pool).
  - Các xe ô tô con đỗ trong bãi (small vehicles).
- **Phân tích chi tiết từng đối tượng trong file nhãn:**
  1. `Dòng 1`: Lớp `10` (`small_vehicle`) | $c_x \approx 153, c_y \approx 491$ $\implies$ Nằm ở ô **`A2`** (chiếc ô tô màu xanh đỗ góc dưới bãi). Dùng cho câu hỏi **Q6** (Từ chối cảm biến vì kích thước $7.2\,\text{px} < p_0^U = 12\,\text{px}$ khi nhận diện sedan vs hatchback).
  2. `Dòng 2 - 4`: Lớp `10` (`small_vehicle`) | $c_x \in [180, 280], c_y \in [180, 330]$ $\implies$ Cả 3 xe đều nằm trọn trong ô **`A1`** (bãi đỗ xe phía trên bên trái). Dùng cho câu hỏi **Q4** (Đếm ô lưới: chính xác có 3 xe).
  3. `Dòng 5`: Lớp `14` (`swimming_pool`) | $c_x \approx 608, c_y \approx 236$ $\implies$ Nằm ở ô **`B1`** (hồ bơi chữ nhật lớn và bể phụ). Dùng cho câu hỏi **Q3-LF** (Nhận diện màu sắc mặt nước: "Màu xanh dương").
  4. `Dòng 6 - 13`: Lớp `4` (`tennis_court`) | 8 sân tennis nằm trải dọc từ ô C1, C2, B2 đến A2/A3.
     - Dùng cho câu hỏi **Q2** (Phủ định khó: ảnh có `tennis_court` nhưng KHÔNG có `basketball_court` $\implies$ Đáp án: "Không").

---

### 4.2. File `dota_P1142.txt`

- **Ảnh tương ứng:** `../images/dota_P1142.jpg` (kích thước $1024 \times 1024$, GSD $\approx 0.25\,\text{m}$).
- **Nội dung ảnh:** Sân đỗ máy bay (apron) tại cảng hàng không.
- **Phân tích chi tiết đối tượng:**
  - `Dòng 1`: Lớp `0` (`airplane`)
    - 4 đỉnh OBB (pixel): `P1(798, 707)`, `P2(641, 462)`, `P3(911, 273)`, `P4(1024, 527)`.
    - Tâm đối tượng: $c_x = 843.5, c_y = 492.2 \implies$ Nằm trọn ở ô **`C2`**.
    - Chiều dài 2 cạnh kề OBB:
      - Cạnh $P_1P_2$: $\sqrt{(641-798)^2 + (462-707)^2} \approx 291.0\,\text{px}$
      - Cạnh $P_2P_3$: $\sqrt{(911-641)^2 + (273-462)^2} \approx 329.6\,\text{px}$ (Trục dài thân máy bay)
    - Góc nghiêng thân máy bay: $\theta = \text{atan2}(273 - 462, 911 - 641) \pmod{180^\circ} \approx 145.0^\circ$.
    - Vì $90^\circ \le 145^\circ < 180^\circ \implies$ Hướng trục chính là **"Tây Bắc - Đông Nam"**.
    - Sử dụng:
      - **Q1:** Phát hiện tồn tại máy bay cánh bằng ($\rho_{px} = 136.44\,\text{px} \ge 6.0\,\text{px} \implies$ Đáp án: "Có").
      - **Q3-HF:** Hướng trục dài thân máy bay $\implies$ Đáp án: "Tây Bắc - Đông Nam".

---

### 4.3. File `dota_P1470.txt`

- **Ảnh tương ứng:** `../images/dota_P1470.jpg` (kích thước $1024 \times 1024$, GSD $\approx 0.30\,\text{m}$).
- **Nội dung ảnh:** Khu liên hợp trường học/thể thao gồm 1 sân bóng đá và cụm 3 sân bóng rổ đặt song song.
- **Phân tích chi tiết từng đối tượng:**
  1. `Dòng 1`: Lớp `13` (`soccer_ball_field`) | $c_x \approx 449, c_y \approx 153 \implies$ Sân bóng đá ở ô **`A1 / B1`**.
  2. `Dòng 2 - 4`: Lớp `5` (`basketball_court`) | 3 sân bóng rổ xếp thành hàng ngang liên tiếp nhau ở ô **`C1`**:
     - _Sân 1 (bên trái):_ $x \in [675, 798]$, tâm $c_x \approx 736$ $\implies$ Bbox: `[675, 21, 798, 235]`.
     - _Sân 2 (chính giữa):_ $x \in [800, 924]$, tâm $c_x \approx 862$ $\implies$ Bbox: `[800, 21, 924, 235]`.
     - _Sân 3 (bên phải):_ $x \in [927, 1024]$, tâm $c_x \approx 975$ $\implies$ Bbox: `[927, 19, 1024, 233]`.
  - Sử dụng:
    - **Q5:** Tham chiếu không gian / thứ hạng vị trí ("Sân bóng rổ ở vị trí thứ hai từ trái sang phải") $\implies$ Đáp án: `"[800, 21, 924, 235]"`.

---

## 5. Quy tắc Tính toán Hình học trong `build_samples.py`

Khi đọc một dòng OBB từ file nhãn, script tự động thực hiện các phép biến đổi toán học:

```python
# 1. Tính tọa độ pixel từ normalized
pts = [(part[k] * W, part[k+1] * H) for k in range(1, 9, 2)]

# 2. Tính tâm đối tượng
cx = sum(p[0] for p in pts) / 4.0
cy = sum(p[1] for p in pts) / 4.0

# 3. Phân ô lưới 3x3
col = "A" if cx < W / 3.0 else ("B" if cx < 2.0 * W / 3.0 else "C")
row = "1" if cy < H / 3.0 else ("2" if cy < 2.0 * H / 3.0 else "3")
cell = f"{col}{row}"

# 4. Tính Axis-Aligned Bounding Box (AABB)
bbox = [min(xs), min(ys), max(xs), max(ys)]

# 5. Phân tích hướng OBB (2-bin classification)
dx1, dy1 = pts[1][0] - pts[0][0], pts[1][1] - pts[0][1]
dx2, dy2 = pts[2][0] - pts[1][0], pts[2][1] - pts[1][1]
# Lấy vector có độ dài lớn hơn làm trục chính
angle_deg = math.degrees(math.atan2(dy_major, dx_major)) % 180.0
direction = "Đông Bắc - Tây Nam" if (angle_deg < 90.0) else "Tây Bắc - Đông Nam"
```
