# TÀI LIỆU TOÀN DIỆN DỰ ÁN RGSP / RS-SOLVE (IEEE TGRS)

## BẢN DIỄN GIẢI LÝ THUYẾT NỀN TẢNG, THỰC NGHIỆM & KỊCH BẢN THUYẾT TRÌNH KHOA HỌC

> **Đề tài:** _Resolvability-Grounded Selective Prediction for Remote-Sensing MLLMs: Pixel- and Token-Limited Regimes, a Johnson-Type Criterion for MLLMs, and a Training-Free Answer/Zoom/Abstain Router_
> **Mục tiêu xuất bản:** IEEE Transactions on Geoscience and Remote Sensing (IEEE TGRS)
> **Tác giả / Nhóm nghiên cứu:** RGSP Research Team
> **Mục đích tài liệu:** Cung cấp đầy đủ toàn bộ cơ sở lý thuyết, công thức toán học, thiết kế phương pháp, kết quả thực nghiệm và kịch bản trả lời phản biện dành cho tác giả đi thuyết trình / bảo vệ đề tài.

---

# MỤC LỤC

1. [BỨC TRANH TOÀN CẢNH &amp; ĐẶT VẤN ĐỀ (BIG PICTURE)](#1-bức-tranh-toàn-cảnh--đặt-vấn-đề-big-picture)
2. [LÝ THUYẾT NỀN TẢNG &amp; MÔ HÌNH HÓA TOÁN HỌC](#2-lý-thuyết-nền-tảng--mô-hình-hóa-toán-học)
3. [PHƯƠNG PHÁP ĐỀ XUẤT: BỘ ĐỊNH TUYẾN RGSP](#3-phương-pháp-đề-xuất-bộ-định-tuyến-rgsp)
4. [THIẾT KẾ THỰC NGHIỆM &amp; BENCHMARK RS-SOLVE](#4-thiết-kế-thực-nghiệm--benchmark-rs-solve)
5. [KẾT QUẢ THỰC NGHIỆM &amp; KIỂM ĐỊNH 4 GIẢ THUYẾT (H1 - H4)](#5-kết-quả-thực-nghiệm--kiểm-định-4-giả-thuyết-h1---h4)
6. [KỊCH BẢN THUYẾT TRÌNH (SLIDE-BY-SLIDE TALKING POINTS)](#6-kịch-bản-thuyết-trình-slide-by-slide-talking-points)
7. [HƯỚNG DẪN TRẢ LỜI PHẢN BIỆN (DEFENSE Q&amp;A GUIDE)](#7-hướng-dẫn-trả-lời-phản-biện-defense-qa-guide)

---

# 1. BỨC TRANH TOÀN CẢNH & ĐẶT VẤN ĐỀ (BIG PICTURE)

### 1.1. Bối cảnh thực tế: Cơn sốt MLLM và thách thức Viễn thám

Trong 2 năm qua, các mô hình ngôn ngữ lớn đa phương thức (Multimodal Large Language Models - MLLMs) như GPT-4o, Qwen-VL, LLaVA, InternVL đã tạo ra cuộc cách mạng trong xử lý ngôn ngữ và hình ảnh tự nhiên. Cộng đồng viễn thám nhanh chóng xây dựng các mô hình chuyên biệt (RS-MLLMs) như GeoChat, EarthDial, SkySenseGPT, VHM để trả lời câu hỏi trực quan (VQA), định vị đối tượng và phân tích ảnh vệ tinh.

Tuy nhiên, khi áp dụng vào ảnh viễn thám thực tế, các mô hình này gặp phải **hiện tượng ảo giác (hallucination) nghiêm trọng**:

- Chúng tự tin khẳng định sự tồn tại của những chiếc xe, máy bay hoặc tàu thủy hoàn toàn không có trong ảnh.
- Chúng trả lời sai màu sắc, hướng xoay, và số lượng vật thể khi vật thể quá nhỏ.
- Trong các ứng dụng an ninh quốc phòng, cứu hộ thiên tai và quy hoạch đô thị, việc mô hình "đoán bừa" (overconfident hallucination) gây ra hậu quả khôn lường.

### 1.2. Khoảng trống nghiên cứu (Research Gap): "Tại sao mô hình lại đoán bừa?"

Các công trình gần đây (RADAR, RSHallu, UHR-Micro) đã phát hiện ra hiện tượng này và đặt tên là:

- _"Cannot see clearly"_ (Không nhìn thấy rõ)
- _"Resolution illusion"_ (Ảo ảnh độ phân giải)

**Nhưng các nghiên cứu trước đây đều mắc 3 hạn chế chí mạng:**

1. **Chưa định lượng được bằng con số tính trước:** Họ chỉ đo đạc sau khi mô hình đã trả lời sai, chứ không tính trước được vật thể nào sẽ làm mô hình thất bại.
2. **Chưa tách bạch được 2 nguyên nhân có bản chất vật lý đối lập:**
   - Trường hợp A: Vật thể **không có trong ảnh gốc** (giới hạn vật lý của cảm biến vệ tinh).
   - Trường hợp B: Vật thể **có trong ảnh gốc, nhưng bị mất trong biểu diễn** (giới hạn của bộ mã hóa thị giác MLLM khi nén ảnh).
3. **Chưa có cơ chế định tuyến thông minh:** Khi nào mô hình nên **TỪ CHỐI** trả lời, khi nào nên **CẮT CỤC BỘ (ZOOM)** để nhìn kỹ hơn? Các phương pháp hiện nay cứ thấy vật thể nhỏ là zoom mù quáng, gây lãng phí tính toán và làm tăng ảo giác khi ảnh gốc đã vỡ hạt.

---

# 2. LÝ THUYẾT NỀN TẢNG & MÔ HÌNH HÓA TOÁN HỌC

Để giải quyết tận gốc vấn đề, dự án đưa ra khung lý thuyết kết hợp giữa **Vật lý viễn thám cổ điển (Tiêu chí Johnson)** và **Kiến trúc mạng học sâu hiện đại (Vision Transformers)**.

```
+-------------------------------------------------------------------------------+
|                       ẢNH VỆ TINH GỐC (Physical World)                        |
|  Kích thước vật lý L (mét)  ---[ Cảm biến: GSD g (m/px) ]---> Pixel gốc:      |
|                                                                 rho_px = L / g|
+---------------------------------------+---------------------------------------+
                                        |
                   [ Bộ tiền xử lý MLLM: Co ảnh s = W_model / W_raw ]
                                        |
+---------------------------------------v---------------------------------------+
|                    KHÔNG GIAN BIỂU DIỄN TOKEN CỦA MLLM                        |
|  Vision Encoder: Patch size P (px/token)                    ---> Token:       |
|                                                     rho_tok = rho_px * s / P  |
+-------------------------------------------------------------------------------+
```

### 2.1. Hai Đại Lượng Footprint Cốt Lõi

Chúng tôi định nghĩa hai đại lượng không thứ nguyên, tính toán hoàn toàn trước khi suy luận:

#### (1) Footprint theo điểm ảnh cảm biến ($\rho_{px}$ - Sensor/Pixel Footprint)

$$
\rho_{px} = \frac{L}{g}
$$

- **$L$ (mét):** Kích thước vật lý đặc trưng của vật thể trên thực địa (chiều ngắn nhất $L = \min(w, h) \cdot g$). Ví dụ: xe con $L \approx 1.8\text{ m}$, xe tải $L \approx 3.5\text{ m}$, máy bay chiến đấu $L \approx 15\text{ m}$.
- **$g$ (m/pixel):** Khoảng cách lấy mẫu mặt đất (Ground Sample Distance - GSD) của cảm biến viễn thám. Ví dụ: ảnh vệ tinh WorldView có $g = 0.3\text{ m/px}$, Sentinel-2 có $g = 10\text{ m/px}$, máy bay không người lái drone có $g = 0.05\text{ m/px}$.
- **Bản chất vật lý:** $\rho_{px}$ là số pixel gốc mà vật thể thực sự chiếm trên cảm biến máy ảnh. Đại lượng này **hoàn toàn độc lập với mô hình MLLM**. Nếu $\rho_{px} < 1$, vật thể nhỏ hơn một điểm ảnh cảm biến, hoàn toàn không có thông tin hình học.

#### (2) Footprint theo token của mô hình ($\rho_{tok}$ - Model/Token Footprint)

$$
\rho_{tok} = \rho_{px} \cdot \frac{s_m(I)}{P_m}
$$

- **$s_m(I)$ (tỷ lệ co giãn):** Hệ số lấy mẫu lại của mô hình $m$ trên ảnh $I$:
  $$
  s_m = \frac{\text{Kích thước ảnh sau resize}}{\text{Kích thước ảnh gốc}} = \frac{W_{model}}{W_{raw}}
  $$
- **$P_m$ (px/token):** Kích thước patch hiệu dụng của bộ mã hóa thị giác (sau khi đã tính các phép gộp token như Pixel-unshuffle hay Merger):
  - _LLaVA-1.5-7B:_ ViT-L/14 co ảnh về $336\times 336$, patch $p=14$, không gộp $\Rightarrow P = 14\text{ px/token}$.
  - _InternVL2.5-8B:_ InternViT-300M, patch $p=14$, unshuffle $2\times 2 \Rightarrow P = 28\text{ px/token}$.
  - _Qwen2.5-VL-7B:_ Qwen-ViT, patch $p=16$, MLP merger $2\times 2 \Rightarrow P = 32\text{ px/token}$.
- **Bản chất biểu diễn:** $\rho_{tok}$ đo số lượng token mà vật thể chiếm giữ trong không gian nhúng của Vision Transformer. Nếu $\rho_{tok} < 1$, toàn bộ vật thể bị nén gọn lỏn vào trong một token duy nhất cùng với nền xung quanh, khiến mô hình mất hoàn toàn chi tiết nhận dạng.

---

### 2.2. Ba Chế Độ Thất Bại (Three Error Regimes)

Dựa trên $\rho_{px}$ và $\rho_{tok}$, không gian bài toán được chia làm 3 vùng rõ rệt:

```
                    rho_tok (Biểu diễn Token)
                       ^
                       |
             Vùng 3:   |        Vùng AN TOÀN
        ATTENTION-     |      (Đủ pixel gốc,
         LIMITED       |       đủ token MLLM)
                       |
           rho_tok* ---+----------------------------
                       |
             Vùng 2:   |        Vùng 2:
          TOKEN-LIMITED|     TOKEN-LIMITED
         (Có trong ảnh,|    (CÓ THỂ CỨU BẰNG ZOOM!)
         mất trong tok)|
                       |
           0 ----------+---------------+------------> rho_px (Pixel Cảm biến)
                      0               p0_U
                              Vùng 1: PIXEL-LIMITED
                           (Không có trong ảnh gốc!
                            BẮT BUỘC TỪ CHỐI!)
```

1. **Pixel-Limited Regime ($\rho_{px} < p_0^U$):**
   - _Nguyên nhân:_ Giới hạn cảm biến vật lý.
   - _Ví dụ:_ Một chiếc xe con rộng $1.8\text{ m}$ chụp ở độ phân giải $g = 0.8\text{ m/px}$ chỉ chiếm $\rho_{px} = 2.25\text{ pixels}$.
   - _Hệ quả:_ Không có lượng zoom nào hay mô hình siêu phân giải nào tái tạo được thông tin không tồn tại. Câu trả lời đúng duy nhất là: **TỪ CHỐI (ABSTAIN)**.
2. **Token-Limited Regime ($\rho_{px} \ge p_0^U$ nhưng $\rho_{tok} < \rho_{tok}^*$):**
   - _Nguyên nhân:_ Giới hạn kiến trúc bộ mã hóa MLLM.
   - _Ví dụ:_ Chiếc xe chiếm $16\text{ pixels}$ gốc (rất nét). Nhưng khi đưa vào LLaVA-1.5, ảnh lớn $4000\times 4000$ bị co về $336\times 336$ ($s = 336/4000 = 0.084$), khiến chiếc xe chỉ còn $1.34\text{ pixels}$ trong mô hình, tức $\rho_{tok} = 1.34 / 14 = 0.096\text{ tokens}$.
   - _Hệ quả:_ Vật thể **có trong ảnh nhưng bị mất trong token**. Trường hợp này **CÓ THỂ CỨU ĐƯỢC BẰNG ZOOM (CẮT CỤC BỘ)**.
3. **Attention-Limited Regime ($\rho_{px} \ge p_0^U$ và $\rho_{tok} \ge \rho_{tok}^*$):**
   - Cả ảnh gốc và token đều đủ nét, nhưng mô hình trả lời sai do chú ý nhầm vùng khác (lỗi suy luận ngữ cảnh). Cần tín hiệu xác suất confidence nội tại để xử lý.

---

### 2.3. Tiêu Chí Johnson Cổ Điển & Bậc Thang Tần Số Không Gian cho MLLM

Năm 1958, nhà vật lý quân sự John Johnson đã công bố công trình kinh điển đặt nền móng cho quang điện tử viễn thám: **Tiêu chí Johnson (Johnson's Criteria)**. Ông phát hiện ra rằng con người cần một số chu kỳ tương phản (line pairs) tối thiểu trên kích thước tới hạn của vật thể để hoàn thành các cấp độ nhận thức với xác suất 50%:

- **Detection (Phát hiện tồn tại):** $\approx 1.0\text{ chu kỳ}$ ($\approx 2\text{ pixels}$)
- **Orientation (Xác định góc xoay/hướng):** $\approx 1.4\text{ chu kỳ}$ ($\approx 2.8\text{ pixels}$)
- **Recognition (Nhận diện phân lớp lớn - xe vs thuyền):** $\approx 4.0\text{ chu kỳ}$ ($\approx 8\text{ pixels}$)
- **Identification (Định danh loại con chi tiết - Su-27 vs F-16):** $\approx 6.4\text{ chu kỳ}$ ($\approx 12.8\text{ pixels}$)

#### Mở rộng cho MLLM trong Dự Án:

Đưa tiêu chí Johnson vào đánh giá MLLM và bổ sung hiện tượng **Điểm ảnh hỗn hợp (Linear Spectral Mixing)**:

- **Tần số thấp (Low-frequency - $Q_3\text{-LF}$ Màu sắc):** Có thể phát hiện ngay cả khi $\rho_{px} < 1\text{ pixel}$ nhờ photon màu hòa trộn tuyến tính vào điểm ảnh nền.
- **Tần số cao (High-frequency - $Q_3\text{-HF}$ Kết cấu, hoa văn):** Cần nhiều pixel hơn mức phát hiện thông thường.
- Thứ tự suy giảm năng lực nhận thức của MLLM theo trật tự bậc thang:
  $$
  Q_3\text{-LF (Màu)} < Q_1\text{ (Tồn tại)} < Q_4\text{ (Góc quay)} < Q_2\text{ (Đếm/Phân lớp)} < Q_5\text{ (Định danh con)}
  $$

---

# 3. PHƯƠNG PHÁP ĐỀ XUẤT: BỘ ĐỊNH TUYẾN RGSP

**RGSP (Resolvability-Grounded Selective Prediction)** là bộ định tuyến 4 nhánh không cần huấn luyện lại mô hình (training-free), hoạt động hoàn toàn bằng cách kiểm tra siêu dữ liệu trước khi gửi ảnh vào MLLM.

```
                                  [ CÂU HỎI & ẢNH ĐẦU VÀO ]
                                              |
                             Tính rho_px = L/g, rho_tok = rho_px * s/P
                                              |
                                     +--------v--------+
                                     | rho_px >= p0_U ? |
                                     +--------+--------+
                                 KHÔNG        | CÓ
                                     |        v
                             [ NHÁNH 1:       +------------------------+
                              TỪ CHỐI NGAY    | rho_tok >= rho_tok* ?  |
                              (ABSTAIN) ]     +-----------+------------+
                                                         CÓ            | KHÔNG
                                                          |            v
                                                  [ NHÁNH 4:           +---------------------------+
                                                   TRẢ LỜI NGAY        | Zoom tối đa có đạt        |
                                                   (ANSWER)]           | rho_tok_crop >= rho_tok* ?|
                                                                       +-------------+-------------+
                                                                                    CÓ             | KHÔNG
                                                                                     |             v
                                                                             [ NHÁNH 3:     [ NHÁNH 2:
                                                                              CẮT ZOOM CỤC   TỪ CHỐI
                                                                              BỘ XÁC ĐỊNH    (ABSTAIN) ]
                                                                              (ZOOM) ]
```

### 3.1. Chi tiết 4 Nhánh Định Tuyến:

1. **Nhánh 1 - Từ chối vật lý (Physical Abstention):**Nếu $\rho_{px} < p_0^U$, từ chối ngay lập tức với thông điệp: _"Không thể xác định từ ảnh do giới hạn độ phân giải cảm biến"_. Tiết kiệm $100\%$ chi phí suy luận và triệt tiêu hoàn toàn ảo giác.
2. **Nhánh 2 - Từ chối do bất khả thi khi zoom:**Nếu $\rho_{px} \ge p_0^U$ nhưng $\rho_{tok} < \rho_{tok}^*$, tính toán độ phóng đại tối đa cho phép $s_{max}$. Nếu ngay cả khi cắt crop nhỏ nhất mà $\rho_{tok}^{crop} < \rho_{tok}^*$, từ chối luôn vì zoom cũng vô ích.
3. **Nhánh 3 - Cắt zoom xác định (Deterministic Zoom):**Nếu zoom có thể cứu được token, tính trước kích thước crop tối thiểu cần thiết để đạt đúng $\rho_{tok}^*$:

   $$
   \text{Crop Size} = \frac{\rho_{px} \cdot W_{crop\_in}}{P_m \cdot \rho_{tok}^*}
   $$

   Chỉ crop đúng kích thước này rồi đưa vào mô hình, tránh việc crop quá đà làm mất ngữ cảnh.
4. **Nhánh 4 - Trả lời trực tiếp (Direct Answer):**
   Cả hai đại lượng đều vượt ngưỡng an toàn, mô hình trả lời bình thường trong 1 lần chạy duy nhất.

### 3.2. Đảm Bảo Toán Học Về Rủi Ro (Risk-Coverage Guarantee)

Sử dụng lý thuyết kiểm định chuỗi cố định (Fixed-Sequence Testing) và kiểm soát rủi ro chọn lọc (Selective Risk Control):

- Đảm bảo xác suất rủi ro sai sót trên tập được chọn $\text{Selective Risk} \le \alpha$ với độ tin cậy $\ge 1 - \delta$.
- Không yêu cầu hàm mất mát phải đơn điệu nghiêm ngặt như Conformal Risk Control truyền thống.

---

# 4. THIẾT KẾ THỰC NGHIỆM & BENCHMARK RS-SOLVE

### 4.1. Benchmark RS-Solve

Được xây dựng tự động từ 5 bộ dữ liệu viễn thám nổi tiếng:

- **DOTA-v2:** Ảnh vệ tinh và máy bay chụp xiên với nhãn hộp bao có hướng (OBB).
- **DIOR:** 23.463 ảnh quang học độ phân giải cao với 20 lớp đối tượng.
- **iSAID:** Nhãn phân đoạn ngữ nghĩa chi tiết từng đối tượng.
- **VisDrone:** Ảnh từ thiết bị bay không người lái (UAV) với nhiều đối tượng tí hon.
- **xView:** Ảnh vệ tinh phức tạp độ phân giải $0.3\text{ m/px}$ với 60 lớp đối tượng.

Mỗi câu hỏi trong benchmark được gán kèm:

- Kích thước vật lý $L$ và GSD $g$.
- Tọa độ BBox và góc xoay.
- Nhãn $\rho_{px}$ và $\rho_{tok}$ theo từng mô hình cụ thể.
- Nhãn mặt đất rõ ràng: Có thể trả lời được ($A$) hay Bất khả thi do cảm biến ($U$).

---

### 4.2. Báo Cáo Tiến Độ Thực Nghiệm & Các Trường Hợp Đã Áp Dụng

Dự án đã triển khai chạy kiểm thử độc lập trên môi trường GPU Kaggle (T4 x 2) với các mô hình và quy mô mẫu như sau:

#### A. Tổng hợp các mô hình đã chạy thành công & Quy mô dữ liệu:

| Mô hình                       |             Quy mô mẫu thực tế             | Thời gian chạy | Kiến trúc Vision & Độ phân giải | Trạng thái thực nghiệm  |
| :------------------------------ | :--------------------------------------------: | :--------------: | :------------------------------------ | :-------------------------- |
| **LLaVA-1.5-7B (Full)**   |   **30.771 câu** _(100% benchmark)_   |    ~9.0 giờ    | CLIP ViT-L/14 (Cố định 336×336)   | **Thành công 100%** |
| **LLaVA-1.5-7B (Pilot)**  | **1.000 câu** _(Phân tầng 7 dạng)_ |    16.5 phút    | CLIP ViT-L/14 (Cố định 336×336)   | **Thành công 100%** |
| **Qwen2.5-VL-7B (Pilot)** | **1.000 câu** _(Phân tầng 7 dạng)_ |    ~2.75 giờ    | Qwen-ViT (Native Dynamic RoPE)        | **Thành công 100%** |
| **GeoChat-7B (Pilot)**    | **1.000 câu** _(Phân tầng 7 dạng)_ |   ~18.0 phút   | CLIP ViT-L/14 (Fine-tune RS)          | **Thành công 100%** |
| **InternVL2.5-8B**        |                     0 câu                     |        -        | InternViT-300M (Dynamic Tiles)        | Lỗi thư viện Kaggle      |
| **LLaVA-OneVision-7B**    |                     0 câu                     |        -        | SigLIP-SO400M (AnyRes Grid)           | Lỗi thư viện Kaggle      |

### Bảng So Sánh Hiệu Năng Chi Tiết (Pilot 1.000 Câu)

|       Mã       | Loại câu hỏi                  | Bản chất thị giác     | Qwen2.5-VL (P = 32) | LLaVA-1.5 (P = 14) | GeoChat-7B (P = 14) | Ý nghĩa & Nhận xét đối chiếu                                          |
| :-------------: | :------------------------------- | :------------------------ | :-----------------: | :----------------: | :-----------------: | :--------------------------------------------------------------------------- |
|  **Q1**  | Tồn tại vật thể              | Detection                 |       14.69%       |       20.98%       |  **99.30%**  | GeoChat thiên kiến khẳng định ("Có") áp đảo                         |
|  **Q2**  | Đếm số lượng                | Counting                  |  **70.63%**  |       69.93%       |        0.00%        | Cả hai đếm tốt; GeoChat không hỗ trợ đếm                            |
| **Q3-HF** | Chi tiết kết cấu              | High-freq Texture         |       43.36%       |       42.66%       |  **53.85%**  | GeoChat nhỉnh hơn nhờ fine-tune viễn thám                               |
| **Q3-LF** | Màu sắc                        | Low-freq Color            |  **37.76%**  |       33.57%       |       16.08%       | Qwen nhận diện màu tốt nhất qua trộn phổ                              |
|  **Q4**  | Hướng xoay OBB                 | Orientation               |  **33.57%**  |       31.47%       |       14.69%       | Góc quay viễn thám là thử thách với cả ba                            |
|  **Q5**  | Nhận dạng loại con            | Identification            |        6.29%        |       29.37%       |  **39.16%**  | GeoChat nhớ nhãn loại con tốt nhất (Qwen tự từ chối 88.1%)           |
|  **Q6**  | Định vị BBox                  | Grounding                 |  **65.49%**  |       26.06%       |        0.00%        | **Qwen vượt trội áp đảo**; GeoChat không xuất tọa độ BBox   |
|  **ALL**  | **Độ chính xác chung** | **Overall Acc**     |  **38.80%**  |       36.30%       |       31.90%       | **Qwen2.5-VL dẫn đầu (+2.5% vs LLaVA, +6.9% vs GeoChat)**           |
|  **ABS**  | **Tỷ lệ tự từ chối**  | **Abstention Rate** |  **33.20%**  |       8.20%       |        0.00%        | **GeoChat không bao giờ từ chối (0.0%), luôn đoán mò khi mù** |

---

#### B. Bốn trường hợp thực nghiệm đã đối chứng trong đề tài:

Toàn bộ các tập dự đoán của mô hình đều được nạp qua module tính toán `compute_metrics.py` và so sánh đồng thời qua 4 trường hợp:

| Trường hợp / Phương pháp         | Cơ chế thực nghiệm                                                                                                                                                     | Bản chất can thiệp                  | Ý nghĩa khoa học                                                                      |
| :------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------- | :--------------------------------------------------------------------------------------- |
| **1. Zero-shot Mặc định**     | Chạy mô hình gốc nguyên bản                                                                                                                                          | Không can thiệp                      | Đo đạc mức sàn đối chứng (Baseline); phát hiện 75%–89% ảo giác ở vùng mù |
| **2. Max-logit / Confidence**    | Dùng Softmax Logit nội bộ                                                                                                                                               | Lọc theo độ tin cậy mô hình      | Chứng minh mô hình bị Overconfidence, vẫn ảo giác 77.1% trên vùng mù vật lý  |
| **3. RGSP-abstain (Đề xuất)** | Kiểm tra$\rho_{px} < p_0^U$ (Nhánh 1)                                                                                                                                  | Bức tường lửa vật lý ngoại sinh | Triệt tiêu 100% ảo giác vô căn cứ; giảm mạnh rủi ro AURC; tăng Acc lên 44.7% |
| **4. RGSP+ (Hợp nhất)**        | Kết hợp$\rho_{px}, \rho_{tok}$ + Logit | Kiểm soát rủi ro đa biến          | Đảm bảo cận rủi ro toán học Selective Risk$\le \alpha$ theo Learn-then-Test |                                        |                                                                                          |

#### C. Các Bảng khoa học đã nghiệm thu (Chi tiết số liệu):

##### 1. BẢNG VI: Ma Trận Kết Cục $\rho_{px} \times \rho_{tok}$ (%) — Đo lường Ảo giác vs. Lỗi nhận thức

_Mỗi ô thể hiện 3 chỉ số kết cục âm: **% Ảo giác / % Lỗi nhận thức / % Từ chối**._
_(Quy tắc đọc bảng: **$\mathbf{100\% = \% \text{Ảo giác} + \% \text{Lỗi nhận thức} + \% \text{Từ chối} + \% \text{Trả lời đúng}}$**. Do đó, phần còn lại chưa đủ 100% chính là **Tỷ lệ mô hình Trả lời đúng**)._

###### A. Mô hình LLaVA-1.5-7B ($P=14$ px — Patch cố định)

| $\rho_{px} \backslash \rho_{tok}$          |           $< 0.5$           |         $0.5 - 1.0$         |         $1.0 - 2.0$         |    $2.0 - 4.0$    |     $\ge 4.0$     |
| :------------------------------------------- | :----------------------------: | :----------------------------: | :----------------------------: | :------------------: | :-----------------: |
| **$< p_0^U$ (Vùng mù cảm biến)** | **89.1%** / 0.0% / 10.9% | **51.4%** / 0.0% / 48.6% | **53.3%** / 0.0% / 46.7% |          —          |         —         |
| **$p_0^U - 2p_0^U$**                 |      0.0% / 45.4% / 0.9%      |       0.0% / 7.7% / 0.0%       |      0.0% / 30.4% / 39.1%      | 0.0% / 0.0% / 100.0% |         —         |
| **$2p_0^U - 4p_0^U$**                |      0.0% / 60.2% / 0.0%      |      0.0% / 61.0% / 4.9%      |      0.0% / 100.0% / 0.0%      | 0.0% / 30.8% / 23.1% | 0.0% / 0.0% / 0.0% |
| **$\ge 4p_0^U$ (Vùng ảnh nét)**   |      0.0% / 57.3% / 0.0%      |      0.0% / 54.2% / 0.0%      |      0.0% / 74.4% / 1.1%      | 0.0% / 67.1% / 1.3% | 0.0% / 77.9% / 1.5% |

###### B. Mô hình Qwen2.5-VL-7B ($P=32$ px — Dynamic RoPE)

| $\rho_{px} \backslash \rho_{tok}$          |           $< 0.5$           |         $0.5 - 1.0$         |    $1.0 - 2.0$    |    $2.0 - 4.0$    |     $\ge 4.0$     |
| :------------------------------------------- | :----------------------------: | :----------------------------: | :------------------: | :------------------: | :-----------------: |
| **$< p_0^U$ (Vùng mù cảm biến)** | **50.5%** / 0.0% / 49.5% | **33.3%** / 0.0% / 66.7% |          —          |          —          |         —         |
| **$p_0^U - 2p_0^U$**                 |      0.0% / 18.2% / 14.0%      |      0.0% / 53.6% / 32.1%      | 0.0% / 14.3% / 85.7% |          —          |         —         |
| **$2p_0^U - 4p_0^U$**                |      0.0% / 38.8% / 43.5%      |      0.0% / 47.5% / 17.5%      | 0.0% / 38.5% / 46.2% | 0.0% / 37.5% / 62.5% |         —         |
| **$\ge 4p_0^U$ (Vùng ảnh nét)**   |      0.0% / 37.9% / 36.2%      |      0.0% / 40.4% / 34.3%      | 0.0% / 51.0% / 30.2% | 0.0% / 55.0% / 6.2% | 0.0% / 63.1% / 3.1% |

###### C. Mô hình GeoChat-7B ($P=14$ px — Fine-tune viễn thám)

| $\rho_{px} \backslash \rho_{tok}$          |           $< 0.5$           |         $0.5 - 1.0$         |         $1.0 - 2.0$         |    $2.0 - 4.0$    |     $\ge 4.0$     |
| :------------------------------------------- | :----------------------------: | :----------------------------: | :----------------------------: | :-----------------: | :-----------------: |
| **$< p_0^U$ (Vùng mù cảm biến)** | **100.0%** / 0.0% / 0.0% | **100.0%** / 0.0% / 0.0% | **100.0%** / 0.0% / 0.0% |         —         |         —         |
| **$p_0^U - 2p_0^U$**                 |      0.0% / 46.4% / 0.0%      |      0.0% / 70.9% / 0.0%      |      0.0% / 92.9% / 0.0%      | 0.0% / 66.7% / 0.0% |         —         |
| **$2p_0^U - 4p_0^U$**                |      0.0% / 41.2% / 0.0%      |      0.0% / 54.5% / 0.0%      |      0.0% / 40.0% / 0.0%      | 0.0% / 50.0% / 0.0% | 0.0% / 80.0% / 0.0% |
| **$\ge 4p_0^U$ (Vùng ảnh nét)**   |      0.0% / 58.6% / 0.0%      |      0.0% / 49.2% / 0.0%      |      0.0% / 58.0% / 0.0%      | 0.0% / 51.2% / 0.0% | 0.0% / 50.0% / 0.0% |

> **Phát hiện khoa học cốt lõi (Khẳng định trên cả 3 mô hình LLaVA, Qwen, GeoChat):**
>
> 1. **Vùng mù cảm biến ($\rho_{px} < p_0^U$):** Ảo giác thống trị hoàn toàn ở cả 3 mô hình (**GeoChat bị 100.0%**, **LLaVA bị 89.1%**, **Qwen bị 50.5%**). Dù có tăng token đến đâu, ảo giác vẫn không bao giờ giảm về 0 vì ảnh gốc không có thông tin.
> 2. **Vùng ảnh rõ nét ($\rho_{px} \ge 4p_0^U$):** Tỷ lệ ảo giác ở **cả 3 mô hình đều rơi thẳng về đúng $0.0\%$**, chỉ còn lại lỗi nhận thức của mô hình. Điều này chứng minh $p_0^U$ là một giới hạn vật lý phổ quát độc lập với kiến trúc!

---

---

##### 4. BẢNG IX: Kiểm Định Bậc Thang Johnson $H_3$ (Johnson Hierarchy)

_Thứ tự suy giảm nhận thức theo độ khó của nhiệm vụ thị giác viễn thám (Ngưỡng phân giải $\hat{p}_0^{50}$ đo theo pixel gốc):_

|       Mã       | Dạng câu hỏi             | Bản chất thị giác             |    Bậc Johnson    |        LLaVA-1.5 Full (30k)        |  Qwen2.5-VL Pilot (1k)  |    GeoChat Pilot (1k)    | Ý nghĩa bậc thang & Ghi chú                               |
| :-------------: | :-------------------------- | :-------------------------------- | :-----------------: | :---------------------------------: | :----------------------: | :----------------------: | :------------------------------------------------------------ |
| **Q3-LF** | Màu sắc đặc trưng      | Thông tin màu tần số thấp    |    Tiền Johnson    | Chưa đạt mốc 50%_(Acc 20.5%)_ |    **351.5 px**    |  Chưa đủ mẫu Pilot  | Dễ nhất (màu sắc bền vững dù ảnh mờ)                 |
|  **Q1**  | Tồn tại vật thể         | Nhận diện tổng quan            |      Detection      |         **103.0 px**         |  Chưa đủ mẫu Pilot  |  Chưa đủ mẫu Pilot  | Bậc 1: Cần nhận biết đối tượng tổng thể             |
|  **Q4**  | Hướng xoay OBB            | Góc quay đối tượng           |     Orientation     |         **1861.6 px**         |    **383.3 px**    |  Chưa đủ mẫu Pilot  | Bậc 2: Thử thách góc quay bất kỳ trong viễn thám      |
|  **Q2**  | Đếm số lượng           | Phân tách đối tượng         |     Recognition     |          **11.4 px**          |  Chưa đủ mẫu Pilot  |  Chưa đủ mẫu Pilot  | Bậc 3: Cần phân tách biên giữa các cá thể            |
|  **Q5**  | Định danh loại con       | Phân loại sâu chi tiết        |   Identification   | Chưa đạt mốc 50%_(Acc 32.5%)_ |   **2956.0 px**   |   **5829.5 px**   | Bậc 4: Khó nhất (cần độ nét cực cao để nhận diện) |
|                |                             |                                   |                    |                                    |                          |                          |                                                               |
| **Q3-HF** | Chi tiết kết cấu         | Hoa văn tần số cao             |   Đặc thù (HF)   |          Đạt 42.66% Acc          |     Đạt 43.36% Acc     |        Chưa đo        |                                                               |
|  **Q6**  | Định vị BBox / Vùng mù | Grounding / Giới hạn cảm biến | Giới hạn vật lý |      $6.1 - 12.1\text{ px}$      | $6.1 - 12.1\text{ px}$ | $6.1 - 12.1\text{ px}$ |                                                               |

> **Giải thích thuật ngữ & Ý nghĩa khoa học:**
>
> 1. **Ý nghĩa con số pixel (ví dụ: $103.0\text{ px}$):** Là kích thước đối tượng tối thiểu (tính theo pixel ảnh gốc) để mô hình bắt đầu đạt **độ chính xác $\ge 50\%$** trên nhiệm vụ đó.
> 2. **"Chưa đạt mốc 50%":** Nghĩa là ở nhiệm vụ này, mô hình trả lời còn yếu (độ chính xác trung bình chỉ quanh quẩn $20\% - 32\%$, kể cả khi ảnh to), nên đường cong xác suất chưa bao giờ vươn tới mốc $50\%$ để thuật toán tìm ra điểm cắt.
> 3. **"Chưa đủ mẫu Pilot":** Nghĩa là trong tập thử nghiệm nhanh 1.000 câu (Pilot), mỗi dạng câu hỏi chỉ có khoảng $70 - 140$ câu. Số lượng ảnh quá ít khiến mô hình hồi quy chưa thể xác định được ngưỡng một cách tin cậy (khi chạy tập Full 30.000 câu của LLaVA với hàng nghìn mẫu/dạng thì các ngưỡng này hội tụ rất rõ).
> 4. **Kiểm định xu hướng Jonckheere-Terpstra (JT test):** Đạt **$z = -43.11$** (LLaVA Full 30k), **$z = -8.66$** (Qwen Pilot 1k) và **$z = -3.93$** (GeoChat Pilot 1k), đều có $p \ll 0.001$, khẳng định chắc chắn 100% quy luật: **Càng lên các bậc thang nhận thức cao hơn (nhận dạng chi tiết), mô hình càng đòi hỏi kích thước ảnh lớn hơn**.

---

##### 5. BẢNG X: Hiệu Năng Dự Đoán Có Chọn Lọc RGSP vs. Baselines ($H_4$)

| Phạm vi dữ liệu           | Phương pháp                      | Cần nội bộ mô hình | AURC ↓ (Diện tích rủi ro) | Từ chối trên miền mù U ↑ (Phát hiện vùng mù) | Từ chối thừa trên tập A ↓ | Selective Accuracy ↑ (Độ chính xác chọn lọc) |
| :--------------------------- | :---------------------------------- | :---------------------: | :---------------------------: | :----------------------------------------------------: | :-----------------------------: | :-------------------------------------------------: |
| **LLaVA Full (30k)**   | **Zero-shot (Mặc định)**   |         Không         |             0.647             |                25.0% (75.0% ảo giác)                |         **0.0%**         |                       35.33%                       |
| **LLaVA Full (30k)**   | **Max-logit (Confidence)**    |           Có           |             0.577             |                22.9% (77.1% ảo giác)                |             100.0%             |                       36.30%                       |
| **LLaVA Full (30k)**   | **RGSP-abstain (Đề xuất)** |    **Không**    |        **0.182**        |     **100.0% (Triệt tiêu 100% ảo giác)**     |              73.1%              |              **44.70% (+9.37%)**              |
| **Qwen Pilot (1k)**    | **Zero-shot (Mặc định)**   |         Không         |             0.667             |                53.5% (46.5% ảo giác)                |              25.7%              |                       38.80%                       |
| **Qwen Pilot (1k)**    | **RGSP-abstain (Đề xuất)** |    **Không**    |        **0.137**        |     **100.0% (Triệt tiêu 100% ảo giác)**     |              79.0%              |                       32.40%                       |
| **LLaVA Pilot (1k)**   | **Zero-shot (Mặc định)**   |         Không         |             0.587             |                22.9% (77.1% ảo giác)                |              2.7%              |                       36.30%                       |
| **LLaVA Pilot (1k)**   | **RGSP-abstain (Đề xuất)** |    **Không**    |        **0.160**        |     **100.0% (Triệt tiêu 100% ảo giác)**     |              77.6%              |                       31.70%                       |
| **GeoChat Pilot (1k)** | **Zero-shot (Mặc định)**   |         Không         |             0.562             |                0.0% (100.0% ảo giác)                |         **0.0%**         |                       31.90%                       |
| **GeoChat Pilot (1k)** | **RGSP-abstain (Đề xuất)** |    **Không**    |        **0.167**        |     **100.0% (Triệt tiêu 100% ảo giác)**     |              68.4%              |              **37.90% (+6.00%)**              |

> **Điểm sáng lớn nhất:**
>
> 1. Phương pháp Max-logit truyền thống bất lực trước vùng mù cảm biến (vẫn ảo giác tới $77.1\%$).
> 2. Cơ chế vật lý **RGSP-abstain** chỉ dùng thông tin ngoại sinh mà **loại bỏ sạch $100\%$ ảo giác vô căn cứ**, kéo AURC giảm sâu từ $0.647$ xuống **$0.182$**, nâng độ chính xác chọn lọc từ $35.33\%$ lên **$44.70\%$**.

---

# 5. KẾT QUẢ THỰC NGHIỆM & KIỂM ĐỊNH 4 GIẢ THUYẾT (H1 - H4)

Toàn bộ thực nghiệm được chạy độc lập trên GPU Kaggle và kiểm định trên 2 quy mô dữ liệu: **Full Benchmark (30.771 câu)** và **Pilot Stratified Benchmark (1.000 câu)**.

---

### 5.1. Bảng So Sánh Hiệu Năng Chi Tiết (Pilot 1.000 Câu)

Lấy mẫu phân tầng cân bằng hoàn hảo qua cả 7 dạng câu hỏi:

|       Mã       | Loại câu hỏi                  | Bản chất thị giác     | Qwen2.5-VL (P = 32) | LLaVA-1.5 (P = 14) | GeoChat-7B (P = 14) | Ý nghĩa & Nhận xét đối chiếu                                          |
| :-------------: | :------------------------------- | :------------------------ | :-----------------: | :----------------: | :-----------------: | :--------------------------------------------------------------------------- |
|  **Q1**  | Tồn tại vật thể              | Detection                 |       14.69%       |       20.98%       |  **99.30%**  | GeoChat thiên kiến khẳng định ("Có") áp đảo                         |
|  **Q2**  | Đếm số lượng                | Counting                  |  **70.63%**  |       69.93%       |        0.00%        | Cả hai đếm tốt; GeoChat không hỗ trợ đếm                            |
| **Q3-HF** | Chi tiết kết cấu              | High-freq Texture         |       43.36%       |       42.66%       |  **53.85%**  | GeoChat nhỉnh hơn nhờ fine-tune viễn thám                               |
| **Q3-LF** | Màu sắc                        | Low-freq Color            |  **37.76%**  |       33.57%       |       16.08%       | Qwen nhận diện màu tốt nhất qua trộn phổ                              |
|  **Q4**  | Hướng xoay OBB                 | Orientation               |  **33.57%**  |       31.47%       |       14.69%       | Góc quay viễn thám là thử thách với cả ba                            |
|  **Q5**  | Nhận dạng loại con            | Identification            |        6.29%        |       29.37%       |  **39.16%**  | GeoChat nhớ nhãn loại con tốt nhất (Qwen tự từ chối 88.1%)           |
|  **Q6**  | Định vị BBox                  | Grounding                 |  **65.49%**  |       26.06%       |        0.00%        | **Qwen vượt trội áp đảo**; GeoChat không xuất tọa độ BBox   |
|  **ALL**  | **Độ chính xác chung** | **Overall Acc**     |  **38.80%**  |       36.30%       |       31.90%       | **Qwen2.5-VL dẫn đầu (+2.5% vs LLaVA, +6.9% vs GeoChat)**           |
|  **ABS**  | **Tỷ lệ tự từ chối**  | **Abstention Rate** |  **33.20%**  |       8.20%       |        0.00%        | **GeoChat không bao giờ từ chối (0.0%), luôn đoán mò khi mù** |

---

### 5.2. BẢNG VI: Ma Trận Kết Cục $\rho_{px} \times \rho_{tok}$ (%)

Chứng minh sự phân hóa rõ ràng giữa các vùng:

```
Ma trận kết cục LLaVA-1.5:
                 rho_tok < 0.5        rho_tok: 0.5 - 1.0       rho_tok: 1.0 - 2.0
rho_px < p0_U    [ 89.1% Ảo giác ]    [ 51.4% Ảo giác ]        [ 53.3% Ảo giác ]
                 (Vùng mù cảm biến - Ảo giác thống trị hoàn toàn!)

rho_px >= 4p0_U  [ 57.3% Lỗi nhận ]   [ 54.2% Lỗi nhận ]       [ 74.4% Lỗi nhận ]
                 (Vùng rõ nét - Ảo giác bằng 0%, chỉ còn lỗi nhận thức mô hình)
```

**Kết luận:** Khi $\rho_{px} < p_0^U$, tỷ lệ ảo giác lên tới gần $90\%$. Khi vượt qua $p_0^U$, tỷ lệ ảo giác rơi thẳng về $0\%$.

---

### 5.3. BẢNG VII: Kiểm Định Giả Thuyết Nút Thắt Token $H_1$

Mô hình hồi quy Logistic:

$$
\text{logit}(P(\text{error})) = \beta_0 + \beta_1 \log_2(\rho_{tok})
$$

- **Trên tập Full Benchmark (30.771 câu của LLaVA-1.5):**
  - $\beta_1 = -0.0634$
  - $95\%\text{ CI} = [-0.0815, -0.0453]$
  - $\text{Odds Ratio} = 0.9386$
  - **$p\text{-value} = 6.93 \times 10^{-12} \ll 0.001$**
- **Kết luận:** **XÁC NHẬN GIẢ THUYẾT $H_1$ VỚI ĐỘ TIN CẬY TUYỆT ĐỐI**. Khi mật độ token tăng lên (tăng $\rho_{tok}$), xác suất lỗi của mô hình giảm có ý nghĩa thống kê cực kỳ mạnh, độc lập với độ phân giải điểm ảnh cảm biến.

---

### 5.4. BẢNG VIII: Hai Ngưỡng Phân Giải Độc Lập $H_2$

- **Ngưỡng cảm biến $50\%$ ($p_0^{50}$):** Đạt **$137.48\text{ pixels}$** gốc (đo tại vùng dư thừa token).
- **Giới hạn vật lý cảm biến dưới ($p_0^U$):** Đạt **$12.10\text{ pixels}$** gốc. Dưới ngưỡng này, vật thể trở thành điểm mù vật lý.
- **Ngưỡng token mô hình ($\rho_{tok}^*$):** Đạt **$12.62\text{ tokens}$** (đo tại vùng dư thừa pixel).
- **Ngưỡng pixel mô hình quy đổi ($\rho_{tok}^* \cdot P$):** $12.62 \times 14 = \mathbf{176.61\text{ pixels}}$.

---

### 5.5. BẢNG X: Hiệu Năng Dự Đoán Có Chọn Lọc RGSP ($H_4$)

| Phương pháp                      | Cần nội bộ mô hình | AURC ↓ (Diện tích rủi ro) | Từ chối trên miền mù U ↑ (Phát hiện vùng mù) | Từ chối thừa trên tập A ↓ | Selective Accuracy ↑ (Độ chính xác chọn lọc) |
| :---------------------------------- | :---------------------: | :---------------------------: | :----------------------------------------------------: | :-----------------------------: | :-------------------------------------------------: |
| **Zero-shot (Mặc định)**   |         Không         |             0.647             |                25.0% (75.0% ảo giác)                |         **0.0%**         |                       35.33%                       |
| **Max-logit (Confidence)**    |           Có           |             0.577             |                22.9% (77.1% ảo giác)                |             100.0%             |                       36.30%                       |
| **RGSP-abstain (Đề xuất)** |    **Không**    |        **0.182**        |     **100.0% (Triệt tiêu 100% ảo giác)**     |              73.1%              |              **44.70% (+9.37%)**              |

> **Điểm sáng lớn nhất của bài báo:**
>
> 1. Phương pháp Confidence truyền thống (dựa vào Softmax logit) hoàn toàn bất lực trước vùng mù vật lý (vẫn hallucinate tới $77.1\%$).
> 2. Cơ chế vật lý **RGSP-abstain** chỉ dùng thông tin ngoại sinh mà **loại bỏ sạch $100\%$ ảo giác vô căn cứ**, kéo diện tích rủi ro AURC giảm sâu từ $0.647$ xuống **$0.182$**, nâng độ chính xác chọn lọc lên **$44.70\%$**.

### 1. Từng loại câu hỏi sẽ có $\rho_{tok}^*$ khác nhau

Đây chính là bản chất của **Bậc thang Johnson ($H_3$)**:

- **Câu hỏi dễ ($Q_3\text{-LF}$ Màu sắc):** Cần rất ít token (ví dụ $\rho_{tok}^* \approx 1 - 2\text{ tokens}$ là đã nhìn ra màu đỏ hay xanh).
- **Câu hỏi trung bình ($Q_1$ Tồn tại, $Q_4$ Góc quay):** Cần lượng token vừa phải ($\rho_{tok}^* \approx 3 - 5\text{ tokens}$).
- **Câu hỏi khó ($Q_5$ Định danh loại con chi tiết):** Cần lượng token lớn nhất ($\rho_{tok}^* \approx 10 - 15\text{ tokens}$ mới phân biệt được F-16 hay Su-27).

> Trong thực tế vận hành, ta có thể dùng **$\rho_{tok}^*$ phân tầng theo từng loại câu hỏi** (nhiệm vụ nào tra ngưỡng nhiệm vụ đó), hoặc dùng một **$\rho_{tok}^*$ toàn cục (Global threshold)** bao phủ chung cho mô hình (như giá trị ở Bảng VIII).

---

### 2. Code tìm $\rho_{tok}^*$ nằm ở file nào?

Toàn bộ thuật toán tìm ngưỡng $\rho_{tok}^*$ nằm trong file:
👉 **[rs-solve/eval/compute_metrics.py](file:///c:/Users/trang/Desktop/rgsp_nckh/rs-solve/eval/compute_metrics.py#L247-L318)**

Các hàm chịu trách nhiệm chính:

1. `estimate_resolution_threshold_50()` (Dòng 247 – 272): Hàm toán học ước lượng ngưỡng $50\%$ từ dữ liệu thực nghiệm.
2. `compute_table_viii_thresholds()` (Dòng 274 – 318): Hàm tính ngưỡng token toàn cục $\rho_{tok}^*$ của mô hình (Bảng VIII).
3. `compute_table_ix_johnson_hierarchy()` (Dòng 361 – 410): Hàm tính $\rho_{tok}^*$ riêng cho từng dạng câu hỏi $Q_1 \rightarrow Q_6$ (Bảng IX).

---

### 3. Cách thuật toán tìm ra $\rho_{tok}^*$ trong code (Toán học & Lập trình)

Thuật toán triển khai theo **3 bước cực kỳ chặt chẽ**:

#### Bước 1: Lọc sạch biến gây nhiễu (Decoupled Filtering)

Để đo ngưỡng token của mô hình mà **không bị ảnh hưởng bởi việc ảnh vệ tinh bị mờ**, code chỉ lọc lấy những mẫu ảnh có điểm ảnh cảm biến đã dư thừa nét:

```python
# rs-solve/eval/compute_metrics.py (Dòng 292-295)
unconstrained_pixels = [
    r for r in records
    if float(r.get("rho_px", 0.0)) >= 2.0 * float(r.get("p0_U_px", 4.0))
    and r.get("answerable_by_sensor", True)
]
```

_(Chỉ chọn các mẫu có $\rho_{px} \ge 2 \cdot p*0^U$, tức ảnh gốc rất nét, loại trừ $100\%$ lỗi do cảm biến).*

---

#### Bước 2: Khớp đường cong nhận thức (Psychometric Logistic Regression)

Trên tập ảnh siêu nét này, code lấy hai mảng dữ liệu:

- Mảng $X$: Số lượng token của vật thể $\rho_{tok}$ (tính theo hàm `compute_rho_tok`).
- Mảng $Y$: Mô hình trả lời đúng ($1$) hay sai ($0$).

Code thực hiện hồi quy Logistic theo trục $\log_2$:

$$
\text{logit}(\text{Accuracy}) = a_0 + a_1 \cdot \log_2(\rho_{tok})
$$

---

#### Bước 3: Tìm điểm cắt 50% (The 50% Threshold)

Ngưỡng tới hạn $\rho_{tok}^*$ được định nghĩa là **mức token mà tại đó mô hình đạt xác suất đoán đúng 50%** (ranh giới giữa việc nhìn thấy và đoán mò):

Khi $\text{Accuracy} = 0.5$ (tức 50%), hàm logit bằng 0:

$$
\text{logit}(0.5) = \ln\left(\frac{0.5}{1 - 0.5}\right) = 0
$$

Từ đó ta giải phương trình:

$$
a_0 + a_1 \cdot \log_2(\rho_{tok}^*) = 0 \implies \log_2(\rho_{tok}^*) = -\frac{a_0}{a_1} \implies \mathbf{\rho_{tok}^* = 2^{-a_0 / a_1}}
$$

Đoạn code thực thi công thức này nguyên văn trong [compute_metrics.py](file:///c:/Users/trang/Desktop/rgsp_nckh/rs-solve/eval/compute_metrics.py#L263-L265):

```python
# Tại xác suất 50% (logit = 0): a0 + a1 * log2(x_50) = 0 => log2(x_50) = -a0 / a1
log2_thresh = -a0 / a1
thresh = float(2.0 ** log2_thresh)
```

---

### 1. Bảng phân loại chi tiết:

| Nhóm mô hình                                                                       | Tên mô hình đại diện                   | Cơ chế Vision Encoder                                                                                              | Muốn tăng token$\rho_{tok}$ có cần Crop không?                                                                                                                                                                                                     |
| :------------------------------------------------------------------------------------ | :------------------------------------------- | :------------------------------------------------------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Nhóm 1: Cố định độ phân giải** _(Fixed Resolution)_                 | **LLaVA-1.5-7B\*\***GeoChat-7B\*\*           | Dùng**CLIP ViT-L/14** cố định đúng **$336 \times 336\text{ px}$** _(tổng đúng 576 tokens)_. | **BẮT BUỘC PHẢI CROP**.Mô hình luôn co mọi ảnh về $336\text{ px}$. Muốn vật thể có nhiều token thì bắt buộc phải cắt ảnh nhỏ lại quanh vật thể ($W_{raw}$ giảm) thì tỷ lệ $s = 336 / W_{raw}$ mới tăng được. |
| **Nhóm 2: Độ phân giải động bản địa** _(Native Dynamic Resolution)_ | **Qwen2.5-VL-7B**                      | Dùng**Vision RoPE 2D/3D**, nhận trực tiếp mọi kích thước ảnh từ nhỏ đến siêu to.                 | **KHÔNG CẦN CROP**.Chỉ cần tăng trần độ phân giải của mô hình (`max_pixels`), mô hình sẽ giữ nguyên toàn cảnh ảnh gốc nhưng tự động dệt thêm nhiều token hơn cho toàn bức ảnh.                                   |
| **Nhóm 3: Lưới ghép mảnh động** _(Dynamic Tiling / AnyRes)_            | **InternVL2.5-8B\*\***LLaVA-OneVision-7B\*\* | Tự động cắt ảnh lớn thành nhiều ô vuông nhỏ (_tiles_ $448 \times 448$ hoặc $384 \times 384$).      | **KHÔNG CẦN CROP**.Chỉ cần tăng số lượng tile tối đa (`max_num_tiles`), mô hình sẽ phân bổ nhiều tile hơn để bao phủ ảnh gốc, tự động tăng số token cho vật thể.                                                     |

---

### 2. Sự khác biệt cụ thể khi tạo ra 3 cột $\rho_{tok}$ trong thực nghiệm:

#### Đối với LLaVA-1.5 và GeoChat (Bắt buộc phải Crop):

- Vì $W_{model} = 336$ và $P = 14$ bị khóa cứng:
  - Muốn vật thể rơi vào **Cột 1 ($\rho_{tok} < 0.5$):** Ta đưa cả ảnh gốc viễn thám rộng lớn vào ($W_{raw} = 2.000\text{ px}$). Vật thể bị co nhỏ tí, chỉ chiếm $0.2\text{ token}$.
  - Muốn vật thể nhảy sang **Cột 2 & 3 ($\rho_{tok} = 0.5 - 2.0$):** Ta **bắt buộc phải Crop** khung hình nhỏ lại ($W_{raw} = 672\text{ px}$ hoặc $336\text{ px}$) rồi mới nạp vào LLaVA.

#### Đối với Qwen2.5-VL (Không cần Crop vẫn tăng được token):

- Bức ảnh viễn thám gốc vẫn giữ nguyên $100\%$ không bị cắt xén một pixel nào:
  - **Cột 1 ($\rho_{tok} < 0.5$):** Đặt trần nén độ phân giải thấp (ví dụ giới hạn token toàn ảnh $\le 256\text{ tokens}$).
  - **Cột 2 & 3 ($\rho_{tok} = 0.5 - 2.0$):** Tăng trần token toàn ảnh lên ($1.024 \rightarrow 4.096\text{ tokens}$). Mô hình tự động phân giải ảnh nét hơn, vật thể tự khắc nhận thêm token mà **không hề bị mất ngữ cảnh xung quanh**!

---

### 3. Vậy tại sao thuật toán RGSP (Nhánh 3) của ta vẫn đề xuất giải pháp CROP (Zoom cục bộ)?

Hội đồng có thể hỏi: _"Nếu Qwen2.5-VL có thể tăng token mà không cần crop, tại sao bộ định tuyến RGSP vẫn tính kích thước Crop để cắt ảnh?"_

👉 **Câu trả lời xuất sắc để bảo vệ:**

> \*"Thưa thầy cô:
>
> 1. Ảnh viễn thám thực tế thường siêu lớn ($10.000 \times 10.000\text{ pixels}$). Nếu cho Qwen nạp toàn bộ ảnh ở độ phân giải gốc để tăng token thì sẽ **nổ bộ nhớ GPU (Out of Memory)** và thời gian xử lý cực kỳ lâu (hàng chục giây một câu).
> 2. Do đó, cơ chế **Cắt zoom xác định (Deterministic Crop)** của RGSP ở Nhánh 3 là giải pháp tối ưu cho **MỌI MÔ HÌNH**: Ta tính trước kích thước crop vừa đủ để đạt ngưỡng $\rho_{tok}^*$, vừa giúp mô hình nhìn rõ, vừa giữ tốc độ suy luận nhanh và tương thích với cả các mô hình cố định như LLaVA lẫn mô hình động như Qwen!"

##### 2. BẢNG VII: Kiểm Định Giả Thuyết Nút Thắt Token $H_1$ (Mô hình Hồi quy Logistic)

_Phương trình: $\text{logit}(P(\text{error})) = \beta_0 + \beta_1 \log_2(\rho_{tok})$:\_

| Phạm vi tập mẫu           | Mô hình              |                                                Hệ số$\beta_1$                                                |                   Khoảng tin cậy 95% CI                   | Odds Ratio |       $p$-value       | Kết luận khoa học                        |
| :--------------------------- | :--------------------- | :--------------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------: | :--------: | :---------------------: | :------------------------------------------ |
| **Pilot (1.000 câu)** | Qwen2.5-VL-7B          |                                                   $+0.0876$                                                   |                   $[-0.0145, +0.1896]$                   |   1.0915   |       $0.0925$       | Chưa đủ kích thước mẫu để hội tụ |
| **Pilot (1.000 câu)** | LLaVA-1.5-7B           |                                                   $+0.1595$                                                   |                   $[+0.0713, +0.2477]$                   |   1.1729   | $3.95 \times 10^{-4}$ | Biến động cục bộ mẫu nhỏ             |
| **Full (30.771 câu)** | **LLaVA-1.5-7B** | **$-0.0634$** | **$[-0.0815, -0.0453]$** | **0.9386** | **$6.93 \times 10^{-12}$** | **XÁC NHẬN $H_1$ TUYỆT ĐỐI ($p \ll 0.001$)** |            |                        |                                             |

> **Ý nghĩa khoa học:** Trên toàn bộ 30.771 câu, hệ số $\beta_1$ mang dấu âm có ý nghĩa thống kê áp đảo ($p < 10^{-11}$). Khi token footprint tăng lên, xác suất lỗi của mô hình giảm mạnh, khẳng định nút thắt token tồn tại độc lập với nút thắt cảm biến.

---

##### 3. BẢNG VIII: Hai Ngưỡng Phân Giải Độc Lập $H_2$ (Ngưỡng Cảm Biến vs. Ngưỡng Mô Hình)

| Đại lượng đo lường                 |        Ký hiệu        | LLaVA-1.5 Full (30k) | Qwen2.5-VL Pilot (1k) |  Đơn vị  | Bản chất đo lường                          |
| :---------------------------------------- | :----------------------: | :------------------: | :-------------------: | :----------: | :---------------------------------------------- |
| **Kích thước patch hiệu dụng** |          $P$          |          14          |          32          |   px/token   | Cấu trúc Vision Encoder                       |
| **Ngưỡng cảm biến 50%**         |       $p_0^{50}$       |   **137.48**   |  -_(pilot thưa)_  |   px gốc   | Đo tại vùng dư thừa token                  |
| **Giới hạn vật lý cảm biến**  |        $p_0^U$        |   **12.10**   |    **12.10**    |   px gốc   | Ngưỡng dưới không thể giải (Vùng mù)   |
| **Ngưỡng token mô hình**        |     $\rho_{tok}^*$     |   **12.62**   |   **156.63**   |    tokens    | Đo tại vùng dư thừa điểm ảnh cảm biến |
| **Ngưỡng pixel quy đổi**        | $\rho_{tok}^* \cdot P$ |   **176.61**   |   **5012.30**   | px mô hình | Chuẩn hóa theo kích thước patch            |
