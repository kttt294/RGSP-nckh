# TỔNG HỢP NGUỒN DỮ LIỆU & QUY CHUẨN THÔNG SỐ VẬT LÝ CHO DỰ ÁN RGSP
*(Được đối chiếu và chuẩn hóa 100% theo hai tài liệu gốc: `Ke_hoach_nghien_cuu_RGSP_RS-MLLM.docx` và `RGSP_Ban_thao_TGRS_VI.md`)*

---

## A. CẢ 5 BỘ DỮ LIỆU NỀN TẢNG (RS-RESOLVE BENCHMARK)

*(Tất cả 5 bộ dữ liệu đều công khai, miễn phí cho nghiên cứu học thuật; tuân thủ nghiêm ngặt điều khoản không phân phối lại ảnh gốc).*

### 1. DOTA-v2.0 (Dataset for Object Detection in Aerial Images)
* **Trích dẫn chuẩn trong bài báo TGRS:**
  * J. Ding, N. Xue, G.-S. Xia et al., *"Object Detection in Aerial Images: A Large-Scale Benchmark and Challenges,"* *IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)*, 2021.
* **Đặc tính kỹ thuật:** GSD biến thiên $0.1 - 4.5\,\text{m}$; nhãn OBB (Oriented Bounding Box) dùng để xác định trục định hướng (orientation).
* **Tình trạng bản quyền:** Miễn phí cho nghiên cứu học thuật (Đại học Vũ Hán - CAPTAIN Lab).
* **Link truy cập chính thức:**
  * Trang chủ tải dữ liệu: [captain-whu.github.io/DOTA/dataset.html](https://captain-whu.github.io/DOTA/dataset.html)
  * Bộ công cụ hỗ trợ (Devkit): [github.com/CAPTAIN-WHU/DOTA_devkit](https://github.com/CAPTAIN-WHU/DOTA_devkit)
* **Dung lượng:** Khoảng $30 - 40\,\text{GB}$.

---

### 2. iSAID (Large-scale Dataset for Instance Segmentation in Aerial Images)
* **Trích dẫn chuẩn trong bài báo TGRS:**
  * S. Waqas Zamir, A. Arora, A. Gupta et al., *"iSAID: A Large-scale Dataset for Instance Segmentation in Aerial Images,"* *IEEE/CVF CVPR Workshops (CVPRW)*, 2019.
* **Đặc tính kỹ thuật:** Dùng chung tập ảnh gốc với DOTA-v2.0 (GSD $0.1 - 4.5\,\text{m}$), cung cấp mặt nạ phân vùng (mask) chính xác đến từng pixel để đo diện tích thực và màu sắc ($Q3\text{-LF}$).
* **Tình trạng bản quyền:** Miễn phí cho nghiên cứu học thuật.
* **Link truy cập chính thức:**
  * Trang chủ dự án: [captain-whu.github.io/iSAID](https://captain-whu.github.io/iSAID/)
  * Bộ công cụ hỗ trợ: [github.com/CAPTAIN-WHU/iSAID_DevKit](https://github.com/CAPTAIN-WHU/iSAID_DevKit)

---

### 3. xView (DIUx / NGA xView Dataset)
* **Trích dẫn chuẩn trong bài báo TGRS:**
  * D. Lam, R. Kuzma, K. McGee et al., *"xView: Objects in Context in Overhead Imagery,"* *arXiv:1802.07856*, 2018.
* **Đặc tính kỹ thuật:** GSD cố định $0.3\,\text{m}$ (ảnh vệ tinh WorldView-3); gồm 60 lớp đối tượng chi tiết có thông số kích thước vật lý cụ thể, đóng vai trò hạt nhân xây dựng các câu hỏi phân loại loại con ($Q3\text{-HF}$).
* **Tình trạng bản quyền & Thủ tục:** Đăng ký tài khoản tại cổng DIU.
* **Link truy cập chính thức:**
  * Cổng đăng ký tải dữ liệu: [challenge.xviewdataset.org/data-download](https://challenge.xviewdataset.org/data-download)
  * Kho mã nguồn chính thức (Baseline & Label mappings): [github.com/DIUx-xView/xView1_baseline](https://github.com/DIUx-xView/xView1_baseline)
* **Dung lượng:** Khoảng $60\,\text{GB}$.

---

### 4. DIOR (Object Detection in Optical Remote Sensing Images)
* **Trích dẫn chuẩn trong bài báo TGRS:**
  * K. Li, G. Wan, G. Cheng, L. Meng, J. Han, *"Object Detection in Optical Remote Sensing Images: A Survey and a New Benchmark,"* *ISPRS Journal of Photogrammetry and Remote Sensing*, 2020.
* **Đặc tính kỹ thuật:** Đại diện cho dải GSD thô ($0.5 - 30\,\text{m}$) gồm 20 danh mục lớp và 23.463 bức ảnh kích thước $800 \times 800$.
* **Tình trạng bản quyền:** Miễn phí cho nghiên cứu học thuật (Đại học Công nghiệp Tây Bắc - NWPU).
* **Link truy cập chính thức:**
  * Trang chủ nghiên cứu & Tải dữ liệu chính thức: [gcheng-nwpu.github.io/#Datasets](https://gcheng-nwpu.github.io/#Datasets)
  * Bản mirror tốc độ cao trên HuggingFace: [huggingface.co/datasets](https://huggingface.co/datasets) (tìm kiếm `DIOR`)

---

### 5. VisDrone (VisDrone-Dataset)
* **Trích dẫn chuẩn trong bài báo TGRS:**
  * P. Zhu, L. Wen, D. Du et al., *"Detection and Tracking Meet Drones Challenge,"* *IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)*, 2021.
* **Đặc tính kỹ thuật:** Ảnh góc nhìn flycam (UAV); đại diện cho cực trị mật độ vật thể nhỏ (crowded small objects); GSD được ước lượng từ độ cao bay.
* **Tình trạng bản quyền:** Miễn phí cho nghiên cứu học thuật (AISKYEYE Team - Đại học Thiên Tân).
* **Link truy cập chính thức:**
  * Kho GitHub chính thức: [github.com/VisDrone/VisDrone-Dataset](https://github.com/VisDrone/VisDrone-Dataset)

---

## B. NGUYÊN TẮC CHUẨN HÓA THÔNG SỐ VẬT LÝ ($L_c^{\min}$, $L_c^{\text{typ}}$)

1. **Chuẩn hóa trường `critical_dimension_type` về 4 ENUM cố định:**
   - `"width"`: Chiều rộng thân/bề rộng vệt (xe cộ, sân tennis, bóng đá, cầu đường, đường băng...).
   - `"diameter"`: Đường kính ngoài (vật thể tròn: bồn chứa dầu, bùng binh, ống khói, tuabin gió, sân vận động...).
   - `"length"`: Chiều dài tổng thể / sải cánh (máy bay, trực thăng, tàu thủy, xe kéo container, toa xe lửa, cần cẩu...).
   - `"extent"`: Bề rộng khuôn viên / vùng hoạt động (cảng biển, bãi đỗ xe, công trường, trạm dừng nghỉ, vùng đổ sụp...).

2. **Tính nhất quán vật lý 100%:**
   - Cả hai giá trị $L_c^{\min}$ và $L_c^{\text{typ}}$ bắt buộc phải đo **cùng một đại lượng vật lý** tương ứng với `critical_dimension_type`. Tuyệt đối không để một số đo chiều rộng và một số đo chiều dài trong cùng một trường.

---

## C. DANH MỤC QUY CHUẨN KỸ THUẬT QUỐC TẾ CHO 84 LỚP CANONICAL

### 1. Nhóm Hàng không & Hạ tầng Hàng không (Aviation & Infrastructure)
* **Cơ sở dữ liệu đặc tính máy bay FAA (FAA Aircraft Characteristics Database - October 2024 Edition):**
  * Lưu cục bộ tại: `rs-solve/reference/aircraft_data.xlsx` (Sheet: `ACD_Data`).
  * Cột Col 15 (`Wingspan_ft_without_winglets_sharklets`), Col 17 (`Length_ft`), Col 27 (`Class`), Col 37 (`Registration_Count`), Col 38 (`TMFS_Operations_FY24`).
  * `small_aircraft`:
    * Chiều dài nhỏ nhất (Piper Cherokee PA-28, Col 17): $23.8\,\text{ft} \times 0.3048 = 7.254\,\text{m} \approx \mathbf{7.25\,\text{m}}$. (10,740 đăng ký FAA).
    * Sải cánh điển hình (Cessna 172 Skyhawk, Col 15): $36.1\,\text{ft} \times 0.3048 = 11.003\,\text{m} = \mathbf{11.00\,\text{m}}$. (>20,000 đăng ký FAA - máy bay phổ biến nhất lịch sử).
  * `airplane`:
    * Cỡ nhỏ huấn luyện: $7.25\,\text{m}$.
    * Phản lực thương mại thân hẹp phổ biến nhất thế giới (Airbus A320-200, Col 15): $111.9\,\text{ft} = \mathbf{34.11\,\text{m}}$ (1.4 triệu chuyến bay FY24) / Boeing 737-800: $112.6\,\text{ft} = \mathbf{34.32\,\text{m}}$ (2.48 triệu chuyến bay FY24).
  * `helicopter`:
    * Trực thăng nhỏ (Robinson R22): Chiều dài tổng thể khi quay cánh quạt $D = 28.7\,\text{ft} = \mathbf{8.75\,\text{m}}$.
    * Trực thăng tiện ích phổ biến (Bell 206B JetRanger): $D = 39.2\,\text{ft} = \mathbf{11.95\,\text{m}}$.
    * Tiêu chuẩn: FAA AC 150/5390-2D (Heliport Design, Appendix A) / ICAO Annex 14 Vol II.
  * `airport`:
    * Bề rộng mặt đường băng tiêu chuẩn ICAO Annex 14 Vol I Section 3.1.10: Code 1A tối thiểu $18.00\,\text{m}$; Code 4C/D/E phản lực thương mại $45.00\,\text{m}$.
  * `helipad`:
    * Bề rộng bãi tiếp đất TLOF theo FAA AC 150/5390-2D: Bãi trực thăng nhỏ tối thiểu $8.00\,\text{m}$; bãi thương mại/bệnh viện $15.00\,\text{m}$ (50 ft).
  * `aircraft_hangar`:
    * Bề rộng cửa mở theo FAA AC 150/5300-13B: Hangar cá nhân T-hangar $12.50\,\text{m}$; hangar bảo dưỡng máy bay thân hẹp $45.00\,\text{m}$.

### 2. Nhóm Phương tiện Đường bộ (Road Vehicles)
* **Quy chuẩn Châu Âu & UNECE (Regulation (EU) 2018/858 & Directive 96/53/EC):**
  * `small_vehicle` (Category M1): Chiều rộng thân xe không gương: xe nhỏ đô thị $1.50\,\text{m}$ (Kei car $1.48\,\text{m}$, Fiat 500 $1.62\,\text{m}$); sedan tiêu chuẩn $1.80\,\text{m}$ (VW Golf $1.79\,\text{m}$, Toyota Camry $1.84\,\text{m}$).
  * `large_vehicle` (Category M3/N3): Chiều rộng tối đa theo luật: $2.20\,\text{m} - 2.55\,\text{m}$.
  * `bus` (Category M3 / UNECE R107): Chiều rộng thân xe buýt: $2.30\,\text{m}$ (midi bus) đến $2.55\,\text{m}$ (xe buýt đô thị và xe khách).
  * `truck` (Category N2/N3): Chiều rộng thùng xe tải: $2.20\,\text{m} - 2.50\,\text{m}$.
  * `van` (Category N1): Chiều rộng thân xe van: $1.70\,\text{m}$ (van nhỏ Caddy/Kangoo) đến $2.00\,\text{m}$ (van lớn Transit/Sprinter).
  * `trailer` (Category O2-O4 / ISO 1726): Chiều rộng thùng rơ-moóc: $2.00\,\text{m} - 2.55\,\text{m}$.
  * `pickup_truck` (US DOT FMVSS 108 / FHWA): Chiều rộng thân xe bán tải không gương: $1.80\,\text{m}$ (cỡ trung Hilux/Ranger) đến $2.03\,\text{m}$ ($79.9\,\text{in}$, cỡ lớn Ford F-150).
  * `utility_truck`: Bề rộng thân xe công vụ Class 4-5: $2.00\,\text{m} - 2.45\,\text{m}$.
  * `truck_tractor`: Chiều rộng cabin đầu kéo: $2.45\,\text{m} - 2.50\,\text{m}$.
  * `truck_tractor_w_box_trailer`: Chiều dài tổng thể đoàn xe sơ-mi rơ-moóc thùng kín: $12.00\,\text{m} - 16.50\,\text{m}$ (Directive 96/53/EC giới hạn tối đa 16.50m).
  * `truck_tractor_w_flatbed_trailer`: Chiều dài tổng thể đoàn xe sơ-mi rơ-moóc sàn phẳng: $12.00\,\text{m} - 16.50\,\text{m}$.
  * `truck_tractor_w_liquid_tank`: Chiều dài tổng thể xe téc chở xăng dầu: $12.00\,\text{m} - 16.50\,\text{m}$.
  * `crane_truck`: Chiều dài xe cẩu tự hành di chuyển trên đường: $9.00\,\text{m} - 12.00\,\text{m}$.

### 3. Nhóm Vi mô & Người đi bộ (Micro-mobility & Pedestrians)
* **Tiêu chuẩn Nhân trắc học & Phương tiện Nhẹ:**
  * `pedestrian` (ISO 7250-1:2017 Item 4.2.2): Chiều rộng hai vai (biacromial breadth): $0.40\,\text{m}$ (người nhỏ) đến $0.50\,\text{m}$ (người lớn mặc trang phục).
  * `people` (VisDrone Benchmark TPAMI 2021): Đường kính cụm đám đông: $1.00\,\text{m} - 2.50\,\text{m}$.
  * `bicycle` (ISO 4210-2:2015 Section 4.7): Chiều rộng tay lái ghi-đông: $0.45\,\text{m}$ (ghi-đông cong đua) đến $0.65\,\text{m}$ (tay lái ngang thành phố/địa hình).
  * `motorcycle` (UNECE R78 / EU 168/2013 Category L3e): Chiều rộng tay lái mô tô: $0.70\,\text{m} - 0.85\,\text{m}$.
  * `tricycle` (GB 7258-2017 Section 4.5): Bề rộng trục bánh sau/thùng xe ba bánh: $0.95\,\text{m} - 1.20\,\text{m}$.
  * `awning_tricycle` (GB 7258-2017): Bề rộng khung mái che xe ba bánh: $1.05\,\text{m} - 1.25\,\text{m}$.

### 4. Nhóm Hàng hải & Tàu thuyền (Maritime Vessels)
* **Quy chuẩn Civil NIIRS Reference Guide (IRSC/FAS 1996) & IMO:**
  * `ship`: Chiều dài tổng thể (LOA): $12.00\,\text{m} - 80.00\,\text{m}$.
  * `motorboat`: Chiều dài thân ca-nô thể thao: $4.57\,\text{m} - 7.62\,\text{m}$ ($15 - 25\,\text{ft}$ theo tiêu chí Civil NIIRS).
  * `sailboat`: Chiều dài thân thuyền buồm một thân: $6.00\,\text{m} - 12.00\,\text{m}$ ($20 - 40\,\text{ft}$).
  * `tugboat`: Chiều dài tàu dắt cảng: $18.00\,\text{m} - 28.00\,\text{m}$.
  * `barge`: Chiều dài sà lan chở hàng: $25.00\,\text{m}$ (sà lan mặt boong) đến $59.44\,\text{m}$ (sà lan Jumbo Hopper 195ft chuẩn USACE).
  * `fishing_vessel`: Chiều dài tàu đánh cá thương mại: $10.00\,\text{m} - 24.00\,\text{m}$ (ngưỡng công ước Cape Town/Torremolinos).
  * `ferry`: Chiều dài phà chở khách và ô tô: $30.00\,\text{m} - 85.00\,\text{m}$.
  * `yacht`: Chiều dài du thuyền cá nhân: $12.00\,\text{m} - 35.00\,\text{m}$ (ngưỡng siêu du thuyền).
  * `container_ship`: Chiều dài tàu chở container: $90.00\,\text{m}$ (feeder) đến $294.13\,\text{m}$ ($965\,\text{ft}$, chuẩn Panamax theo ACP).
  * `oil_tanker`: Chiều dài tàu chở dầu: $80.00\,\text{m}$ (tàu ven biển) đến $245.00\,\text{m}$ (Aframax crude tanker).
  * `harbor`: Chiều dài cầu bến cập tàu: $50.00\,\text{m} - 200.00\,\text{m}$ (World Bank Port Guidelines).

### 5. Nhóm Đường sắt (Railway Rolling Stock & Infrastructure)
* **Quy chuẩn Hiệp hội Đường sắt Quốc tế (UIC Standards):**
  * `railway_vehicle`: Chiều dài qua hai đầu móc nối (UIC 505/571): $14.00\,\text{m} - 20.00\,\text{m}$.
  * `locomotive`: Chiều dài đầu máy xe lửa chính tuyến: $18.00\,\text{m}$ (Siemens Vectron $18.98\,\text{m}$) đến $20.50\,\text{m}$ (GE Evolution $22.3\,\text{m}$).
  * `railway_passenger_car`: Chiều dài toa xe chở khách tiêu chuẩn UIC 567: $21.00\,\text{m}$ đến chuẩn UIC-Z cố định $\mathbf{26.40\,\text{m}}$.
  * `railway_cargo_car`: Chiều dài toa xe hàng có mui tiêu chuẩn UIC 571-1: $12.00\,\text{m} - 16.50\,\text{m}$.
  * `railway_flat_car`: Chiều dài toa xe chở container tiêu chuẩn UIC 571-2: $14.00\,\text{m} - 19.90\,\text{m}$ (cho container 40ft/60ft).
  * `railway_tank_car`: Đường kính ngoài bồn xi-téc đường sắt (UIC 571-3 / DOT-111): $2.40\,\text{m} - 2.80\,\text{m}$.
  * `train_station`: Chiều dài phân khu ke ga hành khách (TSI PRM / UIC Leaflet 700): $50.00\,\text{m} - 200.00\,\text{m}$.

### 6. Nhóm Cơ giới Nặng & Xây dựng (Machinery & Heavy Construction)
* **Tiêu chuẩn Máy làm đất ISO 6165 & Hãng sản xuất Caterpillar/Liebherr/Kalmar:**
  * `haul_truck`: Bề rộng ngoài lốp xe tải mỏ: $6.10\,\text{m}$ (CAT 777G) đến $9.76\,\text{m}$ ($32\,\text{ft}$, CAT 797F).
  * `bulldozer`: Bề rộng lưỡi gạt đất: $3.26\,\text{m}$ (lưỡi bán chữ U CAT D6) đến $4.31\,\text{m}$ (lưỡi SU CAT D8).
  * `ground_grader`: Bề rộng lưỡi san gạt mặt đường: $3.66\,\text{m}$ ($12\,\text{ft}$) đến $4.27\,\text{m}$ ($14\,\text{ft}$, CAT 140).
  * `excavator`: Bề rộng vệt xích máy xúc thủy lực: $2.80\,\text{m}$ (CAT 320: $2.98\,\text{m}$) đến $3.50\,\text{m}$ (CAT 349: $3.64\,\text{m}$).
  * `dump_truck`: Chiều dài xe ben 3-4 trục: $7.50\,\text{m} - 9.00\,\text{m}$.
  * `cement_mixer`: Chiều dài xe bồn trộn bê tông 3-4 trục: $8.00\,\text{m} - 9.50\,\text{m}$.
  * `scraper_tractor`: Bề rộng phủ bì máy cào đất: $3.56\,\text{m}$ (CAT 621K) đến $3.94\,\text{m}$ (CAT 631K).
  * `reach_stacker`: Bề rộng cầu xe đến ngàm gắp container 20ft: $4.15\,\text{m} - 6.06\,\text{m}$ (Kalmar DRG450).
  * `straddle_carrier`: Bề rộng cổng xe bốc container tự hành: $4.80\,\text{m} - 5.00\,\text{m}$ (Kalmar).
  * `tower_crane`: Chiều dài tay cần làm việc nằm ngang: $30.00\,\text{m} - 55.00\,\text{m}$ (Liebherr EC-B / ISO 4301-3).
  * `container_crane`: Tầm với cần vươn giàn cẩu bờ STS qua mạn tàu: $30.48\,\text{m} - 50.00\,\text{m}$ (ISO 4301-5).
  * `mobile_crane`: Chiều dài xe cẩu tự hành di chuyển trên đường: $10.50\,\text{m} - 14.50\,\text{m}$ (Liebherr LTM).
  * `engineering_vehicle`: Bề rộng khung vỏ cơ giới công trình: $2.40\,\text{m} - 3.00\,\text{m}$.
  * `construction_site`: Bề rộng mặt bằng thi công đang hoạt động: $25.00\,\text{m} - 100.00\,\text{m}$ (OSHA 1926).

### 7. Nhóm Sân bãi Thể thao (Sports Facilities)
* **Luật thi đấu chính thức của các Liên đoàn Thể thao Quốc tế:**
  * `tennis_court` (ITF Rules of Tennis Appendix I): Bề rộng mặt sân kẻ vạch: $8.23\,\text{m}$ ($27\,\text{ft}$, sân đơn) và $\mathbf{10.97\,\text{m}}$ ($36\,\text{ft}$, sân đôi).
  * `basketball_court` (FIBA Art. 2.1 / NBA Rule 1): Bề rộng mặt sân kẻ vạch: $\mathbf{15.00\,\text{m}}$ (chuẩn FIBA) và $15.24\,\text{m}$ ($50\,\text{ft}$, NBA/NCAA).
  * `baseball_diamond` (WBSC / OBR Rule 2.01): Cự ly giữa các base $27.43\,\text{m}$ ($90\,\text{ft}$); khoảng cách ra hàng rào ngoài sân $97.54\,\text{m}$ ($320\,\text{ft}$).
  * `soccer_ball_field` (IFAB Law 1): Bề rộng đường biên ngang: tối thiểu $45.00\,\text{m}$ (giải quốc nội) và $\mathbf{68.00\,\text{m}}$ (chuẩn quốc tế FIFA World Cup).
  * `ground_track_field` (World Athletics Manual 2019): Bề rộng lòng sân trong đường chạy oval 400m: $\mathbf{73.00\,\text{m}}$ ($2 \times R = 2 \times 36.50\,\text{m}$); bề rộng toàn bộ 8 làn chạy: $\mathbf{92.52\,\text{m}}$.
  * `swimming_pool` (World Aquatics Part VI): Bề rộng mặt nước: $12.50\,\text{m}$ (bể ngắn 5-6 làn) đến $\mathbf{25.00\,\text{m}}$ (bể chuẩn Olympic 10 làn).
  * `stadium` (FIFA Stadium Guidelines / DIOR): Đường kính ngoài khán đài và mái che: $120.00\,\text{m} - 200.00\,\text{m}$.
  * `golf_course` (USGA / DIOR): Phạm vi khu vực sân tập / green gạt bóng: $50.00\,\text{m} - 150.00\,\text{m}$.

### 8. Nhóm Hạ tầng Giao thông & Đô thị (Transportation & Urban Infrastructure)
* **Quy chuẩn Thiết kế Đường bộ AASHTO & FHWA:**
  * `roundabout` (FHWA-RD-00-067 Exhibit 6-1): Đường kính ngoài vòng xuyến (Inscribed Circle Diameter): $15.00\,\text{m}$ (mini-roundabout) đến $35.00\,\text{m}$ (chuẩn 1 làn).
  * `bridge` (AASHTO LRFD Section 2.5): Bề rộng bản mặt cầu: $8.00\,\text{m}$ (2 làn đường phụ) đến $24.00\,\text{m}$ (4 làn cao tốc).
  * `overpass` (AASHTO Green Book): Bề rộng bản mặt cầu vượt: $8.00\,\text{m}$ (1 làn nhánh) đến $18.00\,\text{m}$ (4 làn đô thị).
  * `expressway_toll_station`: Bề rộng mái che trạm thu phí cắt ngang các làn: $20.00\,\text{m} - 60.00\,\text{m}$.
  * `expressway_service_area`: Bề rộng khuôn viên trạm dừng nghỉ: $80.00\,\text{m} - 200.00\,\text{m}$.
  * `vehicle_lot` (ULI Parking Standards): Bề rộng mô-đun bãi đỗ xe: $20.00\,\text{m}$ (2 dãy đỗ + lối đi) đến $80.00\,\text{m}$ (bãi thương mại).

### 9. Nhóm Công nghiệp, Năng lượng & Logistics (Industrial & Energy)
* **Quy chuẩn Quốc tế API, ISO, IEC, IEEE, ICOLD:**
  * `storage_tank` (API Standard 650 Section 5 / Table A.1a): Đường kính ngoài bồn dầu thép hàn: $10.00\,\text{m} - 30.00\,\text{m}$.
  * `shipping_container` (ISO 668:2020 Series 1 Table 1): Chiều dài danh định container: $\mathbf{6.06\,\text{m}}$ (20ft: $6.058\,\text{m}$) và $\mathbf{12.19\,\text{m}}$ (40ft: $12.192\,\text{m}$).
  * `shipping_container_lot`: Bề rộng cụm dãy container: $15.00\,\text{m} - 60.00\,\text{m}$ (UNCTAD Port Development).
  * `chimney` (CICIND Model Code Part A): Đường kính ngoài chân ống khói bê tông: $3.00\,\text{m} - 12.00\,\text{m}$.
  * `dam` (USACE EM 1110-2-2200 Section 3-3 / ICOLD): Bề rộng đỉnh đập chắn nước: $6.00\,\text{m} - 10.00\,\text{m}$.
  * `windmill` (IEC 61400-1): Đường kính quét của 3 cánh tuabin gió: $40.00\,\text{m} - 100.00\,\text{m}$.
  * `pylon` (IEEE Standard 691 / ASCE 10-15): Bề rộng sải xà ngang đỡ dây điện: $6.00\,\text{m}$ (110kV) đến $18.00\,\text{m}$ (500kV).
  * `tower` (ANSI/TIA-222-H): Bề rộng chân đế tháp viễn thông tự đứng: $4.00\,\text{m} - 10.00\,\text{m}$.

### 10. Nhóm Công trình Dân dụng & Xây dựng (Civil Buildings & Sites)
* **Quy chuẩn Quốc tế IBC, IRC, UNHCR, FEMA:**
  * `building` (IBC 2021 / IRC): Bề rộng nhỏ nhất cạnh đáy tòa nhà: $6.00\,\text{m}$ (nhà ở riêng lẻ) đến $25.00\,\text{m}$ (chung cư/văn phòng).
  * `shed` (IRC 2021 Section R105.2): Bề rộng nhà kho tạm: $2.50\,\text{m} - 6.00\,\text{m}$.
  * `hut_tent` (UNHCR Emergency Shelter Guidelines): Bề rộng lều cứu trợ: $2.50\,\text{m}$ đến $4.00\,\text{m}$ (lều gia đình chuẩn UNHCR $4.00\,\text{m} \times 4.00\,\text{m}$).
  * `damaged_building` (FEMA US&R Field Operations Guide): Phạm vi đống đổ nát công trình: $6.00\,\text{m} - 25.00\,\text{m}$.
  * `facility` (APA Industrial Facility Planning): Bề rộng khuôn viên cơ sở công nghiệp: $30.00\,\text{m} - 100.00\,\text{m}$.
