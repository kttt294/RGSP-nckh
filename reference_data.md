
**CẢ 5 BỘ DỮ LIỆU NÀY ĐỀU CÓ SẴN TRÊN INTERNET VÀ HOÀN TOÀN MIỄN PHÍ 100% CHO MỤC ĐÍCH HỌC THUẬT / NGHIÊN CỨU**

### 1. DOTA-v2.0 (Dataset for Object Detection in Aerial Images)

* **Tình trạng:** **Miễn phí 100%** (Dành cho nghiên cứu học thuật).
* **Thủ tục:** Cần đăng ký một tài khoản miễn phí trên trang web của nhóm nghiên cứu (Đại học Vũ Hán - CAPTAIN Lab) để lấy link Google Drive hoặc Baidu Netdisk.
* **Link trang chủ & Tải về:**
  * Trang chính thức: [captain-whu.github.io/DOTA/dataset.html](https://captain-whu.github.io/DOTA/dataset.html)
  * Mã nguồn hỗ trợ (Devkit): [github.com/CAPTAIN-WHU/DOTA_devkit](https://github.com/CAPTAIN-WHU/DOTA_devkit)
* **Dung lượng:** Khá lớn (khoảng $30 - 40\text{ GB}$).

---

### 2. iSAID (Large-scale Dataset for Instance Segmentation in Aerial Images)

* **Tình trạng:** **Miễn phí 100%**.
* **Thủ tục:** Tải trực tiếp qua Google Drive do tác giả cung cấp trên trang dự án.
* **Link trang chủ & Tải về:**
  * Trang chính thức: [captain-whu.github.io/iSAID](https://captain-whu.github.io/iSAID/)
  * Mã nguồn hỗ trợ: [github.com/CAPTAIN-WHU/iSAID_DevKit](https://github.com/CAPTAIN-WHU/iSAID_DevKit)
* **Ghi chú:** Bộ này dùng chung ảnh gốc với DOTA nhưng chứa các file nhãn mặt nạ (Mask/Segmentation).

---

### 3. xView (DIUx xView Dataset)

* **Tình trạng:** **Miễn phí 100%**, nhưng **CẦN XIN PHÉP / ĐĂNG KÝ TÀI KHOẢN**.
* **Thủ tục:**
  * Đây là bộ dữ liệu do Cơ quan Đổi mới Quốc phòng Hoa Kỳ (DIU) và NGA phát hành từ vệ tinh WorldView-3.
  * Bạn phải vào trang web, tạo tài khoản và bấm chấp thuận **"Thỏa thuận sử dụng dữ liệu phi thương mại (Terms of Use Agreement)"**. Sau khi đồng ý, hệ thống sẽ cấp link tải trực tiếp hoặc lệnh tải qua AWS S3.
    *(Chính vì lý do này mà trong file Kế hoạch, nhóm tác giả có nhắc: "Cần ký thỏa thuận xView ở tuần 1" và "Không phân phối lại ảnh").*
* **Link trang chủ & Tải về:**
  * Trang đăng ký tải dữ liệu: [challenge.xviewdataset.org/data-download](https://challenge.xviewdataset.org/data-download)
  * Mã nguồn hỗ trợ: [github.com/DIUx-xView/xView_toolkit](https://github.com/DIUx-xView/xView_toolkit)
* **Dung lượng:** Khoảng $60\text{ GB}$.

---

### 4. DIOR (Object Detection in Optical Remote Sensing Images)

* **Tình trạng:** **Miễn phí 100%**.
* **Thủ tục:** Bộ dữ liệu do Đại học Công nghiệp Tây Bắc (NWPU) công bố. Link tải được chia sẻ công khai qua Google Drive, Baidu hoặc các mirror trên HuggingFace / Kaggle.
* **Link trang chủ & Tải về:**
  * Kho mã nguồn & tổng hợp link tải: [github.com/cancanfang/DIOR](https://github.com/cancanfang/DIOR)
  * Bản mirror tốc độ cao trên HuggingFace: Tìm kiếm `DIOR dataset` trên [huggingface.co/datasets](https://huggingface.co/datasets) hoặc tải qua Kaggle: [kaggle.com/datasets](https://www.kaggle.com/) (tìm `DIOR remote sensing`).
* **Dung lượng:** Khoảng $10 - 15\text{ GB}$ (khoảng 23.463 bức ảnh).

---

### 5. VisDrone (VisDrone-Dataset)

* **Tình trạng:** **Miễn phí 100%** (Tự do tải ngay lập tức).
* **Thủ tục:** Tải trực tiếp qua Google Drive hoặc Baidu Netdisk từ kho GitHub chính thức của Đại học Thiên Tân (AISKYEYE Team).
* **Link trang chủ & Tải về:**
  * Kho GitHub chính thức: [github.com/VisDrone/VisDrone-Dataset](https://github.com/VisDrone/VisDrone-Dataset)
  * Trang web cuộc thi: [aiskyeye.com](http://aiskyeye.com/)
* **Dung lượng:** Khoảng $10 - 25\text{ GB}$ tùy theo tập Task (ở đây ta chỉ cần tập *VisDrone-DET* cho bài toán Object Detection).

---

### 💡 LỜI KHUYÊN KHI TẢI DỮ LIỆU

1. **Tổng dung lượng**: Toàn bộ 5 bộ dữ liệu này cộng lại sẽ chiếm khoảng **$120\text{ GB} - 150\text{ GB}$ ổ cứng**. Hãy chuẩn bị sẵn một ổ cứng SSD có dung lượng trống tối thiểu $250\text{ GB}$ (để còn giải nén file `.zip / .tar`).
2. **Ưu tiên làm trước**: Hãy vào trang của **xView** đăng ký tài khoản ngay từ bây giờ, vì đôi khi hệ thống tự động của Mỹ mất vài tiếng đến một ngày để xác thực email trước khi mở link tải. Các bộ còn lại như VisDrone, iSAID, DIOR bạn có thể tải về máy bất cứ lúc nào qua Google Drive.
