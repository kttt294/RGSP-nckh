# Báo Cáo Đánh Giá Khoa Học RS-Solve / RGSP: INTERNVL3_5

## 1. Tổng Quan Hiệu Năng Thực Nghiệm
- **Tổng số mẫu đánh giá:** 30,771 câu
- **Độ chính xác toàn cục (Overall Accuracy):** 40.20%
- **Tỷ lệ từ chối toàn cục (Abstention Rate):** 12.18%

### Phân bố độ chính xác theo loại câu hỏi (Question Kind):
| Loại câu hỏi | Mô tả nhiệm vụ | Số mẫu | Độ chính xác (Acc %) | Tỷ lệ từ chối (%) |
| :--- | :--- | :---: | :---: | :---: |
| **Q1** | Tồn tại vật thể (Detection) | 6,000 | 75.35% | 0.00% |
| **Q2** | Đếm số lượng (Counting) | 6,000 | 46.95% | 0.00% |
| **Q3-HF** | Chi tiết kết cấu (High-freq) | 332 | 38.55% | 0.60% |
| **Q3-LF** | Màu sắc vật thể (Low-freq) | 439 | 42.37% | 4.56% |
| **Q4** | Hướng xoay OBB (Orientation) | 6,000 | 30.98% | 16.43% |
| **Q5** | Nhận dạng loại con (Identification) | 6,000 | 28.40% | 26.38% |
| **Q6** | Định vị không gian (Grounding) | 6,000 | 19.27% | 19.27% |

---
## 2. BẢNG VI: Ma Trận Kết Cục $\rho_{px} \times \rho_{tok}$ (%)
> *Mỗi ô: % Hallucination / % Lỗi nhận thức / % Từ chối*

| $\rho_{px} \backslash \rho_{tok}$ | < 0.5 | 0.5 - 1.0 | 1.0 - 2.0 | 2.0 - 4.0 | $\ge$ 4.0 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **< p0_U** | 77.1% / 0.0% / 22.9% | 98.5% / 0.0% / 1.5% | — / — / — | — / — / — | — / — / — |
| **p0_U - 2p0_U** | 0.0% / 50.4% / 15.0% | 0.0% / 23.0% / 0.5% | 0.0% / 54.8% / 0.7% | — / — / — | — / — / — |
| **2p0_U - 4p0_U** | 0.0% / 42.7% / 25.3% | 0.0% / 45.0% / 6.0% | 0.0% / 50.5% / 1.1% | 0.0% / 45.5% / 0.4% | — / — / — |
| **>= 4p0_U** | 0.0% / 48.0% / 13.5% | 0.0% / 42.1% / 5.2% | 0.0% / 37.1% / 2.2% | 0.0% / 25.5% / 0.5% | 0.0% / 7.6% / 0.2% |

---
## 3. BẢNG VII: Kiểm Định Giả Thuyết Nút Thắt Token $H_1$ (Logistic Regression)
$$\text{logit}(P(\text{error})) = \beta_0 + \beta_1 \log_2(\rho_{tok})$$

- **Hệ số $\beta_1$ (log $\rho_{tok}$):** `-0.4639`
- **Khoảng tin cậy 95% (CI 95%):** `[-0.4851, -0.4428]`
- **Tỷ số chênh (Odds Ratio):** `0.6288`
- **Giá trị p (p-value):** `0.0000e+00`
- **Kết luận giả thuyết $H_1$:** ✅ **ĐƯỢC XÁC NHẬN** (Hệ số âm có ý nghĩa thống kê $p < 0.05$)

---
## 4. BẢNG VIII: Hai Ngưỡng Phân Giải Độc Lập $H_2$

| Thông số | Ký hiệu | Giá trị thực nghiệm | Đơn vị | Ghi chú |
| :--- | :---: | :---: | :---: | :--- |
| Kích thước patch hiệu dụng | $P$ | `28` | px | Cấu trúc bộ mã hoá thị giác |
| Ngưỡng cảm biến 50% | $p_0^{50}$ | `29.01` | px gốc | Đo tại vùng unconstrained tokens |
| Giới hạn vật lý cảm biến | $p_0^U$ | `13.57` | px gốc | Ngưỡng dưới không thể giải |
| Ngưỡng token mô hình | $\rho_{tok}^*$ | `0.65` | tokens | Đo tại vùng unconstrained pixels |
| Ngưỡng pixel quy đổi | $\rho_{tok}^* \cdot P$ | `18.17` | px mô hình | Chuẩn hoá theo kiến trúc |

---
## 5. BẢNG IX: Bậc Thang Johnson $H_3$ & Kiểm Định Jonckheere-Terpstra

| Cấp bậc nhiệm vụ | Nhiệm vụ | Số mẫu | Ngưỡng cảm biến $p_0^{{50}}$ (px) |
| :--- | :--- | :---: | :---: |
| **Q3-LF** | Q3-LF Màu sắc (Low-freq) | 420 | `113.67` |
| **Q1** | Q1 Tồn tại (Detection) | 5,986 | `12.64` |
| **Q4** | Q4 Hướng OBB (Orientation) | 2,662 | `388.35` |
| **Q2** | Q2 Đếm / Lớp (Recognition) | 4,261 | `5.18` |
| **Q5** | Q5 Loại con (Identification) | 5,947 | `421.02` |

- **Thống kê kiểm định Jonckheere-Terpstra (J):** `10613763.0` (Kỳ vọng: `18046313.0`, z-score: `-45.83`)
- **p-value:** `1.0000e+00`
- **Kết luận bậc thang Johnson $H_3$:** ❌ **CHƯA ĐẠT ĐIỀU KIỆN**

---
## 6. BẢNG X: Dự Đoán Có Chọn Lọc (Selective Prediction - $H_4$)

| Phương pháp | Cần nội bộ | Gọi thêm | Cov@5% ↑ | Cov@10% ↑ | Cov@20% ↑ | Rủi ro thực @10% | AURC ↓ | Abst.-acc trên U ↑ | Từ chối thừa trên A ↓ | Selective Acc. ↑ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Zero-shot** | — | 0.0 | 0.0% | 0.0% | 0.0% | 47.2% | 0.472 | 18.1% | 8.8% | 40.2% |
| **Max-logit / Confidence** | ✓ | 0.0 | 9.3% | 18.6% | 26.3% | 10.0% | 0.272 | 18.1% | 81.4% | 40.2% |
| **RGSP-abstain (Physical)** | ✗ | 0.0 | 0.0% | 0.0% | 0.0% | 20.3% | 0.033 | 100.0% | 83.8% | 44.6% |
| **RGSP+ (Physical + Confidence)** | ✓ | 0.0 | 0.5% | 1.2% | 9.0% | 9.9% | 0.322 | 100.0% | 98.8% | 44.6% |

---
*Báo cáo được tính toán tự động bằng Engine Đánh giá Khoa học RGSP/RS-Solve.*