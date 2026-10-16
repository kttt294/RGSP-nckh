# Báo Cáo Đánh Giá Khoa Học RS-Solve / RGSP: GEOCHAT

## 1. Tổng Quan Hiệu Năng Thực Nghiệm
- **Tổng số mẫu đánh giá:** 30,771 câu
- **Độ chính xác toàn cục (Overall Accuracy):** 24.08%
- **Tỷ lệ từ chối toàn cục (Abstention Rate):** 0.00%

### Phân bố độ chính xác theo loại câu hỏi (Question Kind):
| Loại câu hỏi | Mô tả nhiệm vụ | Số mẫu | Độ chính xác (Acc %) | Tỷ lệ từ chối (%) |
| :--- | :--- | :---: | :---: | :---: |
| **Q1** | Tồn tại vật thể (Detection) | 6,000 | 0.00% | 0.00% |
| **Q2** | Đếm số lượng (Counting) | 6,000 | 71.02% | 0.00% |
| **Q3-HF** | Chi tiết kết cấu (High-freq) | 332 | 51.20% | 0.00% |
| **Q3-LF** | Màu sắc vật thể (Low-freq) | 439 | 15.95% | 0.00% |
| **Q4** | Hướng xoay OBB (Orientation) | 6,000 | 15.75% | 0.00% |
| **Q5** | Nhận dạng loại con (Identification) | 6,000 | 32.73% | 0.00% |
| **Q6** | Định vị không gian (Grounding) | 6,000 | 0.00% | 0.00% |

---
## 2. BẢNG VI: Ma Trận Kết Cục $\rho_{px} \times \rho_{tok}$ (%)
> *Mỗi ô: % Hallucination / % Lỗi nhận thức / % Từ chối*

| $\rho_{px} \backslash \rho_{tok}$ | < 0.5 | 0.5 - 1.0 | 1.0 - 2.0 | 2.0 - 4.0 | $\ge$ 4.0 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **< p0_U** | 100.0% / 0.0% / 0.0% | 100.0% / 0.0% / 0.0% | 100.0% / 0.0% / 0.0% | — / — / — | — / — / — |
| **p0_U - 2p0_U** | 0.0% / 32.6% / 0.0% | 0.0% / 13.4% / 0.0% | 0.0% / 63.3% / 0.0% | 0.0% / 75.5% / 0.0% | — / — / — |
| **2p0_U - 4p0_U** | 0.0% / 74.3% / 0.0% | 0.0% / 72.3% / 0.0% | 0.0% / 65.7% / 0.0% | 0.0% / 61.3% / 0.0% | 0.0% / 58.3% / 0.0% |
| **>= 4p0_U** | 0.0% / 75.9% / 0.0% | 0.0% / 74.8% / 0.0% | 0.0% / 76.5% / 0.0% | 0.0% / 74.7% / 0.0% | 0.0% / 88.3% / 0.0% |

---
## 3. BẢNG VII: Kiểm Định Giả Thuyết Nút Thắt Token $H_1$ (Logistic Regression)
$$\text{logit}(P(\text{error})) = \beta_0 + \beta_1 \log_2(\rho_{tok})$$

- **Hệ số $\beta_1$ (log $\rho_{tok}$):** `0.2214`
- **Khoảng tin cậy 95% (CI 95%):** `[0.2023, 0.2405]`
- **Tỷ số chênh (Odds Ratio):** `1.2478`
- **Giá trị p (p-value):** `0.0000e+00`
- **Kết luận giả thuyết $H_1$:** ❌ **CHƯA ĐẠT ĐIỀU KIỆN**

---
## 4. BẢNG VIII: Hai Ngưỡng Phân Giải Độc Lập $H_2$

| Thông số | Ký hiệu | Giá trị thực nghiệm | Đơn vị | Ghi chú |
| :--- | :---: | :---: | :---: | :--- |
| Kích thước patch hiệu dụng | $P$ | `14` | px | Cấu trúc bộ mã hoá thị giác |
| Ngưỡng cảm biến 50% | $p_0^{50}$ | `nan` | px gốc | Đo tại vùng unconstrained tokens |
| Giới hạn vật lý cảm biến | $p_0^U$ | `13.57` | px gốc | Ngưỡng dưới không thể giải |
| Ngưỡng token mô hình | $\rho_{tok}^*$ | `nan` | tokens | Đo tại vùng unconstrained pixels |
| Ngưỡng pixel quy đổi | $\rho_{tok}^* \cdot P$ | `nan` | px mô hình | Chuẩn hoá theo kiến trúc |

---
## 5. BẢNG IX: Bậc Thang Johnson $H_3$ & Kiểm Định Jonckheere-Terpstra

| Cấp bậc nhiệm vụ | Nhiệm vụ | Số mẫu | Ngưỡng cảm biến $p_0^{{50}}$ (px) |
| :--- | :--- | :---: | :---: |
| **Q3-LF** | Q3-LF Màu sắc (Low-freq) | 420 | `—` |
| **Q1** | Q1 Tồn tại (Detection) | 5,986 | `—` |
| **Q4** | Q4 Hướng OBB (Orientation) | 2,662 | `—` |
| **Q2** | Q2 Đếm / Lớp (Recognition) | 4,261 | `—` |
| **Q5** | Q5 Loại con (Identification) | 5,947 | `—` |

- **Thống kê kiểm định Jonckheere-Terpstra (J):** `6738936.5` (Kỳ vọng: `7372979.5`, z-score: `-7.02`)
- **p-value:** `1.0000e+00`
- **Kết luận bậc thang Johnson $H_3$:** ❌ **CHƯA ĐẠT ĐIỀU KIỆN**

---
## 6. BẢNG X: Dự Đoán Có Chọn Lọc (Selective Prediction - $H_4$)

| Phương pháp | Cần nội bộ | Gọi thêm | Cov@5% ↑ | Cov@10% ↑ | Cov@20% ↑ | Rủi ro thực @10% | AURC ↓ | Abst.-acc trên U ↑ | Từ chối thừa trên A ↓ | Selective Acc. ↑ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Zero-shot** | — | 0.0 | 0.0% | 0.0% | 0.0% | 62.2% | 0.622 | 0.0% | 0.0% | 24.1% |
| **Max-logit / Confidence** | ✓ | 0.0 | 0.0% | 0.0% | 0.0% | 1.0% | 0.756 | 0.0% | 100.0% | 24.1% |
| **RGSP-abstain (Physical)** | ✗ | 0.0 | 0.0% | 0.0% | 0.0% | 79.4% | 0.151 | 100.0% | 81.0% | 38.9% |
| **RGSP+ (Physical + Confidence)** | ✓ | 0.0 | 0.0% | 0.0% | 0.0% | 0.0% | 0.693 | 100.0% | 100.0% | 38.9% |

---
*Báo cáo được tính toán tự động bằng Engine Đánh giá Khoa học RGSP/RS-Solve.*