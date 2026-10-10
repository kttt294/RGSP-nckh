# Báo Cáo Đánh Giá Khoa Học RS-Solve / RGSP: LLAVA_1_5

## 1. Tổng Quan Hiệu Năng Thực Nghiệm
- **Tổng số mẫu đánh giá:** 30,771 câu
- **Độ chính xác toàn cục (Overall Accuracy):** 35.29%
- **Tỷ lệ từ chối toàn cục (Abstention Rate):** 11.59%

### Phân bố độ chính xác theo loại câu hỏi (Question Kind):
| Loại câu hỏi | Mô tả nhiệm vụ | Số mẫu | Độ chính xác (Acc %) | Tỷ lệ từ chối (%) |
| :--- | :--- | :---: | :---: | :---: |
| **Q1** | Tồn tại vật thể (Detection) | 6,000 | 39.77% | 0.00% |
| **Q2** | Đếm số lượng (Counting) | 6,000 | 42.17% | 0.00% |
| **Q3-HF** | Chi tiết kết cấu (High-freq) | 332 | 42.77% | 0.00% |
| **Q3-LF** | Màu sắc vật thể (Low-freq) | 439 | 17.77% | 0.00% |
| **Q4** | Hướng xoay OBB (Orientation) | 6,000 | 33.83% | 28.47% |
| **Q5** | Nhận dạng loại con (Identification) | 6,000 | 32.47% | 1.87% |
| **Q6** | Định vị không gian (Grounding) | 6,000 | 29.08% | 29.08% |

---
## 2. BẢNG VI: Ma Trận Kết Cục $\rho_{px} \times \rho_{tok}$ (%)
> *Mỗi ô: % Hallucination / % Lỗi nhận thức / % Từ chối*

| $\rho_{px} \backslash \rho_{tok}$ | < 0.5 | 0.5 - 1.0 | 1.0 - 2.0 | 2.0 - 4.0 | $\ge$ 4.0 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **< p0_U** | 93.8% / 0.0% / 6.2% | 57.6% / 0.0% / 42.4% | 44.5% / 0.0% / 55.5% | — / — / — | — / — / — |
| **p0_U - 2p0_U** | 0.0% / 57.6% / 0.9% | 0.0% / 41.1% / 0.1% | 0.0% / 35.3% / 40.0% | 0.0% / 26.1% / 53.8% | — / — / — |
| **2p0_U - 4p0_U** | 0.0% / 58.3% / 1.7% | 0.0% / 73.6% / 2.0% | 0.0% / 47.5% / 0.0% | 0.0% / 35.7% / 32.3% | 0.0% / 32.0% / 36.9% |
| **>= 4p0_U** | 0.0% / 56.4% / 0.0% | 0.0% / 52.7% / 0.1% | 0.0% / 62.5% / 0.4% | 0.0% / 53.1% / 0.2% | 0.0% / 47.2% / 3.1% |

---
## 3. BẢNG VII: Kiểm Định Giả Thuyết Nút Thắt Token $H_1$ (Logistic Regression)
$$\text{logit}(P(\text{error})) = \beta_0 + \beta_1 \log_2(\rho_{tok})$$

- **Hệ số $\beta_1$ (log $\rho_{tok}$):** `-0.0623`
- **Khoảng tin cậy 95% (CI 95%):** `[-0.0804, -0.0442]`
- **Tỷ số chênh (Odds Ratio):** `0.9396`
- **Giá trị p (p-value):** `1.5968e-11`
- **Kết luận giả thuyết $H_1$:** ✅ **ĐƯỢC XÁC NHẬN** (Hệ số âm có ý nghĩa thống kê $p < 0.05$)

---
## 4. BẢNG VIII: Hai Ngưỡng Phân Giải Độc Lập $H_2$

| Thông số | Ký hiệu | Giá trị thực nghiệm | Đơn vị | Ghi chú |
| :--- | :---: | :---: | :---: | :--- |
| Kích thước patch hiệu dụng | $P$ | `14` | px | Cấu trúc bộ mã hoá thị giác |
| Ngưỡng cảm biến 50% | $p_0^{50}$ | `138.48` | px gốc | Đo tại vùng unconstrained tokens |
| Giới hạn vật lý cảm biến | $p_0^U$ | `13.57` | px gốc | Ngưỡng dưới không thể giải |
| Ngưỡng token mô hình | $\rho_{tok}^*$ | `13.10` | tokens | Đo tại vùng unconstrained pixels |
| Ngưỡng pixel quy đổi | $\rho_{tok}^* \cdot P$ | `183.37` | px mô hình | Chuẩn hoá theo kiến trúc |

---
## 5. BẢNG IX: Bậc Thang Johnson $H_3$ & Kiểm Định Jonckheere-Terpstra

| Cấp bậc nhiệm vụ | Nhiệm vụ | Số mẫu | Ngưỡng cảm biến $p_0^{{50}}$ (px) |
| :--- | :--- | :---: | :---: |
| **Q3-LF** | Q3-LF Màu sắc (Low-freq) | 420 | `—` |
| **Q1** | Q1 Tồn tại (Detection) | 5,986 | `102.99` |
| **Q4** | Q4 Hướng OBB (Orientation) | 2,662 | `1861.57` |
| **Q2** | Q2 Đếm / Lớp (Recognition) | 4,261 | `11.35` |
| **Q5** | Q5 Loại con (Identification) | 5,947 | `—` |

- **Thống kê kiểm định Jonckheere-Terpstra (J):** `6571993.0` (Kỳ vọng: `11441081.5`, z-score: `-43.20`)
- **p-value:** `1.0000e+00`
- **Kết luận bậc thang Johnson $H_3$:** ❌ **CHƯA ĐẠT ĐIỀU KIỆN**

---
## 6. BẢNG X: Dự Đoán Có Chọn Lọc (Selective Prediction - $H_4$)

| Phương pháp | Cần nội bộ | Gọi thêm | Cov@5% ↑ | Cov@10% ↑ | Cov@20% ↑ | Rủi ro thực @10% | AURC ↓ | Abst.-acc trên U ↑ | Từ chối thừa trên A ↓ | Selective Acc. ↑ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Zero-shot** | — | 0.0 | 0.0% | 0.0% | 0.0% | 58.9% | 0.589 | 25.0% | 3.9% | 35.3% |
| **Max-logit / Confidence** | ✓ | 0.0 | 0.0% | 0.0% | 0.0% | 0.0% | 0.595 | 25.0% | 100.0% | 35.3% |
| **RGSP-abstain (Physical)** | ✗ | 0.0 | 0.0% | 0.0% | 0.0% | 54.0% | 0.152 | 100.0% | 71.8% | 44.6% |
| **RGSP+ (Physical + Confidence)** | ✓ | 0.0 | 0.0% | 0.0% | 0.0% | 1.0% | 0.569 | 100.0% | 100.0% | 44.6% |

---
*Báo cáo được tính toán tự động bằng Engine Đánh giá Khoa học RGSP/RS-Solve.*