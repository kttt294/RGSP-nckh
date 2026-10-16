# Báo Cáo Đánh Giá Khoa Học RS-Solve / RGSP: QWEN3

## 1. Tổng Quan Hiệu Năng Thực Nghiệm
- **Tổng số mẫu đánh giá:** 30,771 câu
- **Độ chính xác toàn cục (Overall Accuracy):** 41.10%
- **Tỷ lệ từ chối toàn cục (Abstention Rate):** 23.66%

### Phân bố độ chính xác theo loại câu hỏi (Question Kind):
| Loại câu hỏi | Mô tả nhiệm vụ | Số mẫu | Độ chính xác (Acc %) | Tỷ lệ từ chối (%) |
| :--- | :--- | :---: | :---: | :---: |
| **Q1** | Tồn tại vật thể (Detection) | 6,000 | 54.47% | 2.42% |
| **Q2** | Đếm số lượng (Counting) | 6,000 | 63.55% | 0.73% |
| **Q3-HF** | Chi tiết kết cấu (High-freq) | 332 | 41.87% | 9.64% |
| **Q3-LF** | Màu sắc vật thể (Low-freq) | 439 | 51.48% | 14.58% |
| **Q4** | Hướng xoay OBB (Orientation) | 6,000 | 35.08% | 29.28% |
| **Q5** | Nhận dạng loại con (Identification) | 6,000 | 18.43% | 54.12% |
| **Q6** | Định vị không gian (Grounding) | 6,000 | 33.17% | 33.17% |

---
## 2. BẢNG VI: Ma Trận Kết Cục $\rho_{px} \times \rho_{tok}$ (%)
> *Mỗi ô: % Hallucination / % Lỗi nhận thức / % Từ chối*

| $\rho_{px} \backslash \rho_{tok}$ | < 0.5 | 0.5 - 1.0 | 1.0 - 2.0 | 2.0 - 4.0 | $\ge$ 4.0 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **< p0_U** | 63.4% / 0.0% / 36.6% | 88.1% / 0.0% / 11.9% | — / — / — | — / — / — | — / — / — |
| **p0_U - 2p0_U** | 0.0% / 17.6% / 17.2% | 0.0% / 58.6% / 9.3% | 0.0% / 54.3% / 6.0% | — / — / — | — / — / — |
| **2p0_U - 4p0_U** | 0.0% / 34.7% / 46.8% | 0.0% / 43.2% / 19.8% | 0.0% / 54.9% / 5.2% | 0.0% / 52.7% / 2.5% | — / — / — |
| **>= 4p0_U** | 0.0% / 41.8% / 39.6% | 0.0% / 41.2% / 22.5% | 0.0% / 40.9% / 10.1% | 0.0% / 28.8% / 1.9% | 0.0% / 12.1% / 0.3% |

---
## 3. BẢNG VII: Kiểm Định Giả Thuyết Nút Thắt Token $H_1$ (Logistic Regression)
$$\text{logit}(P(\text{error})) = \beta_0 + \beta_1 \log_2(\rho_{tok})$$

- **Hệ số $\beta_1$ (log $\rho_{tok}$):** `-0.3840`
- **Khoảng tin cậy 95% (CI 95%):** `[-0.4041, -0.3638]`
- **Tỷ số chênh (Odds Ratio):** `0.6811`
- **Giá trị p (p-value):** `0.0000e+00`
- **Kết luận giả thuyết $H_1$:** ✅ **ĐƯỢC XÁC NHẬN** (Hệ số âm có ý nghĩa thống kê $p < 0.05$)

---
## 4. BẢNG VIII: Hai Ngưỡng Phân Giải Độc Lập $H_2$

| Thông số | Ký hiệu | Giá trị thực nghiệm | Đơn vị | Ghi chú |
| :--- | :---: | :---: | :---: | :--- |
| Kích thước patch hiệu dụng | $P$ | `32` | px | Cấu trúc bộ mã hoá thị giác |
| Ngưỡng cảm biến 50% | $p_0^{50}$ | `39.88` | px gốc | Đo tại vùng unconstrained tokens |
| Giới hạn vật lý cảm biến | $p_0^U$ | `13.57` | px gốc | Ngưỡng dưới không thể giải |
| Ngưỡng token mô hình | $\rho_{tok}^*$ | `1.30` | tokens | Đo tại vùng unconstrained pixels |
| Ngưỡng pixel quy đổi | $\rho_{tok}^* \cdot P$ | `41.55` | px mô hình | Chuẩn hoá theo kiến trúc |

---
## 5. BẢNG IX: Bậc Thang Johnson $H_3$ & Kiểm Định Jonckheere-Terpstra

| Cấp bậc nhiệm vụ | Nhiệm vụ | Số mẫu | Ngưỡng cảm biến $p_0^{{50}}$ (px) |
| :--- | :--- | :---: | :---: |
| **Q3-LF** | Q3-LF Màu sắc (Low-freq) | 420 | `61.94` |
| **Q1** | Q1 Tồn tại (Detection) | 5,986 | `44.24` |
| **Q4** | Q4 Hướng OBB (Orientation) | 2,662 | `1285.98` |
| **Q2** | Q2 Đếm / Lớp (Recognition) | 4,261 | `0.43` |
| **Q5** | Q5 Loại con (Identification) | 5,947 | `555.54` |

- **Thống kê kiểm định Jonckheere-Terpstra (J):** `7292762.0` (Kỳ vọng: `14114417.5`, z-score: `-50.03`)
- **p-value:** `1.0000e+00`
- **Kết luận bậc thang Johnson $H_3$:** ❌ **CHƯA ĐẠT ĐIỀU KIỆN**

---
## 6. BẢNG X: Dự Đoán Có Chọn Lọc (Selective Prediction - $H_4$)

| Phương pháp | Cần nội bộ | Gọi thêm | Cov@5% ↑ | Cov@10% ↑ | Cov@20% ↑ | Rủi ro thực @10% | AURC ↓ | Abst.-acc trên U ↑ | Từ chối thừa trên A ↓ | Selective Acc. ↑ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Zero-shot** | — | 0.0 | 0.0% | 0.0% | 0.0% | 52.7% | 0.527 | 30.3% | 19.8% | 41.1% |
| **Max-logit / Confidence** | ✓ | 0.0 | 0.0% | 11.7% | 24.7% | 10.0% | 0.336 | 30.3% | 88.3% | 41.1% |
| **RGSP-abstain (Physical)** | ✗ | 0.0 | 0.0% | 0.0% | 0.0% | 26.0% | 0.044 | 100.0% | 83.2% | 44.3% |
| **RGSP+ (Physical + Confidence)** | ✓ | 0.0 | 9.4% | 11.3% | 14.4% | 10.0% | 0.336 | 100.0% | 88.7% | 44.3% |

---
*Báo cáo được tính toán tự động bằng Engine Đánh giá Khoa học RGSP/RS-Solve.*