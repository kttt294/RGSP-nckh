# Báo Cáo Đánh Giá Khoa Học RS-Solve / RGSP: LLAVA_NEXT

## 1. Tổng Quan Hiệu Năng Thực Nghiệm
- **Tổng số mẫu đánh giá:** 30,771 câu
- **Độ chính xác toàn cục (Overall Accuracy):** 46.39%
- **Tỷ lệ từ chối toàn cục (Abstention Rate):** 57.95%

### Phân bố độ chính xác theo loại câu hỏi (Question Kind):
| Loại câu hỏi | Mô tả nhiệm vụ | Số mẫu | Độ chính xác (Acc %) | Tỷ lệ từ chối (%) |
| :--- | :--- | :---: | :---: | :---: |
| **Q1** | Tồn tại vật thể (Detection) | 6,000 | 56.10% | 11.45% |
| **Q2** | Đếm số lượng (Counting) | 6,000 | 37.27% | 13.80% |
| **Q3-HF** | Chi tiết kết cấu (High-freq) | 332 | 50.90% | 3.61% |
| **Q3-LF** | Màu sắc vật thể (Low-freq) | 439 | 32.12% | 10.48% |
| **Q4** | Hướng xoay OBB (Orientation) | 6,000 | 51.22% | 86.28% |
| **Q5** | Nhận dạng loại con (Identification) | 6,000 | 1.52% | 98.02% |
| **Q6** | Định vị không gian (Grounding) | 6,000 | 86.67% | 86.67% |

---
## 2. BẢNG VI: Ma Trận Kết Cục $\rho_{px} \times \rho_{tok}$ (%)
> *Mỗi ô: % Hallucination / % Lỗi nhận thức / % Từ chối*

| $\rho_{px} \backslash \rho_{tok}$ | < 0.5 | 0.5 - 1.0 | 1.0 - 2.0 | 2.0 - 4.0 | $\ge$ 4.0 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **< p0_U** | 22.8% / 0.0% / 77.2% | 19.5% / 0.0% / 80.5% | — / — / — | — / — / — | — / — / — |
| **p0_U - 2p0_U** | 0.0% / 42.2% / 29.8% | 0.0% / 11.8% / 79.6% | 0.0% / 9.4% / 79.5% | — / — / — | — / — / — |
| **2p0_U - 4p0_U** | 0.0% / 22.2% / 66.5% | 0.0% / 10.2% / 70.0% | 0.0% / 11.7% / 78.4% | 0.0% / 8.8% / 78.8% | — / — / — |
| **>= 4p0_U** | 0.0% / 20.7% / 61.4% | 0.0% / 19.9% / 51.4% | 0.0% / 20.2% / 40.9% | 0.0% / 18.1% / 21.7% | 0.0% / 9.9% / 2.9% |

---
## 3. BẢNG VII: Kiểm Định Giả Thuyết Nút Thắt Token $H_1$ (Logistic Regression)
$$\text{logit}(P(\text{error})) = \beta_0 + \beta_1 \log_2(\rho_{tok})$$

- **Hệ số $\beta_1$ (log $\rho_{tok}$):** `-0.5112`
- **Khoảng tin cậy 95% (CI 95%):** `[-0.5332, -0.4892]`
- **Tỷ số chênh (Odds Ratio):** `0.5998`
- **Giá trị p (p-value):** `0.0000e+00`
- **Kết luận giả thuyết $H_1$:** ✅ **ĐƯỢC XÁC NHẬN** (Hệ số âm có ý nghĩa thống kê $p < 0.05$)

---
## 4. BẢNG VIII: Hai Ngưỡng Phân Giải Độc Lập $H_2$

| Thông số | Ký hiệu | Giá trị thực nghiệm | Đơn vị | Ghi chú |
| :--- | :---: | :---: | :---: | :--- |
| Kích thước patch hiệu dụng | $P$ | `14` | px | Cấu trúc bộ mã hoá thị giác |
| Ngưỡng cảm biến 50% | $p_0^{50}$ | `92.03` | px gốc | Đo tại vùng unconstrained tokens |
| Giới hạn vật lý cảm biến | $p_0^U$ | `13.57` | px gốc | Ngưỡng dưới không thể giải |
| Ngưỡng token mô hình | $\rho_{tok}^*$ | `1.97` | tokens | Đo tại vùng unconstrained pixels |
| Ngưỡng pixel quy đổi | $\rho_{tok}^* \cdot P$ | `27.63` | px mô hình | Chuẩn hoá theo kiến trúc |

---
## 5. BẢNG IX: Bậc Thang Johnson $H_3$ & Kiểm Định Jonckheere-Terpstra

| Cấp bậc nhiệm vụ | Nhiệm vụ | Số mẫu | Ngưỡng cảm biến $p_0^{{50}}$ (px) |
| :--- | :--- | :---: | :---: |
| **Q3-LF** | Q3-LF Màu sắc (Low-freq) | 420 | `260.25` |
| **Q1** | Q1 Tồn tại (Detection) | 5,986 | `41.11` |
| **Q4** | Q4 Hướng OBB (Orientation) | 2,662 | `—` |
| **Q2** | Q2 Đếm / Lớp (Recognition) | 4,261 | `56.01` |
| **Q5** | Q5 Loại con (Identification) | 5,947 | `1627.99` |

- **Thống kê kiểm định Jonckheere-Terpstra (J):** `1384959.0` (Kỳ vọng: `3782376.5`, z-score: `-42.07`)
- **p-value:** `1.0000e+00`
- **Kết luận bậc thang Johnson $H_3$:** ❌ **CHƯA ĐẠT ĐIỀU KIỆN**

---
## 6. BẢNG X: Dự Đoán Có Chọn Lọc (Selective Prediction - $H_4$)

| Phương pháp | Cần nội bộ | Gọi thêm | Cov@5% ↑ | Cov@10% ↑ | Cov@20% ↑ | Rủi ro thực @10% | AURC ↓ | Abst.-acc trên U ↑ | Từ chối thừa trên A ↓ | Selective Acc. ↑ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Zero-shot** | — | 0.0 | 0.0% | 0.0% | 0.0% | 71.4% | 0.714 | 77.6% | 46.7% | 46.4% |
| **Max-logit / Confidence** | ✓ | 0.0 | 0.0% | 0.0% | 9.7% | 0.0% | 0.487 | 77.6% | 100.0% | 46.4% |
| **RGSP-abstain (Physical)** | ✗ | 0.0 | 0.0% | 0.0% | 0.0% | 32.1% | 0.039 | 100.0% | 87.8% | 41.7% |
| **RGSP+ (Physical + Confidence)** | ✓ | 0.0 | 0.2% | 0.2% | 1.6% | 9.3% | 0.522 | 100.0% | 99.8% | 41.7% |

---
*Báo cáo được tính toán tự động bằng Engine Đánh giá Khoa học RGSP/RS-Solve.*