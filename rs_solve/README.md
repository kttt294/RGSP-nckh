# RS-Solve: bộ mẫu 35 câu hỏi

Mã này hiện thực **7 loại câu hỏi trong Bảng IV của bản thảo RGSP**. Tài liệu nghiên cứu gọi benchmark đầy đủ là *RS-Resolve*; thư mục này dùng tên *RS-Solve* theo yêu cầu cho bản mẫu.

## Chạy ngay trên CPU

```powershell
python rs_solve/build_rs_solve.py --mode demo
```

Kết quả ở `rs_solve/demo_output/`: `rs_solve_qa.jsonl` có đúng **35 dòng, mỗi loại Q1, Q2, Q3-HF, Q3-LF, Q4, Q5, Q6 có 5 dòng**; `images/` có 5 sơ đồ 512×512 để thử pipeline và `summary.json` ghi số lượng. Các ảnh là **sơ đồ minh hoạ**, không phải ảnh viễn thám. GSD 0,3 m/px và `p0_U_px=6` trong chế độ này chỉ là **giả định cho fixture**, không phải ước lượng từ dữ liệu hay kết quả nghiên cứu. Mọi dòng được gắn `sample_status=illustrative_only` và `p0_status=illustrative_unvalidated`; không dùng để đánh giá mô hình hoặc báo số liệu bài báo.

Mỗi câu có trường `rho_px`, `L_m`, GSD, tọa độ OBB hoặc bbox, câu hỏi/lựa chọn/đáp án, lớp, ô lưới, tình trạng kiểm duyệt. `rho_tok_measured=null` vì chưa đọc lưới token từ bộ tiền xử lý mô hình. Trường `rho_tok_proxy_resize_only` chỉ là phép tính minh hoạ `rho_px × min(1, max_side / max(H,W)) / P`; **không được dùng thay cho** `rho_tok` đo từ lưới token thật của Qwen/InternVL/LLaVA hay mô hình khác.

## Sinh từ DOTA-v2.0 thật

Chuẩn bị **ảnh và nhãn cùng phiên bản** theo cấu trúc:

```text
DOTA-v2/
  images/P0000.png
  labelTxt/P0000.txt
```

Script đọc GSD trong dòng `gsd:` của nhãn DOTA, OBB và cờ `difficult`; ảnh thiếu GSD hoặc thiếu file nhãn tương ứng sẽ bị bỏ qua. DOTA chính thức mô tả cấu trúc OBB và metadata tại [trang dataset](https://captain-whu.github.io/DOTA/dataset). Không gán GSD bằng giá trị trung bình toàn bộ DOTA.

Ba tệp JSON đầu vào phải do nhóm nghiên cứu lập và kiểm:

- `review.json`: khóa là image ID. Ví dụ `{"P0000":{"absent_classes":["baseball-diamond"],"colors":{"3":"đỏ"}}}`. `absent_classes` chỉ thêm sau khi kiểm ảnh và nhãn để Q2 không biến thiếu nhãn thành phủ định giả. `colors` là chỉ số object theo **thứ tự dòng nhãn hợp lệ, bắt đầu từ 0**, chỉ nhập sau khi xem ảnh mịn để tạo Q3-LF.
- `threshold.json`: `{"value_px": <số từ pilot>, "status": "pilot_human_and_models_calibrated"}`. Script từ chối chế độ DOTA khi chưa có ngưỡng `p₀^U` từ pilot người và mô hình; đặc biệt Q6 không có nhãn đáng tin trước bước này.
- `min_sizes.json`: ánh xạ lớp vắng mặt sang **kích thước vật lý nhỏ nhất có nguồn kiểm chứng**, đơn vị mét, ví dụ dạng `{"baseball-diamond": <L_min_m>}`. Đây là đầu vào bắt buộc của Q2; không lấy kích thước vật cản dương thay thế.

```powershell
python rs_solve/build_rs_solve.py --mode dota `
  --dota-root C:\duong_dan\DOTA-v2 `
  --review C:\duong_dan\review.json `
  --threshold C:\duong_dan\threshold.json `
  --min-sizes C:\duong_dan\min_sizes.json `
  --output C:\duong_dan\rs_solve_real_35
```

Mặc định chọn 5 câu/loại và báo lỗi nếu một loại thiếu ứng viên. DOTA không có nhãn màu chuẩn nên Q3-LF cần `review.json`. Q2 cần rà soát phủ định. Q5 cần ít nhất 3 instance cùng lớp trong một ô với thứ tự trái–phải tách biệt. Q6 cần instance có `0.5 p₀^U ≤ ρ_px < p₀^U`. Dữ liệu thật sinh ra vẫn có `sample_status=candidate_needs_visual_QC`; người nghiên cứu cần rà từng ảnh–câu hỏi trước khi đưa vào benchmark. Script chỉ xuất đường dẫn ảnh, không sao chép/phân phối lại ảnh DOTA.

Để thêm **proxy** cho một chính sách resize đơn giản, truyền `--token-config views.json` với mảng `[ {"name":"tên_view", "max_side_px":512, "effective_patch_px":32} ]`. Chính sách tiling/anyres và `rho_tok` cuối cùng phải được lấy từ lưới token thật và ánh xạ tọa độ theo Mục III-A / Phụ lục B của bản thảo.

## Quy tắc gắn nhãn

- `ρ_px = min(hai cạnh OBB)` và `L = ρ_px × GSD`. Q2 dùng `L_c^min / GSD` cho lớp được hỏi nhưng vắng mặt. Q4 dùng trung vị `ρ_px` của các instance được đếm.
- Q1: có vật thể; `q1_group=iso_candidate` khi có tối đa 3 instance cùng lớp. Cờ này chưa thay thế phép kiểm khoảng cách **4 token thật**.
- Q2: phủ định khó từ các cặp hình thái trong Bảng IV, chỉ khi vắng mặt đã kiểm và có `L_c^min`.
- Q3-HF: hướng **trục dài 0–180°**, không suy heading; bỏ OBB gần vuông và gần ranh giới 45°. Q3-LF: màu trong 6 nhóm, chỉ khi đã kiểm bằng ảnh mịn.
- Q4: đếm trong ô 3×3, bỏ box có dưới 50% diện tích nằm trong ô; nhãn “không xác định” khi trung vị `ρ_px < p₀^U`.
- Q5: chọn bbox thứ hai từ trái trong cùng ô; các bbox ứng viên nằm trong lựa chọn. Q6: câu hướng trục về instance sát dưới ngưỡng cảm biến, đáp án từ chối.

Đây là bộ sinh **mẫu**. Nó chưa thay thế các bước của benchmark đầy đủ: kiểm thiếu nhãn quy mô lớn, đo lưới token từng mô hình, đối chứng mật độ, hiệu chỉnh người, kiểm màu/trục, chia train/calibration/test theo ảnh và thẩm định đáp án.
