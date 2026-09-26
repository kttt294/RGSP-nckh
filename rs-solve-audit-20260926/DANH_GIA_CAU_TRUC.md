# Đánh giá cấu trúc RS-Solve và kiểm tra thư mục

Ngày kiểm tra: 26/09/2026. Thư mục nguồn: `C:\Users\trang\Desktop\rgsp_nckh\rs-solve`.

## Kết luận

**Hai JSON có nền tảng hợp lý cho demo, nhưng chưa nên chốt làm schema benchmark nghiên cứu.** Giữ dữ liệu người dùng tổng hợp, bổ sung định nghĩa, nguồn và trạng thái xác minh; sửa các ánh xạ sai và cách script dùng dữ liệu trước khi mở rộng sinh mẫu.

Việc bốn bảng được tổng hợp thủ công là phương pháp hợp lệ. Kiểm tra này không suy luận chúng được sinh bằng script, và không coi việc tổng hợp thủ công là lỗi. Tính nhất quán giữa các bản xuất không tự chứng minh từng con số đúng với mọi vật thể thuộc lớp.

## Những gì đã kiểm tra

- Lập danh mục và SHA-256 cho 25 file, gồm cả 2 file bytecode; đọc hai script, hai Markdown, các nhãn, JSON/JSONL, CSV; kiểm tra kích thước và metadata của cả 3 JPEG.
- Đối chiếu toàn bộ 123 dòng nhãn nguồn: DOTA 18, iSAID 15, DIOR 20, VisDrone 10, xView 60.
- Đối chiếu toàn bộ 84 lớp giữa `merged_labels.csv`, `physical_sizes.csv`, `physical_sizes.json`: không trùng khóa, không lệch tập lớp; tất cả trường tương ứng khớp nhau; mọi cặp kích thước thỏa `0 < L_min_m <= L_typ_m`.
- Hai bản câu hỏi JSON/JSONL khớp đủ 7 mẫu; ID không trùng, class_id tồn tại, ảnh tồn tại, đáp án thuộc lựa chọn.
- Đọc nội dung 5 PDF hợp lệ, phát hiện 1 file HTML mang đuôi PDF; kiểm tra workbook FAA, các header và dòng aircraft liên quan. Đối chiếu chọn lọc các định nghĩa với nguồn chính thức.
- Tái hiện các nhánh lỗi của generator trên nhãn thật hoặc tập con có kiểm soát từ nhãn thật, không chạy `main()` và không ghi đè đầu ra người dùng.

Giới hạn: đây là rà soát toàn bộ cấu trúc và nội dung bảng, cộng với kiểm chứng nguồn có chọn lọc. Chưa xác minh từng thông số trong toàn bộ 84 lớp với mọi tiêu chuẩn được viện dẫn. Chưa khôi phục ảnh DOTA gốc và lịch sử chuyển đổi, chưa hiệu chỉnh ngưỡng bằng người/MLLM.

## 1. Cấu trúc `mau_7_questions.json`

Danh sách các object là phù hợp. JSONL cũng phù hợp cho xử lý từng mẫu. Không cần đổi cấu trúc chỉ để tăng độ lồng nhau. Điều cần sửa là ý nghĩa trường và khả năng truy ngược bằng chứng.

| Thành phần hiện tại | Đánh giá | Cấu trúc nên dùng |
|---|---|---|
| `id`, `kind` | Hợp lý | Giữ; thêm `schema_version`. ID duy nhất khi sinh nhiều mẫu. Q6 thêm `base_task` để biết nhiệm vụ gốc. |
| `image_path` | Đường dẫn tương đối phù hợp cho chia sẻ dataset | Thêm `image_id` tham chiếu manifest ảnh: dataset/version/split/source_image_id, kích thước ảnh, định danh cảnh gốc, nguồn tải, lịch sử crop/resize. Không cần chép manifest vào mọi câu. |
| `gsd_m` | Đơn vị có thể hiểu nhưng thiếu hệ quy chiếu pixel và nguồn | Đặt ở manifest ảnh; dùng `gsd_native_m_per_px`, `gsd_effective_m_per_px`, `status`, `source_ref`. Giá trị chưa biết để null. Resize phải có scale; crop không resize không tự đổi GSD. |
| `question` | Hợp lý | Giữ; thêm `language`, `template_id`, `scope`, cấu hình lưới dùng chung. |
| `choices` | Chuỗi dùng được cho demo | Mỗi lựa chọn có ID, text và giá trị có kiểu. Số đếm là số nguyên; hộp là mảng 4 số; lựa chọn từ chối có type riêng. |
| `answer` | Dễ chấm demo, nhưng phụ thuộc nguyên văn chuỗi | Dùng `correct_choice_id` và `semantic_answer` có kiểu. Không coi nhãn sự thật, đáp án cần xuất và quyết định từ chối là một trường duy nhất. |
| `class_id` | Khóa nối hợp lý | Giữ ở task/target; cần taxonomy_version. Không đưa class_id vốn là nhãn ẩn vào đầu vào mô hình khi nó tiết lộ đáp án. |
| `target_cell` | Có ích cho câu hỏi vùng | Định nghĩa grid_id, cách xác định ô, quy tắc biên. Không có target_cell ở câu hỏi toàn ảnh là hợp lệ. |
| `target_box_xyxy` | Đang mang nhiều ý nghĩa | Tách `target_instance_ids`, `evidence_instance_ids`, `region`, `instance_geometry`, `union_box`. Q4 không thể tái đếm từ một union box. Q2 không có target thật nên box null là đúng. |
| Hình học còn thiếu | Mất OBB sau khi chuyển HBB | Lưu OBB/polygon gốc, hệ tọa độ, image_id, đơn vị pixel, x sang phải/y xuống dưới, clipped/truncated. Đừng chỉ lưu HBB để suy hướng và cạnh ngắn. |
| `L_m` | Không phân biệt số đo và prior | Tách `physical_prior` với `instance_measurement`. Prior tham chiếu class/dimension/estimate/version; số đo instance ghi phương pháp và nguồn. Q2 dùng prior của lớp vắng mặt phải ghi rõ đây là vật thể giả định. |
| `rho_px` | Một tên cho hai phép đo khác nhau | Phân biệt `rho_px_native`, `rho_px_display`, `rho_px_prior_estimate`, cùng `method`, `dimension_id`, `aggregation`. Không ghi số ước lượng vào trường đo thực tế. |
| `p0_U_px` | Giá trị không kèm căn cứ hiệu chỉnh | Tham chiếu `calibration_id`, task, đơn vị, protocol/version, population và status. Hằng số demo phải ghi `provisional`. |
| `answerable_by_sensor` | Boolean ép kết luận khi thiếu bằng chứng | Dùng enum `answerable`, `unanswerable`, `unknown` và evidence/status. Chưa hiệu chỉnh hoặc chưa rõ native scale thì giữ unknown. |
| `rho_tok` | null hiện tại là hợp lý vì chưa đo | Khi có nhiều mô hình/cấu hình, lưu ở bảng run riêng: sample_id/model/processor/version/input_size/token_grid/target footprint. Không có một rho_tok bất biến cho mỗi câu. |
| `isolation_flag` | Có ích nhưng thiếu quy tắc và chứng cứ | Lưu số instance cùng lớp, khoảng cách láng giềng, rule_id và trạng thái. Q2 có thể not_applicable. Điều kiện dựa trên token phải gắn model/config. |
| `confused_present_class` | Đúng hướng cho Q2 | Thêm instance gây nhầm, evidence của sự hiện diện và phép xác minh lớp hỏi vắng mặt; thiếu nhãn không đủ chứng minh vắng mặt. |
| `unanswerable_reason` | Văn bản giải thích có ích | Thêm reason_code, calibration_id, evidence_refs; văn bản được sinh từ giá trị thực, không chứa số gán cứng. |
| Chất lượng / nguồn | Còn thiếu | `generation_version`, `template_id`, `review_status`, phương pháp tạo nhãn, bằng chứng chú thích, split_group theo cảnh gốc. |

### Thông tin riêng theo loại câu

| Loại | Dữ liệu tối thiểu ngoài phần chung |
|---|---|
| Q1 | Presence truth, evidence instances, phạm vi ảnh/vùng, bằng chứng điều kiện isolated/aggregated. |
| Q2 | Lớp hỏi, lớp gây nhầm thực sự hiện diện, coverage của bộ nhãn, phương pháp xác minh vắng mặt, L_min có phạm vi áp dụng. |
| Q3-HF | Attribute subtype (orientation/subclass), angle hoặc subclass truth, định nghĩa trục, bin edges/tie rule; loại near-square và gần ranh giới; hướng theo ảnh hay hướng địa lý. |
| Q3-LF | Attribute name, palette/version, màu quan sát thực tế, mask/crop hoặc bằng chứng người gán nhãn. |
| Q4 | Instance IDs được tính, count nguyên, cell inclusion rule, đối tượng bị loại và lý do, rho từng instance và phép tổng hợp median. |
| Q5 | Target và candidate instance IDs, quan hệ/thứ hạng, quy tắc sắp xếp/tie, tọa độ và tiêu chí chấm. |
| Q6 | Nhiệm vụ gốc, đáp án chi tiết nếu có bằng chứng độc lập, nhãn từ chối, điều kiện chọn band, nguồn hiệu chỉnh và nguồn xác minh answerability. |

### Hai rò rỉ đáp án cần xử lý ở thiết kế câu

- Q4 sinh lựa chọn `[n-1, n, n+1]` và luôn chọn n. Mô hình có thể chọn giá trị giữa mà không xem ảnh. Xáo thứ tự không loại bỏ quy luật median này; cần thay cả cách chọn distractor.
- Q5 hiện hỏi vật thể thứ hai từ trái và cho ba box của đúng ba vật thể đó. Có thể sắp xếp tọa độ để chọn box thứ hai mà không xem ảnh, kể cả xáo thứ tự. Tọa độ trong lựa chọn tự nó là định dạng hợp lệ; vấn đề là toàn bộ thông tin cần giải đã có trong lựa chọn. Có thể chuyển sang xuất box tự do, hoặc thêm quan hệ/thuộc tính thị giác cần thiết và thiết kế candidates không giải được chỉ bằng tọa độ.

### Cách tổ chức gọn

```text
manifest.json           schema_version, taxonomy_version, grid conventions, splits
images.jsonl            một dòng mỗi phiên bản ảnh; nguồn, dimensions, GSD, transforms
instances.jsonl         instance_id, image_id, class, OBB/mask, annotation provenance
questions.jsonl         id, kind/base_task, image_id, prompt, targets, gold, resolution, quality
model_observations.jsonl sample_id + model/config/run -> rho_tok và thông tin preprocessing
```

35 mẫu vẫn có thể nhúng các nhóm trên vào hai JSON hiện tại. Tách thành nhiều file là lựa chọn triển khai khi mở rộng, không phải điều kiện khoa học bắt buộc. Khi đánh giá, loader chỉ chuyển những trường được giao thức cho phép tới mô hình; gold, target đáp án và nhãn đánh giá phải được giữ cho scorer.

## 2. Cấu trúc `physical_sizes.json`

Map `{class_id: record}` là phù hợp cho tra cứu; `class_id` lặp trong record không phải lỗi nếu kiểm tra chúng bằng nhau. Tên Anh/Việt, domain và mô tả kích thước nên giữ. Vấn đề chính là mỗi lớp hiện chỉ có một cặp số dù nhiều phép đo và phạm vi đang khác nhau.

| Thành phần | Quyết định |
|---|---|
| Khóa lớp / tên hiển thị | Giữ. Có thể nhóm tên trong `names: {en, vi}` nhưng không bắt buộc. |
| `domain_group` | Giữ như phân nhóm. Không dùng nó thay thế quan hệ lớp cha/con. Nếu thực sự đa nhóm thì dùng array. |
| `dataset_mappings` | Chuyển mỗi ánh xạ thành record có dataset, version, source_label, source_id, relation, status, source_ref. Một lớp có nhiều nhãn thì lưu array, tránh chuỗi dấu phẩy. |
| Quan hệ mapping | Định nghĩa chiều: `source_broader` = nhãn nguồn bao hàm canonical; `source_narrower` = nhãn nguồn là tập con của canonical; thêm exact/overlap/unverified. Không tự chia nhãn rộng thành lớp hẹp khi không có chú thích bổ sung. |
| `L_min_m`, `L_typ_m` | Có thể giữ trong một dimension record nếu cùng đại lượng/phạm vi và có định nghĩa thống kê. Nếu chỉ là ví dụ nhỏ và mẫu đại diện, gọi đúng tên `reference_lower_m` / `representative_m`; không gọi minimum/typical như thống kê đã được đo. |
| `critical_dimension_type` | width/length/diameter/extent chưa đủ. Dùng mã cụ thể như fuselage_length, wingspan, vehicle_body_width, court_playing_width, footprint_short_side; ghi rõ whole_object hay component. |
| `critical_dimension_desc` | Giữ để giải thích, nhưng công thức và đơn vị cần trường có cấu trúc để kiểm tra tự động. |
| Một chiều cho mọi task | Chưa đủ. Cho phép `dimensions[]` và task chọn dimension. Phân biệt footprint của vật thể với kích thước bộ phận cần nhận biết. |
| `standard_source`, `standard_org`, `reference_url` | Nên là `sources[]` có source_id; từng estimate trỏ đúng source, trang/mục/bảng/hàng/cột, phiên bản và phạm vi. Nguồn cho L_min có thể khác L_typ. |
| Cách tổng hợp | Thêm original_value/unit, conversion, statistic, population/subtype, selection_method, uncertainty nếu có, evidence_status, verification_notes. `manual_compilation` là giá trị hợp lệ. Không tự điền khoảng tin cậy khi chưa có dữ liệu. |
| Phiên bản | Thêm schema_version, taxonomy_version, size_table_version; chọn CSV hoặc JSON làm bản biên tập chính, xuất và kiểm tra bản còn lại để tránh sửa lệch về sau. |

### Ví dụ vì sao cần định nghĩa dimension/statistic

1. `small_aircraft`: 7.25 m lấy chiều dài Piper Cherokee, 11.00 m lấy sải cánh Cessna 172. Workbook thực sự có 23.8 ft (Piper, dòng 318), 36.1 ft wingspan và 27.2 ft length (Cessna, dòng 137). Các phép đổi đơn vị đúng, nhưng đây là hai chiều vật lý khác nhau. Nếu định nghĩa L là `min(length, wingspan)` thì với chính chiếc Cessna này phải dùng 8.29 m, không phải 11.00 m. Không suy từ ví dụ Piper rằng 7.25 m là minimum cho mọi small aircraft. Chỉ số Col 15/17 trong mô tả là zero-based; cột Excel tương ứng là 16/18, vì vậy nên lưu tên cột.
2. `airport`: 18/45 m mô tả bề rộng đường băng; class_id là toàn sân bay. Có thể lưu như component prior nhưng không gán trực tiếp thành cạnh ngắn OBB sân bay.
3. `baseball_diamond`: 27.43 m khoảng cách giữa các base và 97.54 m khoảng cách tới hàng rào là hai đoạn khác nhau. `length` không diễn đạt được sự khác biệt này.
4. `roundabout`: tài liệu FHWA địa phương đưa khoảng mini-roundabout 13–25 m và urban single-lane 30–40 m (Appendix B, Exhibit B-1, PDF trang 266). 15 và 35 m có thể là các giá trị đại diện trong phạm vi đó; 15 m không phải minimum chung được bảng này hỗ trợ. Mục trích dẫn cần sửa thành đúng bảng.
5. `soccer_ball_field`: IFAB PDF trang 33 quy định bề rộng 45–90 m nói chung, 64–75 m cho trận quốc tế. 68 m có thể là lựa chọn đại diện, không phải kích thước duy nhất bắt buộc theo điều khoản này.

### Lỗi mapping/nội dung đã xác nhận

- `dam -> xView: Dam` không có trong cả raw table người dùng lẫn danh mục xView chính thức. Nhãn 76 là Damaged Building. Cần bỏ mapping này và tính lại số dataset của dam; không đổi raw để hợp thức hóa mapping sai. [Danh mục xView](https://github.com/DIUx-xView/data_utilities/blob/master/xview_class_labels.txt).
- `DIOR: vehicle` xuất hiện ở cả small_vehicle và large_vehicle; `xView: Truck` ở cả large_vehicle và truck. Đây là quan hệ nhiều đích cần khai báo hierarchy/ambiguity, không tự động xem hai đích là nhãn chính xác của cùng instance.
- `people` của VisDrone hiện được mô tả là đám đông, đi cùng đường kính 1–2.5 m. Bài gốc phân biệt người đứng/đi với người ở tư thế khác; nhãn people/person không định nghĩa một cụm đông người. Do đó cần sửa định nghĩa lớp và xem lại dimension. [Bài TPAMI, chú thích 6](https://static.aminer.cn/upload/pdf/249/2019/322/616aacc05244ab9dcb31d1ed_0.pdf).
- Danh mục xView có quan hệ cha/con trong bài gốc, Table 2, PDF trang 11. Một array các tên vẫn chưa đủ nếu bỏ mất quan hệ đó. Một vài cách viết dài trong raw khác tên rút gọn tại code xView; dùng source_id + version và lưu alias, không coi mọi khác biệt chữ là khác lớp.

## 3. Kết quả kiểm tra nguồn và script ảnh hưởng đến hai JSON

### Tài liệu nguồn

| File | Kết quả kiểm tra nội dung |
|---|---|
| `reference/aircraft_data.xlsx` | Workbook đọc được; ACD_Data 389 hàng kể cả header, Data_Dictionary 95 hàng. Các số Piper/Cessna nêu trên tra được. Điều này chưa xác minh toàn bộ prior hàng không. |
| `reference/dota_paper_tpami.pdf` | 10 trang; thực tế là **Single-Shot Non-Gaussian Measurements for Optical Phase Estimation**, không phải bài DOTA. |
| `reference/visdrone_paper_tpami.pdf` | 32 trang; thực tế là **A Blueprint for the Milky Way’s Stellar Populations. III. Spatial Distributions and Population Fractions of Local Halo Stars**, không phải bài VisDrone. |
| `reference/fiba_basketball_rules_2022.pdf` | Bắt đầu bằng `<!DOCTYPE html>`; không phải PDF luật FIBA. Cần lấy đúng tài liệu trước khi coi nguồn cục bộ đã được kiểm chứng. |
| `reference/fhwa_roundabouts.pdf` | PDF 277 trang, đúng Roundabouts: An Informational Guide; có các khoảng kích thước nêu trên. |
| `reference/ifab_laws_of_the_game_2023_24.pdf` | PDF 230 trang, có Law 1 và các kích thước nêu trên. |
| `reference/xview_paper.pdf` | PDF 16 trang, đúng bài xView; có 60 lớp và quan hệ cha/con. |

### Script và mẫu

| Vấn đề | Bằng chứng / ảnh hưởng | Ưu tiên |
|---|---|---|
| GSD không nhất quán | P1053 cùng một JPEG được ghi 0.30 trong Q2/Q3-LF/Q4 và 0.25 trong Q6. Không có phép biến đổi ảnh riêng tương ứng. Ba ảnh chỉ có JFIF metadata, README ghi giá trị xấp xỉ. Chưa có căn cứ xác nhận native GSD. | Cao |
| Lớp prior bị dùng như kích thước instance | Q4 ghi rho=6; các cạnh ngắn OBB tương ứng trong JPEG là 18.4, 18.0, 16.2 px (median 18.0). Đây là pixel ảnh đang có, chưa thể gọi native sensor pixel khi thiếu lịch sử resize. | Cao |
| Ngưỡng p0 chưa hiệu chỉnh | Hằng số 4/5/6/8/12 được script dùng để kết luận sensor-answerability, trong khi bản thảo yêu cầu hiệu chỉnh người và pilot. Cần provisional/unknown và calibration_ref. | Cao |
| Q3-HF | Máy bay P1142 có tỷ lệ cạnh ngắn/dài khoảng 0.883, vượt ngưỡng loại near-square 0.8 trong bản thảo. Cách ánh xạ atan2 sang hai đường chéo đang đảo vì trục y của ảnh hướng xuống. Không có bằng chứng north-up nên không gán phương địa lý thật. OBB cũng không mặc nhiên là trục thân máy bay. | Cao |
| Q3-LF | Màu được gán theo class (pool xanh, soccer xanh lá), không đọc màu ảnh; đúng với một mẫu cụ thể không xác minh được generator cho ảnh khác. | Cao |
| Q2 fallback | Với tập con 3 nhãn basketball thật từ P1470, generator vẫn hỏi basketball và trả lời Không khi không tìm được cặp gây nhầm. Đây là kiểm tra nhánh, không khẳng định Q2 đã lưu mắc lỗi đó. | Cao |
| Q6 dưới band | Truyền g=0.5 cho P1053 cho rho=3.6 < 0.5*12 nhưng trả Xe sedan, answerable=true; không có subclass truth. Band chọn mẫu không phải điều kiện đủ để gán answerable ngoài band. | Cao |
| Q5 chỉ có 2 candidate | Tập con 2 sân bóng rổ thật: câu hỏi vẫn nói thứ hai, target lại là sân trái nhất. | Cao |
| Q4/Q5 không đủ instance | Với P1142 chỉ 1 nhãn, Q4 lỗi min(empty), Q5 lỗi index. Nên skip và ghi rejection reason, không tạo nhãn dự phòng. | Vừa |
| Hai parser dùng class map khác | ID 1–14 khác nhau; script test còn thiếu 15–17. File test không chứng minh production parser đúng vì tự chép lại thuật toán, không có assert. | Cao trước mở rộng |
| Coverage và annotation version | DOTA chuẩn có class name/difficult và metadata; ba nhãn hiện là numeric normalized OBB, đã qua chuyển đổi. Phải lưu converter class map và dataset version; không dùng thứ tự dòng của raw table như ID nguồn mặc định. | Cao |

[Tài liệu DOTA chính thức](https://captain-whu.github.io/DOTA/dataset) mô tả định dạng annotation, metadata GSD có thể thiếu và yêu cầu dùng nhãn đúng phiên bản. Một số đỉnh hiện vượt biên ảnh không tự chứng minh nhãn sai: cần phân biệt polygon đầy đủ, phần nhìn thấy và quy tắc vật thể bị cắt.

## Thứ tự hoàn thiện đề xuất

1. Chốt định nghĩa từng trường, tách prior/số đo/nhãn sự thật/answerability và thêm nguồn/version/status.
2. Sửa dam, people và quan hệ mapping; xác minh/tải lại ba file nguồn sai nội dung.
3. Gắn ảnh với annotation/version, GSD và lịch sử biến đổi thật; dùng null cho thông tin chưa có.
4. Sửa logic generator, loại mẫu không đủ điều kiện; thiết kế lại distractor Q4/Q5.
5. Hiệu chỉnh ngưỡng, kiểm tra thủ công thuộc tính/absence và chống trùng cảnh giữa các split trước khi phát hành benchmark.

Không cần GPU cho các sửa đổi cấu trúc và kiểm tra này. GPU chỉ cần khi đo preprocessing/MLLM và chạy thí nghiệm sau đó.

## Tái lập và giới hạn notebook

`audit_checks.json` chứa kết quả kiểm tra cấu trúc, hình học và nhánh generator; `reference_checks.json` chứa thông tin tài liệu và dòng workbook đã xem. Notebook đồng hành chạy các cell Python tuần tự và lưu stdout bằng nbformat; đã kiểm cấu trúc notebook và kết quả. Môi trường hiện thiếu nbclient/ipykernel nên chưa chạy bằng kernel Jupyter, chưa kiểm giao diện trong notebook viewer. Để kiểm Jupyter độc lập sau khi cài các gói đó: `jupyter nbconvert --execute --to notebook --inplace audit.ipynb`. Dữ liệu gốc được giữ nguyên; notebook nhận SOURCE_DIR để tái lập ở vị trí khác.
