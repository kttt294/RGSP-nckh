# Dự đoán có chọn lọc dựa trên khả năng phân giải cho mô hình ngôn ngữ đa phương thức ảnh viễn thám: Hai nút thắt pixel-token, tiêu chí Johnson cho MLLM và bộ định tuyến trả lời / zoom / từ chối không cần huấn luyện

**Resolvability-Gated Selective Prediction for Remote-Sensing MLLMs: Pixel- and Token-Limited Regimes, a Johnson-Type Criterion for MLLMs, and a Training-Free Answer/Zoom/Abstain Router**

_Bản thảo chuẩn bị nộp IEEE TGRS. Các ô `—` để trống chờ số liệu thực nghiệm; diễn giải kết quả viết ở thể điều kiện._

---

## Tóm tắt

Các mô hình ngôn ngữ lớn đa phương thức cho ảnh viễn thám (RS-MLLM) hallucinate nhiều nhất khi vật thể được hỏi quá nhỏ so với những gì mô hình thực sự nhìn thấy. Các công trình gần đây đã đặt tên hiện tượng ("cannot see clearly", "resolution illusion") nhưng chưa định lượng nó bằng đại lượng tính trước được, chưa tách hai nguyên nhân có hệ quả vận hành đối lập — vật thể _không có trong ảnh_ (giới hạn cảm biến) hay _có trong ảnh nhưng không có trong biểu diễn token_ (giới hạn bộ mã hoá) — và chưa hỏi khi nào mô hình nên từ chối, khi nào nên nhìn kỹ hơn. Viễn thám cổ điển đã có câu trả lời cho câu hỏi thứ nhất dưới dạng tiêu chí Johnson và thang NIIRS: số pixel trên chiều tới hạn quyết định vật thể có thể được phát hiện, nhận dạng hay định danh. Chúng tôi đưa tiêu chí đó vào MLLM và mở rộng nó bằng hai đại lượng tính trước từ siêu dữ liệu: **footprint-in-pixels** ρ_px = L/g (kích thước vật lý chia GSD — đúng đại lượng của Johnson) và **footprint-in-tokens** ρ_tok = ρ_px·s/P (ρ_px nhân hệ số lấy mẫu lại, chia patch hiệu dụng của bộ mã hoá — đại lượng Johnson không có). Chúng tôi (i) xây dựng **RS-Resolve**, benchmark 30-40 nghìn câu hỏi tự sinh từ chú thích phát hiện vật thể và GSD của DOTA-v2, DIOR, xView, VisDrone và iSAID, mỗi câu gắn ρ_px, ρ_tok theo mô hình, mật độ, tần số không gian của thuộc tính hỏi và cờ "không thể trả lời do độ phân giải cảm biến" neo bằng ngưỡng của người; (ii) thực hiện chẩn đoán trên ≥ 8 MLLM bằng thí nghiệm giữ ảnh cố định và thay đổi trần độ phân giải đầu vào của mô hình (ρ_px cố định, ρ_tok đổi), cho thấy ρ_tok dự đoán thất bại độc lập với ρ_px; ngưỡng cảm biến (theo pixel gốc) gần độc lập mô hình trong khi ngưỡng bộ mã hoá ổn định theo token nhưng không theo pixel mô hình hay GSD; và thuộc tính suy giảm theo thứ tự tần số không gian đúng bậc thang detection-recognition-identification của Johnson; (iii) đề xuất **RGSP**, bộ định tuyến không cần huấn luyện với bốn nhánh: từ chối khi ρ_px dưới ngưỡng cảm biến; từ chối khi zoom tối đa cho phép vẫn không đưa ρ_tok qua ngưỡng bộ mã hoá; zoom xác định với kích thước crop tính trước khi zoom đủ; trả lời trực tiếp khi cả hai đủ. Ngưỡng được hiệu chỉnh bằng fixed-sequence testing để đảm bảo selective risk ≤ α với xác suất ≥ 1−δ. Trên cấu hình triển khai chỉ dùng bảng kích thước lớp và GSD, RGSP đạt coverage `—` tại rủi ro 10 % trên tập trả lời được, so với `—` của confidence nội tại tốt nhất, `—` của điểm bằng chứng thị giác và `—` của quy tắc tra bảng lớp x GSD, với `—` lần gọi mô hình bổ sung trung bình. Zoom giảm thất bại chỉ ở ô "có trong ảnh, không có trong token" — hai loại lỗi tách được bằng hai biến tính trước.

**Từ khoá:** mô hình ngôn ngữ đa phương thức, ảnh viễn thám, hallucination, dự đoán có chọn lọc, kiểm soát rủi ro, khoảng cách lấy mẫu mặt đất, tiêu chí Johnson, trả lời câu hỏi thị giác.

---

## I. Giới thiệu

RS-MLLM [GeoChat, VHM, EarthDial, LHRS-Bot, SkySenseGPT] đang trở thành giao diện thống nhất cho VQA, mô tả, định vị và đếm trên ảnh viễn thám, đồng thời với một dòng công trình ghi nhận chúng hallucinate — khẳng định sự tồn tại, thuộc tính hoặc số lượng mà ảnh không đủ bằng chứng [RSHallu, RADAR, CHOICE]. Phân tích lỗi trên ảnh siêu lớn chỉ ra sai sót tập trung vào vật thể chiếm rất ít pixel [XLRS-Bench, RSHR-Bench, UHR-Micro]. RADAR gọi đây là Type 2 "cannot see clearly", phân biệt với Type 1 "cannot find"; UHR-Micro gọi khoảng cách giữa độ phân giải danh nghĩa và nhận thức vi mô là "resolution illusion".

Câu hỏi "bao nhiêu pixel thì thấy được" không mới với viễn thám. Tiêu chí Johnson [Johnson 1958] định lượng số chu kỳ (cặp vạch) trên chiều tới hạn của vật thể cần cho từng mức nhiệm vụ — phát hiện, định hướng, nhận dạng, định danh — và thang NIIRS cùng phương trình GIQE [Leachtenauer] nối GSD, MTF và tỷ số tín hiệu/nhiễu với khả năng diễn giải của con người. Điều mới với MLLM là có **hai** nút thắt thay vì một. Xe con 1,8 m ở GSD 0,6 m chiếm 3 pixel gốc — _không có lượng zoom nào_ thêm thông tin vào 3 pixel đó; đây là giới hạn Johnson. Nhưng cùng chiếc xe ở GSD 0,15 m chiếm 12 pixel gốc; nếu mô hình co ảnh 4 000² về 1 024² thì xe còn 3 pixel _trong không gian mô hình_, rơi gọn vào một token 32 px — vật thể **có trong ảnh nhưng không có trong biểu diễn**, và cắt vùng quanh nó rồi đưa lại sẽ khôi phục đủ 12 pixel. Trường hợp thứ nhất chỉ có một câu trả lời đúng: từ chối. Trường hợp thứ hai sửa được bằng zoom — điều các phương pháp cắt ảnh thích nghi [ViCrop, RADAR, MAP-Agent, GeoEyes] đang khai thác mà không nói rõ. Hai trường hợp phân biệt được trước khi chạy mô hình bằng hai con số:

$$\rho_{px}=\frac{L}{g},\qquad \rho_{tok}=\rho_{px}\cdot\frac{s_m(I)}{P_m},$$

với L kích thước vật lý (m), g GSD (m/px), s*m(I) hệ số lấy mẫu lại của mô hình m trên ảnh I, P_m patch hiệu dụng (px/token). ρ_px đo nút thắt \_cảm biến* — đại lượng của Johnson; ρ*tok đo nút thắt \_bộ mã hoá* — đại lượng Johnson không có vì con người không có patch. Bài báo trả lời ba câu hỏi:

1. **Thất bại có dự đoán được trước khi chạy mô hình không, và từng nút thắt đóng góp bao nhiêu?** Chúng tôi giữ ảnh cố định (ρ_px cố định, ngữ cảnh cố định) và chỉ thay đổi trần độ phân giải đầu vào của mô hình để đổi ρ_tok — thí nghiệm nội tại từng mô hình không có biến gây nhiễu.
2. **Khi nào zoom giúp, khi nào chỉ nên từ chối?** Ma trận 2x2 theo (ρ_px, ρ_tok): zoom chỉ có ích ở ô "có trong ảnh, không có trong token", và chỉ khi zoom tối đa cho phép đủ đưa ρ_tok qua ngưỡng.
3. **Tiên nghiệm tính trước có tốt hơn confidence sinh từ mô hình không?** Dự đoán có chọn lọc cho VLM [SIEVES, FUSE, Variational VQA] và benchmark không thể trả lời [TUBench, HaloQuest, MoHoBench] đều dựa vào tín hiệu _sinh từ mô hình_. ρ*px và ρ_tok quyết định \_trước* khi tốn lần suy luận nào.

Ba đóng góp:

- **RS-Resolve**: benchmark tự sinh, mở rộng được bằng script, với nhãn ρ_px (độc lập mô hình), ρ_tok (theo mô hình, đọc từ lưới token thực), mật độ, tần số không gian của thuộc tính hỏi, cờ cô lập / tổng hợp, và tập "không thể trả lời do độ phân giải cảm biến" với đáp án xác định từ ρ_px và ngưỡng của người. Thiết kế biến thiên trần độ phân giải tách ρ_tok khỏi ρ_px; lấy mẫu lại có kiểm soát tách mật độ khỏi kích thước.
- **Chẩn đoán đa mô hình** trên ≥ 8 MLLM: (H1) ρ_tok mang đóng góp dự đoán riêng sau khi cố định ảnh, kiểm định bằng mô hình hỗn hợp nội-ảnh; (H2) ngưỡng cảm biến p₀ gần độc lập mô hình và gần ngưỡng của người, ngưỡng bộ mã hoá ρ*\_tok ổn định *theo token* xuyên bộ mã hoá trong khi ngưỡng tương đương theo pixel mô hình hoặc GSD thì không; (H3) thuộc tính suy giảm theo thứ tự tần số không gian, khớp bậc thang Johnson; (H4) zoom giảm thất bại chỉ ở ô (ρ_px ≥ p₀, ρ_tok < ρ*).
- **RGSP**: bộ định tuyến bốn nhánh trả lời / zoom xác định / từ chối, không huấn luyện, quyết định không cần chạy mô hình, ngưỡng hiệu chỉnh bằng fixed-sequence testing đảm bảo selective risk ≤ α với xác suất ≥ 1−δ; so với confidence nội tại, điểm bằng chứng thị giác, prompt từ chối, focus test của RADAR, và **quy tắc tra bảng lớp x GSD** — đối chứng trực tiếp cho câu hỏi "chuẩn hoá theo patch có thêm gì". Chúng tôi kiểm tra chuyển giao xuyên bộ dữ liệu và báo rủi ro thực đo được, không chỉ bảo đảm lý thuyết.

Chúng tôi không claim là người đầu tiên đề xuất dự đoán có chọn lọc cho VLM, ghi nhận kích thước ảnh hưởng MLLM, xây benchmark không thể trả lời, hay nối số pixel với khả năng diễn giải — Johnson làm điều đó năm 1958. Điều mới: đo tiêu chí Johnson cho MLLM thay cho người, tách thêm nút thắt bộ mã hoá mà người không có, và dùng hai đại lượng đó làm tín hiệu định tuyến có bảo đảm rủi ro.

---

## II. Công trình liên quan

### A. Hallucination trong RS-MLLM

RSHallu [x] xây ba bộ đánh giá (2 023 / 15 396 / 30 000 câu) và giảm hallucination bằng hiệu chỉnh logit kèm prompt nhận biết viễn thám. RADAR [x] phân loại lỗi grounding thành Type 1 "cannot find" và Type 2 "cannot see clearly", đề xuất zoom dựa attention có điều kiện theo câu hỏi kèm _focus test_ dựa entropy attention để fallback; RSHBench đi kèm gồm 371 cặp ảnh-câu hỏi. VHM [x] huấn luyện trung thực với câu hỏi đánh lừa (HnstD); CHOICE [x] ghi nhận VHM là RS-VLM duy nhất không tụt so với VLM tổng quát ở phát hiện hallucination, và năng lực từ chối của VLM tổng quát dưới 70 %. XLRS-Bench [x] và RSHR-Bench [x] ghi nhận lỗi tập trung ở vật thể ít pixel trên ảnh cạnh 8 500-10 000 px. Không công trình nào tách nguyên nhân cảm biến khỏi nguyên nhân bộ mã hoá; chỉ VHM có cơ chế từ chối, nhưng là từ chối _về sự không tồn tại_ học được, không về khả năng phân giải.

### B. Khả năng phát hiện theo độ phân giải: từ Johnson đến MLLM

Viễn thám và ảnh quân sự có truyền thống dài định lượng "bao nhiêu pixel thì đủ". Tiêu chí Johnson [Johnson 1958] cho số chu kỳ trên chiều tới hạn cần cho detection (≈1), orientation (≈1,4), recognition (≈4), identification (≈6,4), với xác suất 50 %; các bản cập nhật [Vollmerhausen; Driggers] tinh chỉnh bằng hàm nhiệm vụ. Thang NIIRS [Leachtenauer] và phương trình GIQE nối GSD, MTF, SNR với mức diễn giải của người; các nghiên cứu phát hiện vật thể nhỏ trong viễn thám xác nhận ngưỡng theo pixel cho từng lớp (ví dụ ≈ 0,5 m/px cho gia súc; xe 4x1,5 m ở 0,5 m/px chiếm 8x3 px) [RSSOD; cattle detection]. Toàn bộ văn liệu này đo **người** hoặc **detector chuyên dụng**, và có đúng một nút thắt: pixel gốc.

Với MLLM tổng quát, "Do You See Me" [x] cho thấy thất bại tập trung "ở hoặc dưới độ phân giải patch của bộ mã hoá" trên hình tổng hợp; ViCrop [x] cải thiện độ chính xác bằng cắt vùng — bằng chứng rằng một phần lỗi là token-limited. UHR-Micro [x] định nghĩa "resolution illusion" theo tỷ lệ diện tích và đề xuất MAP-Agent; ScaleEarth [x] coi GSD là biến vật lý liên tục để _huấn luyện_ (CS-HLoRA, SSE-U). Không nhóm nào đo tiêu chí Johnson cho MLLM trên ảnh viễn thám thật, tách ρ*px khỏi ρ_tok, hỏi khi nào zoom vô ích, hay dùng hai đại lượng đó để định tuyến. Bậc thang nhiệm vụ của Johnson còn dự đoán thứ tự suy giảm của thuộc tính: phát hiện sự tồn tại mất sau cùng, định danh loại con mất trước; và bài toán \_mixed pixel* [spectral unmixing] cho biết thuộc tính phổ (màu) sống dưới kích thước pixel qua trộn tuyến tính trong khi thuộc tính hình học thì không. Chúng tôi đưa cả hai vào thiết kế câu hỏi (Mục III-C).

### C. Dự đoán có chọn lọc và câu hỏi không thể trả lời

Geifman và El-Yaniv [x] định nghĩa selective risk / coverage và thuật toán SGR chọn ngưỡng với bảo đảm (1−δ); Learn-then-Test [Angelopoulos] tổng quát hoá thành kiểm định đa mức với fixed-sequence testing, không cần loss đơn điệu; Conformal Risk Control [Angelopoulos] cần loss đơn điệu và kiểm soát rủi ro _biên_ — không phải đại lượng chúng tôi cần (Mục III-E). Cho VLM, SIEVES [x] hiệu chỉnh selector từ điểm bằng chứng thị giác, tăng coverage tới ba lần, **có tập con viễn thám** (MME-RealWorld-Lite); FUSE [x] hợp nhất bất định epistemic và aleatoric; Variational VQA [x] dùng suy diễn biến phân. Benchmark không thể trả lời [TUBench, HaloQuest, MoHoBench, MM-AQA, UA-Bench] không có miền viễn thám và không định nghĩa "không thể trả lời" theo vật lý ảnh. OATS-RS [x] áp dụng risk-coverage cho phân loại cảnh zero-shot. Điểm chung: tín hiệu tin cậy sinh từ mô hình. Tiên nghiệm của chúng tôi bổ sung, không thay thế — RGSP+ kết hợp ρ với logit và focus test.

### D. Zoom, tiling và chọn token

GeoLLaVA-8K [x], LRS-VQA [x], UHR-BAT [x], GeoEyes [x], UAV-MAS [x] mở rộng độ phân giải hữu dụng bằng cắt tỉa, chọn token, tập trung theo yêu cầu hoặc điều phối công cụ. Tất cả ngầm định "nhìn kỹ hơn thì tốt hơn". Chúng tôi chỉ ra điều đó đúng trong đúng một ô của ma trận (ρ_px, ρ_tok), và chỉ khi zoom tối đa cho phép đủ.

### E. Định vị

**BẢNG I. Định vị so với công trình liên quan.**

| Phương pháp            | Miền            | Tín hiệu            | Cần chạy mô hình để quyết định | Không huấn luyện | Từ chối           | Tách cảm biến / bộ mã hoá | Định lượng kích thước → thất bại | Zoom                    |
| ---------------------- | --------------- | ------------------- | ------------------------------ | ---------------- | ----------------- | ------------------------- | -------------------------------- | ----------------------- |
| Johnson / NIIRS / GIQE | RS (người)      | ρ_px, MTF, SNR      | —                              | —                | —                 | chỉ cảm biến              | ✓ (người)                        | —                       |
| RADAR (focus test)     | RS              | Entropy attention   | Có                             | ✓                | fallback          | ✗                         | ✗                                | ✓                       |
| RSHallu-Shield         | RS              | Hiệu chỉnh logit    | Có                             | ✓                | ✗                 | ✗                         | ✗                                | ✗                       |
| VHM                    | RS              | Trung thực học được | Có                             | ✗                | ✓ (không tồn tại) | ✗                         | ✗                                | ✗                       |
| UHR-Micro / MAP-Agent  | RS              | —                   | Có                             | ✓                | ✗                 | ✗                         | một phần (tỷ lệ diện tích)       | ✓                       |
| ScaleEarth             | RS              | Bất định GSD        | Có                             | ✗                | ✗                 | ✗                         | ✗                                | ✗                       |
| SIEVES                 | Tổng quát (+RS) | Điểm bằng chứng     | Có                             | ✓                | ✓                 | ✗                         | ✗                                | ✗                       |
| FUSE                   | Tổng quát       | Bất định Bayes      | Có                             | ✓                | ✓                 | ✗                         | ✗                                | ✗                       |
| ViCrop / Do You See Me | Tổng quát       | —                   | Có                             | ✓                | ✗                 | ✗                         | ✓ (theo patch)                   | ✓                       |
| OATS-RS                | RS (cảnh)       | Confidence          | Có                             | ✓                | ✓                 | ✗                         | ✗                                | ✗                       |
| **RGSP**               | RS              | **ρ_px, ρ_tok**     | **Không**                      | ✓                | ✓                 | **✓**                     | ✓ (MLLM)                         | **định tuyến xác định** |

Chúng tôi phân biệt với "From Pixels to Tokens" [x] — quy hallucination về lớp căn chỉnh, không định nghĩa đại lượng kích thước theo token.

---

## III. Phương pháp

### A. Hai footprint

Xét câu hỏi q về vật thể lớp c trong ảnh I có GSD g. L (m) là kích thước vật lý đặc trưng: nếu q tham chiếu instance có box hoặc mask, L = min(w, h)·g (báo thêm max(w, h) và √diện tích mask cho lớp thuôn dài); nếu q hỏi sự tồn tại của lớp chưa chắc có, L = L_c^min — **kích thước nhỏ nhất hợp lý** của lớp theo bảng (Phụ lục A), để logic "ảnh không loại trừ được lớp này" đúng cho mọi thành viên của lớp.

**Footprint theo pixel gốc.** ρ_px = L/g là số pixel cảm biến trên chiều nhỏ của vật thể — chính là đại lượng của tiêu chí Johnson (Johnson dùng chu kỳ, bằng ρ_px/2 khi lấy mẫu Nyquist). Độc lập mô hình. Dùng để định nghĩa đáp án benchmark.

**Footprint theo token.** Mô hình m biến ảnh I thành lưới token bằng bộ tiền xử lý riêng: co giãn (Qwen3-VL với trần max_pixels; LLaVA-1.5 co về 336; GeoChat về 504), chia tile (InternVL3.5: tile 448² cộng thumbnail; LLaVA-OneVision: base view cộng lưới anyres), rồi mã hoá với patch p và gộp token (MLP-merger 2x2 của Qwen3-VL; pixel-unshuffle của InternVL). Gọi P_m là patch hiệu dụng (px mô hình / token, sau gộp) và s_m(I, v) là hệ số px mô hình / px gốc của view v chứa vật thể. Định nghĩa

$$\rho_{tok}^{(m)}(q,I)=\max_{v\ni\text{obj}}\ \rho_{px}\cdot\frac{s_m(I,v)}{P_m}. \tag{1}$$

Lấy max trên các view (tile, thumbnail, base view) vì ρ_tok đo view thuận lợi nhất mà mô hình có. **Trong thực nghiệm, s_m(I, v) không tính bằng công thức**: chúng tôi ghi lưới token và ánh xạ toạ độ của bộ tiền xử lý cho từng ảnh và từng trần độ phân giải, rồi đọc ρ_tok từ số token mà box vật thể phủ trên chiều nhỏ. Công thức chỉ để giải thích; Bảng XII kiểm tra công thức sai lệch bao nhiêu so với lưới thực.

**BẢNG II. Bộ mã hoá thị giác, patch hiệu dụng và trần độ phân giải dùng trong thí nghiệm biến thiên.** Cột s chỉ minh hoạ; số liệu thực đọc từ lưới token.

| Mô hình                                       | Bộ mã hoá         | Chính sách đầu vào           | Patch p | Gộp token     | **P (px/token)** | Trần độ phân giải dùng để biến thiên         | Nguồn    |
| --------------------------------------------- | ----------------- | ---------------------------- | ------- | ------------- | ---------------- | -------------------------------------------- | -------- |
| Qwen3-VL-8B                                   | SigLIP2-SO400M    | co giãn động theo max_pixels | 16      | merger 2x2    | **32**           | max_pixels ∈ {256², 512², 1 024², 2 048²}    | [cfg]    |
| InternVL3.5-8B                                | InternViT-300M    | tile 448² + thumbnail        | 14      | unshuffle 2x2 | **28**           | max_tiles ∈ {1, 4, 12}                       | [cfg]    |
| LLaVA-OneVision-7B                            | SigLIP-SO400M@384 | base + anyres                | 14      | không         | **14 / ô**       | grid ∈ {1x1, 2x2, 3x3}                       | [cfg]    |
| LLaVA-1.5-7B                                  | CLIP ViT-L/14@336 | co về 336                    | 14      | không         | **14**           | resize ∈ {224, 336} (nội suy vị trí cho 448) | [cfg]    |
| GeoChat-7B                                    | CLIP ViT-L/14@504 | co về 504                    | 14      | không         | **14**           | resize ∈ {336, 504}                          | [cfg]    |
| SkySenseGPT / EarthDial / VHM / LHRS-Bot-Nova | —                 | —                            | —       | —             | —                | theo cấu hình                                | [cfg]    |
| GPT-5.x / Gemini 3 Pro ‡                      | không biết        | không biết                   | —       | —             | n/a              | không biến thiên được                        | chỉ ρ_px |

Hệ quả then chốt: với ảnh đầu vào cố định và trần độ phân giải cố định, s là hằng và ρ_tok tỷ lệ cứng với ρ_px ở **mọi** mô hình. Có hai cách phá tỷ lệ này. **Cách 1 — đổi trần độ phân giải của mô hình trên cùng một ảnh**: mọi thứ trong ảnh không đổi (ρ_px, ngữ cảnh, số instance, mật độ, vị trí), chỉ s đổi. Đây là thiết kế trung tâm (Mục IV-C). **Cách 2 — đổi kích thước crop quanh cùng vật thể**: s đổi nhưng ngữ cảnh cũng đổi; đây là thao tác zoom thực làm, dùng làm thiết kế thứ hai và cho H4.

**Biến đối chứng.** Mật độ (instance cùng lớp trên mỗi token; khoảng cách láng giềng gần nhất theo token); số lượng trong khung; log g; lớp (hiệu ứng cố định); loại cảnh; vị trí vật thể trong khung. Không dùng areaFrac trong thiết kế cách 2 vì nó cộng tuyến hoàn hảo với ρ_tok nội-instance (log areaFrac = 2·log s + const).

### B. Ba chế độ lỗi và phạm vi của khung

- **Pixel-limited** (ρ_px < p₀): thông tin không có trong ảnh gốc — giới hạn Johnson. Không mô hình nào, không thao tác nào trên ảnh này khắc phục được. Câu trả lời đúng cho thuộc tính tần số cao là "không thể xác định từ ảnh".
- **Token-limited** (ρ_px ≥ p₀, ρ_tok < ρ*\_m): vật thể có trong ảnh nhưng bộ tiền xử lý làm nó rơi dưới ngưỡng token. Zoom (tăng s) có thể khắc phục — *nếu\* s tối đa cho phép đủ.
- **Attention-limited** (cả hai đủ, mô hình chú ý sai vùng — Type 1 của RADAR): **không dự đoán được từ siêu dữ liệu.** Khung ρ không bao phủ; cần tín hiệu từ mô hình. Đây là lý do của RGSP+.

Hai ngưỡng cảm biến khác nhau về vai trò:

- **p₀^50**: điểm thất bại 50 % theo ρ*px của một mô hình dưới điều kiện không bị nút thắt token — tương đương ngưỡng Johnson 50 %. Chỉ dùng để \_mô tả* đường cong và so xuyên mô hình (H2).
- **p₀^U**: giá trị ρ*px dưới đó độ chính xác thuộc tính tần số cao **không phân biệt được với mức ngẫu nhiên** (kiểm định nhị thức một phía, α = 0,05, trên cụm ρ_px) ở \_mọi* mô hình pilot **và** ở người (Mục IV-F). p₀^U := min(p₀^U,models, p₀^U,human) — bảo thủ về phía "còn trả lời được". Chỉ p₀^U định nghĩa tập không thể trả lời U. p₀^U được ước lượng **một lần từ pilot, trên phần hiệu chỉnh của pilot, cố định trước khi sinh benchmark đầy đủ**, và không cập nhật theo mô hình đánh giá sau đó; báo độ nhạy mọi kết quả với p₀^U ± 1 ô.

ρ*\_m là ngưỡng *bộ mã hoá*, theo mô hình. Cả p₀ và ρ* là ngưỡng thực nghiệm: tương phản, MTF, pan-sharpening, số kênh phổ và phương pháp lấy mẫu lại dịch chuyển chúng — đúng như GIQE có thêm MTF và SNR ngoài GSD. ρ_px và ρ_tok là đại lượng bậc nhất; phần dư trong hồi quy là chỗ cho tiên nghiệm bậc hai.

### C. Thuộc tính theo tần số không gian và bậc thang Johnson

Không mọi thuộc tính mất cùng lúc dưới p₀. Màu là thuộc tính tần số thấp: xe đỏ chiếm 0,3 pixel vẫn kéo giá trị phổ pixel về phía đỏ (mixed pixel). Trục định hướng, hình dạng, loại con và đếm là tần số cao. Bậc thang Johnson dự đoán thứ tự trong nhóm tần số cao: tồn tại (detection) < trục định hướng (orientation) < loại lớp (recognition) < loại con (identification). Chúng tôi phân tầng mọi câu hỏi thuộc tính thành **LF** (màu) và **HF** (tồn tại-iso, trục định hướng 2 bin, trục định hướng 4 bin, lớp, loại con); chỉ HF định nghĩa "không thể xác định"; LF là đối chứng dương. Dự đoán kiểm định được (H3): **ngưỡng của LF thấp hơn HF; trong HF, ngưỡng tăng theo bậc thang Johnson.** Nếu ρ đúng là đại lượng phân giải, thứ tự này phải tái hiện; nếu không, ρ đang đo cái khác. Với LF chúng tôi cũng ước lượng sàn p₀^LF (điểm không phân biệt được với ngẫu nhiên) để định nghĩa tập trả lời được của LF.

### D. Phân loại kết cục và định nghĩa metric

Mỗi câu trả lời được gán một kết cục:

- **Hallucination**: khẳng định tồn tại lớp không có (Q2 với ρ_px giả định ≥ p₀^U); gán thuộc tính HF, vị trí hoặc số lượng xác định cho instance có ρ_px < p₀^U; trả lời có nội dung khi đáp án là "không thể xác định".
- **Lỗi nhận thức**: sai ở vùng ρ_px ≥ p₀^U. Báo riêng.
- **Từ chối**: chọn "Không thể xác định từ ảnh". Đúng trên tập không thể trả lời; là _từ chối thừa_ trên tập trả lời được.

Tập **trả lời được** A = {q ∈ HF : ρ_px ≥ p₀^U} ∪ {q ∈ LF : ρ_px ≥ p₀^LF}; tập **không thể trả lời** U = phần bù. Trên A: **coverage** = tỷ lệ được trả lời; **selective risk** = tỷ lệ thất bại trong số được trả lời; AURC theo Geifman và El-Yaniv. Trên U: **abstention accuracy** = tỷ lệ từ chối đúng. Từ chối đúng trên U không bị tính là mất coverage. Báo thêm **selective accuracy tổng** trên A ∪ U. Tuỳ chọn từ chối trong mọi câu trắc nghiệm là "**Không thể xác định từ ảnh**" — trung tính, không nhắc độ phân giải, để không gợi ý lý do từ chối.

### E. RGSP: bộ định tuyến bốn nhánh với bảo đảm selective risk

Cho mô hình m, mức rủi ro α, mức tin cậy 1−δ. Định nghĩa **s_max** là hệ số lấy mẫu lại lớn nhất mà bộ định tuyến được phép tạo ra bằng zoom (mặc định s_max = 1: cắt về độ phân giải gốc, không phóng; báo thêm s_max = 2 trong ablation), và

$$\rho_{tok}^{max}=\rho_{px}\cdot\frac{s_{max}}{P_m}, \tag{2}$$

giá trị ρ_tok tốt nhất đạt được bằng zoom, tính trước từ siêu dữ liệu. Bộ định tuyến:

$$\pi(q,I)=\begin{cases}\text{từ chối (pixel-limited)} & \rho_{px}<p_0^U\\ \text{từ chối (zoom không đủ)} & \rho_{px}\ge p_0^U,\ \rho_{tok}^{max}<\tau\\ \text{zoom về crop }c^*\text{ rồi trả lời} & \rho_{px}\ge p_0^U,\ \rho_{tok}<\tau\le\rho_{tok}^{max}\\ \text{trả lời trực tiếp} & \rho_{tok}\ge\tau\end{cases} \tag{3}$$

với **c\*** là kích thước crop lớn nhất quanh vị trí vật thể sao cho s_m(crop) đưa ρ_tok ≥ τ — tính từ chính sách tiền xử lý đã ghi, không cần chạy mô hình. p₀^U cố định trước (Mục III-B); τ là tham số duy nhất được hiệu chỉnh. **Nhánh zoom chỉ áp cho câu hỏi có vị trí vật thể trong câu** (Q3-Q5: ô lưới, góc, mô tả vị trí). Với câu hỏi tồn tại (Q1-Q2), vị trí chưa biết: bảng chính dùng **RGSP-abstain** (nhánh 3 thay bằng từ chối); Phụ lục D báo biến thể tiling xác định (quét ảnh bằng tile cỡ c\*, chi phí = số tile, tính trước) để hoàn chỉnh. Biến thể RGSP-abstain là phiên bản chi phí bằng không thuần tuý cho mọi loại câu hỏi.

**Chọn τ với bảo đảm.** Selective risk R(τ) = E[ℓ | được trả lời dưới τ] không đơn điệu theo τ, nên Conformal Risk Control — cần loss đơn điệu và chỉ kiểm soát rủi ro biên — không áp dụng được; rủi ro biên còn ép về ≤ α được bằng cách từ chối nhiều, vô nghĩa với claim coverage. Chúng tôi dùng fixed-sequence testing của Learn-then-Test với bound Clopper-Pearson:

**Thuật toán 1 — Hiệu chỉnh τ (fixed-sequence, Clopper-Pearson)**

```
Đầu vào: D_cal = {(q_i, I_i, ρ_px,i, ρ_tok,i, ρ_tok,i^max)} thuộc A; lưới τ₁ > τ₂ > … > τ_K (bảo thủ → lỏng); α; δ
Chuẩn bị (chạy mô hình một lần cho mỗi câu và mỗi nhánh khả dĩ):
    ℓ_i^direct ← thất bại khi trả lời trực tiếp
    ℓ_i^zoom   ← thất bại khi trả lời sau zoom về c*(τ)   # chỉ Q3-Q5; với Q1-Q2 nhánh này không tồn tại
τ̂ ← +∞
for k = 1..K:
    S_k ← {i : ρ_tok,i ≥ τ_k}  ∪  {i : ρ_tok,i < τ_k ≤ ρ_tok,i^max, q_i ∈ Q3-Q5}   # tập được trả lời dưới τ_k
    ℓ_i ← ℓ_i^direct nếu ρ_tok,i ≥ τ_k, ngược lại ℓ_i^zoom
    n_k ← |S_k|;  x_k ← Σ_{i∈S_k} ℓ_i
    U_k ← ClopperPearsonUpper(x_k, n_k, δ)
    if U_k ≤ α:  τ̂ ← τ_k
    else:        break
return τ̂
```

Các giả thuyết H_k: "R(τ_k) > α" kiểm định theo thứ tự cố định, dừng ở lần đầu không bác được → tỷ lệ lỗi gia đình ≤ δ không cần hiệu chỉnh đa kiểm định, và **P(R(τ̂) ≤ α) ≥ 1−δ** khi D_cal và test i.i.d. hoặc exchangeable [LTT, Thm. 1]. Hiệu chỉnh trên toàn hệ (kể cả nhánh zoom), vì bảo đảm phải áp cho câu trả lời người dùng nhận được.

**Hai điều cần nói rõ về chi phí.** (i) "Không cần chạy mô hình" áp cho _quyết định định tuyến khi triển khai_; hiệu chỉnh cần chạy mô hình một lần trên n ≈ 300 câu cho mỗi mô hình mới (Bảng VIII kiểm tra n = 300 đủ chưa). (ii) RGSP-abstain: không lần gọi thêm. RGSP đầy đủ: một lần gọi thêm _chỉ khi_ rơi vào nhánh zoom; báo số lần gọi kỳ vọng trên phân phối test.

**Exchangeability không tự động thoả** khi hiệu chỉnh trên DOTA và triển khai trên xView. Do đó: mọi bảng báo _rủi ro thực đo được_ trên test; Bảng XIII kiểm tra chuyển giao xuyên bộ dữ liệu; biến thể **Mondrian** hiệu chỉnh τ riêng theo ô GSD.

**RGSP+.** Để đo ρ _bổ sung_ gì cho tín hiệu mô hình và chạm tới chế độ attention-limited, hợp nhất ρ_tok với max-logit (hoặc focus test của RADAR) bằng hồi quy logistic. D_cal chia đôi theo ảnh: **nửa khớp** hệ số hồi quy, **nửa hiệu chỉnh** chạy Thuật toán 1 trên điểm hợp nhất — tránh dùng cùng dữ liệu cho hai việc. RGSP+ cần một lần chạy mô hình như mọi baseline confidence nội tại.

**Quy tắc tra bảng làm đối chứng.** Trong cấu hình triển khai, L = L*c theo lớp; gate "ρ_tok ≥ τ" tương đương "g ≤ L_c·s/(P·τ)". Reviewer có quyền hỏi RGSP khác gì bảng tra lớp x GSD. Chúng tôi đưa chính bảng tra đó — ngưỡng theo lớp trên g, hiệu chỉnh bằng cùng Thuật toán 1 nhưng \_không* chia cho s/P — làm baseline "Naive". Nếu chuẩn hoá theo bộ mã hoá không thêm coverage tại α, RGSP không hơn bảng tra và chúng tôi sẽ nói vậy. Bằng chứng thứ hai là H2.

---

## IV. Benchmark RS-Resolve

### A. Nguồn dữ liệu

Năm bộ phát hiện vật thể công khai có GSD hoặc suy được GSD. DOTA-v2.0 [x] nguồn chính (GSD 0,1-4,5 m, nhiều lớp, OBB); iSAID [x] mask instance trên cùng ảnh. DIOR [x] bổ sung GSD thô (tới 30 m). xView [x] GSD 0,3 m với 60 lớp con có kích thước vật lý — tập kiểm soát g và nguồn câu hỏi loại con. VisDrone [x] cực trị mật độ; GSD ước lượng từ độ cao, gắn cờ, chỉ dùng cho độ nhạy. Gắn ρ_px hậu kỳ cho mẫu có GSD trong RSHBench, RSHalluEval và CHOICE-HD. Không phân phối lại ảnh; phát hành QA, mã ảnh, nhãn và script.

**BẢNG III-a. Thống kê theo nguồn.**

| Nguồn               | Ảnh | GSD (m)   | Nguồn GSD    | Lớp | Q1  | Q2  | Q3-HF | Q3-LF | Q4  | Q5  | Q6  | Tổng       | Gốc / lấy mẫu lại | Tập biến thiên | Kiểm tay              |
| ------------------- | --- | --------- | ------------ | --- | --- | --- | ----- | ----- | --- | --- | --- | ---------- | ----------------- | -------------- | --------------------- |
| DOTA-v2.0 / iSAID   | —   | 0,1-4,5   | siêu dữ liệu | —   | —   | —   | —     | —     | —   | —   | —   | —          | — / —             | —              | —                     |
| DIOR                | —   | 0,5-30    | siêu dữ liệu | —   | —   | —   | —     | —     | —   | —   | —   | —          | — / —             | —              | —                     |
| xView               | —   | 0,3       | cố định      | —   | —   | —   | —     | —     | —   | —   | —   | —          | — / 0             | —              | —                     |
| VisDrone            | —   | ước lượng | độ cao (cờ)  | —   | —   | —   | —     | —     | —   | —   | —   | —          | — / 0             | —              | —                     |
| Ngoài (ρ_px hậu kỳ) | —   | —         | —            | —   | —   | —   | —     | —     | —   | —   | —   | —          | —                 | —              | —                     |
| **Tổng**            |     |           |              |     |     |     |       |       |     |     |     | **30-40k** |                   |                | **2 000 + 500 + 500** |

### B. Loại câu hỏi

**BẢNG IV. Loại câu hỏi, đáp án và nhãn.** Mọi câu ở dạng trắc nghiệm có tuỳ chọn "Không thể xác định từ ảnh", kèm giao thức mở chấm bằng LLM-judge và kiểm tra người 500 mẫu.

| Loại                             | Ví dụ                                                                                                                 | Đáp án                                                                                                                                                       | Điều kiện sinh                                                                                                                                                                                                                      | Nhãn                                      |
| -------------------------------- | --------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------- |
| Q1 Tồn tại (+)                   | "Có xe ô tô nhỏ trong ảnh không?"                                                                                     | Có                                                                                                                                                           | **Q1-iso** (≤ 3 instance cùng lớp trong khung, láng giềng ≥ 4 token) và **Q1-agg** (tổng hợp)                                                                                                                                       | ρ_px, ρ_tok, số lượng, mật độ, cờ iso/agg |
| Q2 Tồn tại (−), phủ định khó     | "Có sân bóng chày trong ảnh không?" (chỉ có sân bóng đá)                                                              | **"Không"** nếu ρ_px giả định (L_c^min của lớp hỏi) ≥ p₀^U; **"Không thể xác định"** nếu < p₀^U                                                              | Cặp lớp gần nghĩa, **khác hình thái, không khác cỡ**: sân bóng chày / sân bóng đá; cầu / đập; bể chứa / bùng binh; sân tennis / sân bóng rổ; trực thăng / máy bay nhỏ. Không dùng small/large-vehicle, không dùng tàu hàng / thuyền | ρ_px giả định, ρ_tok giả định             |
| Q3-HF Thuộc tính tần số cao      | "Trục dài của xe ở ô B2 gần với hướng nào: Bắc-Nam hay Đông-Tây?" (2 bin) / 4 bin / "Tàu ở góc trên trái là loại gì?" | Từ OBB: **trục định hướng 0-180°** (không hỏi heading, trừ máy bay có mũi rõ và đã kiểm tra tay); lớp con từ xView; **"Không thể xác định"** nếu ρ_px < p₀^U | Chỉ instance cô lập; ghi bậc Johnson (orientation / recognition / identification)                                                                                                                                                   | ρ_px, ρ_tok, HF, bậc Johnson              |
| Q3-LF Thuộc tính tần số thấp     | "Xe ở ô B2 màu gì?"                                                                                                   | Màu lượng hoá 6 nhóm, **GT lấy từ ảnh gốc mịn trong cặp lấy mẫu lại** (không từ mask 1-2 px); kiểm tra người                                                 | Vị trí chỉ rõ; sinh trên cả ρ_px < p₀^U để đo suy giảm LF                                                                                                                                                                           | ρ_px, ρ_tok, LF                           |
| Q4 Đếm theo ô lưới               | "Có bao nhiêu xe trong ô B2?"                                                                                         | Số từ chú thích, loại box < 50 % trong ô; "Không thể xác định" nếu ρ_px trung vị < p₀^U                                                                      | Báo riêng theo mật độ                                                                                                                                                                                                               | ρ_px trung vị, số lượng, mật độ           |
| Q5 Tham chiếu                    | "Xe thứ ba từ trái ở hàng trên" → box                                                                                 | Box; "Không thể xác định" nếu ρ_px < p₀^U                                                                                                                    | Chỉ khi thứ tự xác định được                                                                                                                                                                                                        | ρ_px, láng giềng                          |
| Q6 Không thể trả lời do cảm biến | Câu Q3-HF / Q4 / Q5 về instance có ρ_px **ngay dưới** p₀^U (0,5·p₀^U ≤ ρ_px < p₀^U)                                   | "Không thể xác định từ ảnh"                                                                                                                                  | Không dùng câu bất khả thi hiển nhiên                                                                                                                                                                                               | ρ_px, lý do                               |

Đáp án benchmark chỉ phụ thuộc ρ_px và p₀^U — độc lập mô hình đánh giá; ρ_tok không tham gia định nghĩa đáp án. Mọi ước lượng p₀ và ρ\* chỉ dùng Q1-iso và Q3/Q4/Q5 trên instance cô lập.

### C. Thí nghiệm biến thiên trần độ phân giải (thiết kế trung tâm cho H1, H2)

Với mỗi ảnh trong tập biến thiên (mục tiêu 4 000 instance cô lập trải 5 ô ρ_px, trên crop cố định 1 024² hoặc ảnh gốc nếu nhỏ hơn), cùng câu hỏi được hỏi ở **mọi trần độ phân giải** của mô hình liệt kê trong Bảng II. Ảnh không đổi → ρ_px, ngữ cảnh, số instance, mật độ, vị trí không đổi; chỉ s_m đổi (đọc từ lưới token thực) → chỉ ρ_tok đổi. Cho câu hỏi j về instance i ở trần r:

$$\text{logit}\,P(\ell_{ijr}=1)=\beta_0+\beta_1\log\rho_{tok,ir}+u_i+v_{c(i)}, \tag{4}$$

với u_i hiệu ứng ngẫu nhiên theo instance (hấp thụ ρ_px, lớp cụ thể, tương phản, ngữ cảnh — mọi thứ cố định theo ảnh) và v_c theo lớp. **H1 là β₁ < 0 có ý nghĩa.** Không có biến gây nhiễu nào đổi cùng ρ_tok trong thiết kế này, nên (4) không cần thêm biến đối chứng; các biến đó chỉ vào mô hình mô tả xuyên-instance. Với mô hình đóng không đổi được trần, chúng tôi báo đường cong theo kích thước crop (thiết kế 2) mà không quy về β₁.

**Thiết kế 2 — biến thiên crop (cho H4 và cho zoom thực tế).** Bốn crop đồng tâm 512 / 768 / 1 024 / 1 536 quanh cùng instance, vị trí ngẫu nhiên trong crop. s đổi nhưng ngữ cảnh cũng đổi; mô hình hỗn hợp cho thiết kế này chứa log ρ_tok, count, density và **không chứa areaFrac** (cộng tuyến hoàn hảo nội-instance). Kết quả thiết kế 2 dùng để kiểm tra β₁ tái hiện khi zoom là thao tác thật, và làm nguồn cho ma trận H4.

### D. Tách mật độ khỏi kích thước

Phân tầng 5 ô GSD x 5 ô ρ_px x 3 ô mật độ, ≥ 300 câu ở mọi ô lệch đường chéo. Ô thiếu bổ sung bằng lấy mẫu lại có kiểm soát ảnh GSD mịn (Lanczos-3 kèm PSF Gaussian) — đổi ρ_px, giữ mật độ theo token. Báo riêng gốc / lấy mẫu lại tại ρ_px khớp và Δp₀.

**BẢNG III-b. Số câu theo ô ρ_px x mật độ, tách Q1-iso / Q1-agg.** (Ô ρ_px theo bội 2; điều chỉnh sau pilot.)

| ρ_px \ mật độ | Thưa (iso) | Trung bình | Dày (agg) | Tổng |
| ------------- | ---------- | ---------- | --------- | ---- |
| < 2           | —          | —          | —         | —    |
| 2-4           | —          | —          | —         | —    |
| 4-8           | —          | —          | —         | —    |
| 8-16          | —          | —          | —         | —    |
| ≥ 16          | —          | —          | —         | —    |
| **Tổng**      | —          | —          | —         | —    |

### E. Kiểm soát chất lượng

- _Thiếu nhãn._ Kiểm tay 2 000 mẫu Q1/Q2 nghiêng về ρ_px < 8; báo κ; loại lớp có tỷ lệ thiếu > 5 %.
- _Cắt biên._ Loại instance < 50 % trong ô lưới; độ nhạy 30/50/70 %.
- _GSD._ Siêu dữ liệu gốc; VisDrone gắn cờ, độ nhạy ±30 %.
- _Màu._ GT từ ảnh mịn của cặp lấy mẫu lại; kiểm tra người 300 mẫu; loại màu hỗn hợp.
- _Trục định hướng._ Kiểm tra tay 300 mẫu OBB gần 45° (ranh giới bin) và loại instance gần vuông (w/h > 0,8, trục không xác định).
- _Tiên nghiệm ngôn ngữ._ Chạy chỉ văn bản trên hai LLM **theo ô** (mẫu câu x lớp x ô GSD); loại ô vượt mức ngẫu nhiên có ý nghĩa (nhị thức, α = 0,01) _và_ vượt ≥ 15 điểm.

### F. Ngưỡng của người (neo cho p₀^U)

500 câu Q3-HF trên instance cô lập, trải đều 5 ô ρ_px, ba người chú thích có kinh nghiệm ảnh viễn thám, xem ảnh ở độ phân giải gốc với công cụ phóng tự do (loại bỏ nút thắt token của người). Ước lượng p₀^human,50 (50 %) và p₀^human,U (không phân biệt được với ngẫu nhiên) theo từng bậc Johnson. Báo κ giữa người. p₀^U của benchmark := min(p₀^U,models-pilot, p₀^U,human). Đây là neo độc lập mọi mô hình cho tập U: "người có kinh nghiệm, phóng tự do, cũng không xác định được".

### G. So với benchmark hiện có

**BẢNG V. So sánh với benchmark hallucination / từ chối viễn thám.**

| Benchmark      | #QA    | GSD / mẫu | Nhãn ρ_px       | Nhãn ρ_tok theo mô hình | Tập không thể trả lời | Neo người | Tách cảm biến / bộ mã hoá | Tần số / bậc Johnson | Tách kết cục   | Nhãn mật độ |
| -------------- | ------ | --------- | --------------- | ----------------------- | --------------------- | --------- | ------------------------- | -------------------- | -------------- | ----------- |
| RSHBench       | 371    | ✗         | ✗               | ✗                       | ✗                     | ✗         | ✗                         | ✗                    | Type 1/2 (tay) | ✗           |
| RSHalluEval    | 2 023  | ✗         | ✗               | ✗                       | ✗                     | ✗         | ✗                         | ✗                    | ✗              | ✗           |
| CHOICE-HD      | —      | ✗         | ✗               | ✗                       | "không thể phán đoán" | ✗         | ✗                         | ✗                    | ✗              | ✗           |
| UHR-Micro      | 11 253 | ✗         | tỷ lệ diện tích | ✗                       | ✗                     | ✗         | ✗                         | ✗                    | ✗              | ✗           |
| VHM-HnstD      | —      | ✗         | ✗               | ✗                       | không tồn tại         | ✗         | ✗                         | ✗                    | ✗              | ✗           |
| RSVLM-QA       | —      | ✗         | ✗               | ✗                       | ✗                     | ✗         | ✗                         | ✗                    | ✗              | số lượng    |
| **RS-Resolve** | 30-40k | **✓**     | **✓**           | **✓**                   | **✓ (cảm biến)**      | **✓**     | **✓**                     | **✓**                | **✓**          | **✓**       |

---

## V. Thiết lập thực nghiệm

### A. Mô hình

Ba nhóm (Bảng II). **Thí nghiệm biến thiên trần độ phân giải và crop chạy đầy đủ trên bốn mô hình**: Qwen3-VL-8B, InternVL3.5-8B, LLaVA-1.5-7B, GeoChat-7B; các mô hình còn lại (LLaVA-OneVision, SkySenseGPT, EarthDial, VHM, LHRS-Bot-Nova) chạy trên tập con 1 000 instance. Hai mô hình đóng (GPT-5.x, Gemini 3 Pro; phiên bản và ngày truy cập trong Phụ lục) chạy 3 000 câu; tham gia phân tích theo ρ_px (H3, p₀) và đường cong theo crop. ScaleEarth [x] chạy làm điểm tham chiếu "GSD-aware training-based" nếu weights công khai. Mọi mô hình: s_m(I, v) đọc từ bộ tiền xử lý.

### B. Baseline

**A - không gate:** zero-shot; RADAR (zoom attention, không fallback); prompt kiểu RSHallu. **B - từ chối bằng prompt:** mềm / vừa / nghiêm; trung thực kiểu VHM. **C - confidence nội tại:** max-logit; entropy; tự khai; self-consistency k = 5; focus test RADAR. **D - bằng chứng thị giác:** điểm kiểu SIEVES; điểm phát hiện LAE-DINO làm confidence, chỉ trên xView/VisDrone (tập huấn luyện LAE-DINO gồm DOTA, DIOR). **E - zoom:** cắt + phóng quanh vùng attention kiểu ViCrop; MAP-Agent / GeoEyes nếu khả dụng. **Naive:** bảng tra lớp x GSD hiệu chỉnh bằng Thuật toán 1 trên g. **F - đề xuất:** RGSP-abstain (mọi loại câu hỏi; chi phí bằng không); RGSP (định tuyến bốn nhánh, Q3-Q5); RGSP+ (ρ_tok + max-logit; ρ_tok + focus test); RGSP với box GT (oracle). Mọi phương pháp cùng prompt gốc, cùng tuỳ chọn từ chối, cùng tập test; báo số lần gọi thêm và độ trễ.

### C. Giao thức và thống kê

Tách hiệu chỉnh / test **theo ảnh gốc** (mọi crop và mọi trần độ phân giải của một ảnh cùng phía); 20 % ảnh hiệu chỉnh; với RGSP+ nửa khớp / nửa hiệu chỉnh trong 20 % đó. α ∈ {5, 10, 20} %, δ = 0,05. Mọi số liệu: trung bình ± CI 95 % bootstrap 1 000 lần theo ảnh; 3 seed cho phương pháp lấy mẫu. Coverage tại α so bằng paired bootstrap. AUROC so bằng DeLong. Mô hình hỗn hợp (4) khớp bằng glmer với Laplace; CI của β bằng Wald và bootstrap theo cụm ảnh (không dùng profile likelihood). Không báo hallucination rate mà không kèm coverage; coverage luôn trên tập A.

### D. Tiêu chí dừng đăng ký trước

Pilot 2 tuần: 800 instance cô lập trên DOTA-v2 + xView, crop cố định 1 024², **mọi trần độ phân giải** của ba mô hình (Qwen3-VL-8B, LLaVA-1.5, GeoChat), câu hỏi Q1-iso, Q3-HF (trục 2 bin, lớp), Q3-LF; song song 300 câu cho người (Mục IV-F).

- **K1:** β₁ của log ρ_tok trong (4) không âm có ý nghĩa ở ≥ 2/3 mô hình → nút thắt token không tồn tại hoặc quá yếu; dừng ở benchmark + phân tích Johnson cho MLLM (vẫn là đóng góp, hạ venue).
- **K2:** không đủ 300 câu ở ô lệch đường chéo sau lấy mẫu lại → không tách mật độ-kích thước; bỏ claim đó.
- **K3:** mô hình tự từ chối ≥ 70 % ở ρ_px < p₀^U với prompt trung tính → vấn đề nhỏ hơn dự đoán; hạ trọng số H4-a.
- **K4:** Q3-LF không suy giảm chậm hơn Q3-HF theo ρ_px → phân tầng tần số không có cơ sở; gộp lại và bỏ H3.
- **K5:** p₀^U của người và của mô hình pilot lệch nhau > 2 lần → neo người không dùng được; dùng p₀^U,models và ghi rõ.
- **Go:** β₁ < 0 có ý nghĩa ở ≥ 2/3 mô hình, thứ tự LF < HF tái hiện, tự từ chối < 30 %, p₀^U người-mô hình cùng bậc.

Kết quả pilot: `—`.

---

## VI. Kết quả

_Diễn giải dưới đây là khung điều kiện; phải viết lại theo số thực, và đổi kết luận nếu số đi ngược kỳ vọng._

### A. Nút thắt token tồn tại độc lập với nút thắt pixel (H1)

**BẢNG VI. Tỷ lệ kết cục (%) trên tập biến thiên trần độ phân giải theo ô ρ_px (hàng) x ρ_tok (cột), Q1-iso + Q3-HF.** Mỗi ô: hallucination / lỗi nhận thức / từ chối. Cùng ảnh xuất hiện ở nhiều cột (các trần khác nhau) nhưng chỉ một hàng.

| Mô hình                                                           | ρ_px \ ρ_tok         | < 0,5     | 0,5-1     | 1-2       | 2-4       | ≥ 4       |
| ----------------------------------------------------------------- | -------------------- | --------- | --------- | --------- | --------- | --------- |
| Qwen3-VL-8B                                                       | < p₀^U               | — / — / — | — / — / — | — / — / — | — / — / — | — / — / — |
|                                                                   | p₀^U-2p₀^U           | —         | —         | —         | —         | —         |
|                                                                   | 2p₀^U-4p₀^U          | —         | —         | —         | —         | —         |
|                                                                   | ≥ 4p₀^U              | —         | —         | —         | —         | —         |
| InternVL3.5-8B                                                    | (4 hàng)             | —         | —         | —         | —         | —         |
| LLaVA-1.5-7B                                                      | (4 hàng)             | —         | —         | —         | —         | —         |
| GeoChat                                                           | (4 hàng)             | —         | —         | —         | —         | —         |
| Tập con: LLaVA-OV / SkySenseGPT / EarthDial / VHM / LHRS-Bot-Nova | (4 hàng mỗi mô hình) | —         | —         | —         | —         | —         |

Nếu H1 đúng, đọc **theo hàng** (cùng ảnh), thất bại giảm khi ρ*tok tăng — hiệu ứng thuần của bộ mã hoá; đọc **theo cột**, thất bại vẫn giảm khi ρ_px tăng — hiệu ứng thuần của cảm biến. Hàng ρ_px < p₀^U kỳ vọng phẳng theo cột: khi thông tin không có trong ảnh, thêm token không giúp. Tỷ lệ từ chối kỳ vọng \_không* tăng ở hàng đầu — mô hình đoán thay vì từ chối.

**BẢNG VII. Mô hình hỗn hợp (4): hệ số log ρ_tok sau khi hấp thụ mọi đặc tính cố định của ảnh bằng hiệu ứng ngẫu nhiên.**

| Mô hình             | Thiết kế          | β₁ (log ρ_tok) | CI 95 % (Wald / bootstrap cụm) | p   | Var(u_i) | ICC | ΔAUC nội-instance |
| ------------------- | ----------------- | -------------- | ------------------------------ | --- | -------- | --- | ----------------- |
| Qwen3-VL-8B         | trần độ phân giải | —              | —                              | —   | —        | —   | —                 |
| Qwen3-VL-8B         | crop (thiết kế 2) | —              | —                              | —   | —        | —   | —                 |
| InternVL3.5-8B      | trần / crop       | —              | —                              | —   | —        | —   | —                 |
| LLaVA-1.5-7B        | trần / crop       | —              | —                              | —   | —        | —   | —                 |
| GeoChat             | trần / crop       | —              | —                              | —   | —        | —   | —                 |
| Tập con (5 mô hình) | trần              | —              | —                              | —   | —        | —   | —                 |

β₁ < 0 có ý nghĩa ở thiết kế trần độ phân giải là bằng chứng cho H1 không có biến gây nhiễu. Sự khớp giữa hai dòng trần / crop cho biết zoom thực tế (đổi ngữ cảnh) có tái hiện hiệu ứng token thuần hay không.

### B. Hai ngưỡng: cảm biến gần độc lập mô hình và gần người, bộ mã hoá ổn định theo token (H2)

Ước lượng p₀^50 dưới **điều kiện không bị nút thắt token**: trần độ phân giải cao nhất, ρ_tok ≥ 4, Q3-HF trên instance cô lập, hồi quy theo log ρ_px. Ước lượng ρ\*\_m dưới **điều kiện không bị nút thắt pixel**: ρ_px ≥ 2p₀^U, hồi quy theo log ρ_tok.

**BẢNG VIII. Ngưỡng cảm biến p₀ và ngưỡng bộ mã hoá ρ\*\_tok.**

| Mô hình                                                                                                  | P (px) | p₀^50 (px gốc) ± CI | p₀^U (px gốc) | ρ\*\_tok ± CI | ρ\*\_tok từ n = 300 | Ngưỡng theo px mô hình (ρ\*·P) | Δp₀ gốc − lấy mẫu lại | Δρ\* xuyên bộ dữ liệu |
| -------------------------------------------------------------------------------------------------------- | ------ | ------------------- | ------------- | ------------- | ------------------- | ------------------------------ | --------------------- | --------------------- |
| **Người (phóng tự do)**                                                                                  | —      | —                   | —             | n/a           | n/a                 | n/a                            | —                     | —                     |
| LLaVA-1.5-7B                                                                                             | 14     | —                   | —             | —             | —                   | —                              | —                     | —                     |
| GeoChat                                                                                                  | 14     | —                   | —             | —             | —                   | —                              | —                     | —                     |
| LLaVA-OneVision-7B                                                                                       | 14     | —                   | —             | —             | —                   | —                              | —                     | —                     |
| InternVL3.5-8B                                                                                           | 28     | —                   | —             | —             | —                   | —                              | —                     | —                     |
| Qwen3-VL-8B                                                                                              | 32     | —                   | —             | —             | —                   | —                              | —                     | —                     |
| SkySenseGPT / EarthDial / VHM / LHRS-Bot-Nova                                                            | —      | —                   | —             | —             | —                   | —                              | —                     | —                     |
| GPT-5.x / Gemini 3 Pro ‡                                                                                 | n/a    | —                   | —             | n/a           | n/a                 | n/a                            | —                     | —                     |
| _Hệ số biến thiên xuyên mô hình: p₀^50_                                                                  |        | —                   |               |               |                     |                                |                       |                       |
| _Hệ số biến thiên xuyên mô hình: ρ\*\_tok_                                                               |        |                     |               | —             |                     |                                |                       |                       |
| _Hệ số biến thiên xuyên mô hình: ngưỡng GSD theo lớp (Naive)_                                            |        |                     |               |               |                     | —                              |                       |                       |
| _So với Johnson (chu kỳ → px): detection ≈ 2, orientation ≈ 2,8, recognition ≈ 8, identification ≈ 12,8_ |        | —                   |               |               |                     |                                |                       |                       |

H2 được ủng hộ nếu (i) p₀^50 có hệ số biến thiên nhỏ xuyên mô hình, kể cả mô hình đóng, và cùng bậc với người và với Johnson; (ii) ρ*\_tok có hệ số biến thiên nhỏ hơn hẳn ngưỡng GSD theo lớp của Naive; (iii) ngưỡng theo px mô hình tỷ lệ với P. Nếu ρ*\_tok phân tán mà ngưỡng GSD ổn định, chuẩn hoá theo patch không thêm gì và H2 bị bác. Dòng cuối là điểm nối trực tiếp với văn liệu viễn thám: MLLM có tuân tiêu chí Johnson hay không, và lệch về phía nào.

### C. Thứ tự suy giảm theo tần số không gian và bậc thang Johnson (H3)

**BẢNG IX. Ngưỡng 50 % theo ρ_px của từng loại thuộc tính, dưới điều kiện không bị nút thắt token.**

| Mô hình        | Q3-LF màu | Q1-iso tồn tại (detection) | Trục 2 bin (orientation) | Trục 4 bin | Lớp (recognition) | Loại con xView (identification) | Kiểm định thứ tự đơn điệu (p, Jonckheere) |
| -------------- | --------- | -------------------------- | ------------------------ | ---------- | ----------------- | ------------------------------- | ----------------------------------------- |
| Người          | —         | —                          | —                        | —          | —                 | —                               | —                                         |
| Qwen3-VL-8B    | —         | —                          | —                        | —          | —                 | —                               | —                                         |
| InternVL3.5-8B | —         | —                          | —                        | —          | —                 | —                               | —                                         |
| LLaVA-1.5-7B   | —         | —                          | —                        | —          | —                 | —                               | —                                         |
| GeoChat        | —         | —                          | —                        | —          | —                 | —                               | —                                         |
| GPT-5.x ‡      | —         | —                          | —                        | —          | —                 | —                               | —                                         |
| Gemini 3 Pro ‡ | —         | —                          | —                        | —          | —                 | —                               | —                                         |

Nếu ρ_px đúng là đại lượng phân giải, ngưỡng tăng từ trái sang phải. Kiểm định rẻ nhưng khó giả: một biến chỉ hấp thụ GSD hay lớp sẽ không tái tạo thứ tự này. Kết quả trên mô hình đóng và trên người đặc biệt có giá trị vì không cần biết P.

### D. Dự đoán có chọn lọc (H4-a)

**BẢNG X. Coverage trên tập A tại selective risk cố định, cấu hình triển khai (L_c + GSD, không box GT), Qwen3-VL-8B.** Rủi ro thực @10 % phải ≤ 10. Các mô hình khác: Bảng S3.

| Nhóm         | Phương pháp                       | Cần nội bộ | Gọi thêm / câu (kỳ vọng) | Cov@5 % ↑ | Cov@10 % ↑ | Cov@20 % ↑ | Rủi ro thực @10 % | AURC ↓ | Abst.-acc trên U ↑ | Từ chối thừa trên A ↓ | Selective acc. tổng ↑ |
| ------------ | --------------------------------- | ---------- | ------------------------ | --------- | ---------- | ---------- | ----------------- | ------ | ------------------ | --------------------- | --------------------- |
| A            | Zero-shot                         | —          | 0                        | 100       | 100        | 100        | (gốc)             | —      | —                  | 0                     | —                     |
| A            | RADAR (chỉ zoom)                  | ✓          | 1                        | 100       | 100        | 100        | —                 | —      | —                  | 0                     | —                     |
| A            | Prompt kiểu RSHallu               | ✗          | 0                        | 100       | 100        | 100        | —                 | —      | —                  | 0                     | —                     |
| B            | Prompt từ chối mềm / vừa / nghiêm | ✗          | 0                        | —         | —          | —          | —                 | —      | —                  | —                     | —                     |
| B            | Trung thực kiểu VHM               | ✗          | 0                        | —         | —          | —          | —                 | —      | —                  | —                     | —                     |
| C            | Max-logit                         | ✓          | 0                        | —         | —          | —          | —                 | —      | —                  | —                     | —                     |
| C            | Entropy                           | ✓          | 0                        | —         | —          | —          | —                 | —      | —                  | —                     | —                     |
| C            | Tự khai                           | ✗          | 0                        | —         | —          | —          | —                 | —      | —                  | —                     | —                     |
| C            | Self-consistency k = 5            | ✗          | 4                        | —         | —          | —          | —                 | —      | —                  | —                     | —                     |
| C            | Focus test RADAR                  | ✓          | 1                        | —         | —          | —          | —                 | —      | —                  | —                     | —                     |
| D            | Điểm kiểu SIEVES                  | ✗          | 1-2                      | —         | —          | —          | —                 | —      | —                  | —                     | —                     |
| D            | LAE-DINO ‡                        | ✗          | 1                        | —         | —          | —          | —                 | —      | —                  | —                     | —                     |
| Ref          | ScaleEarth (nếu có)               | —          | 0                        | 100       | 100        | 100        | —                 | —      | —                  | 0                     | —                     |
| **Naive**    | **Bảng tra lớp x GSD**            | ✗          | 0                        | —         | —          | —          | —                 | —      | —                  | —                     | —                     |
| **F**        | **RGSP-abstain (ρ_px, ρ_tok)**    | **✗**      | **0**                    | **—**     | **—**      | **—**      | **—**             | **—**  | **—**              | **—**                 | **—**                 |
| **F**        | **RGSP (bốn nhánh, Q3-Q5)**       | ✗          | **—** (chỉ nhánh zoom)   | **—**     | **—**      | **—**      | **—**             | **—**  | **—**              | **—**                 | **—**                 |
| **F**        | **RGSP+ (ρ_tok + max-logit)**     | ✓          | 0                        | —         | —          | —          | —                 | —      | —                  | —                     | —                     |
| **F**        | **RGSP+ (ρ_tok + focus test)**    | ✓          | 1                        | —         | —          | —          | —                 | —      | —                  | —                     | —                     |
| _F (oracle)_ | _RGSP với box GT †_               | ✗          | 0                        | _—_       | _—_        | _—_        | _—_               | _—_    | _—_                | _—_                   | _—_                   |

Ba so sánh quyết định: RGSP-abstain với (i) confidence nội tại tốt nhất, (ii) điểm bằng chứng kiểu SIEVES, (iii) bảng tra lớp x GSD. Không vượt (iii) → đóng góp thu về quy tắc GSD theo lớp, bài phải định vị lại thành chẩn đoán + benchmark + tiêu chí Johnson cho MLLM. Vượt (iii) nhưng không vượt (i) → ρ _bổ sung_ tín hiệu mô hình; giá trị nằm ở RGSP+ và chi phí bằng không. RGSP bốn nhánh kỳ vọng coverage cao hơn RGSP-abstain trên Q3-Q5 nhờ chuyển nhóm token-limited có thể zoom từ từ chối sang zoom, với chi phí một lần gọi trên đúng nhóm đó; nhóm "zoom không đủ" vẫn bị từ chối nên rủi ro được giữ.

### E. Zoom giúp đúng một ô (H4-b)

**BẢNG XI. Δ thất bại so với zero-shot (điểm phần trăm, âm = tốt) theo ma trận (ρ_px, ρ_tok trước zoom), Qwen3-VL-8B, Q3-HF (có vị trí).**

| Phương pháp                         | (ρ_px < p₀^U, ρ_tok < ρ\*) pixel-limited | (ρ_px < p₀^U, ρ_tok ≥ ρ\*) | (ρ_px ≥ p₀^U, ρ_tok < ρ\* ≤ ρ_tok^max) **token-limited, zoom đủ** | (ρ_px ≥ p₀^U, ρ_tok^max < ρ\*) **zoom không đủ** | (ρ_px ≥ p₀^U, ρ_tok ≥ ρ\*) resolvable | Gọi thêm / câu |
| ----------------------------------- | ---------------------------------------- | -------------------------- | ----------------------------------------------------------------- | ------------------------------------------------ | ------------------------------------- | -------------- |
| Zero-shot                           | 0                                        | 0                          | 0                                                                 | 0                                                | 0                                     | 0              |
| RADAR (zoom attention)              | —                                        | —                          | —                                                                 | —                                                | —                                     | 1              |
| Cắt + phóng kiểu ViCrop (s_max = 2) | —                                        | —                          | —                                                                 | —                                                | —                                     | 1              |
| MAP-Agent / GeoEyes                 | —                                        | —                          | —                                                                 | —                                                | —                                     | ~3-4           |
| RGSP-abstain                        | _từ chối_                                | _từ chối_                  | _từ chối_                                                         | _từ chối_                                        | 0                                     | 0              |
| **RGSP (s_max = 1)**                | _từ chối_                                | _từ chối_                  | **— (zoom về c\*)**                                               | _từ chối_                                        | 0                                     | —              |
| **RGSP (s_max = 2)**                | _từ chối_                                | _từ chối_                  | **— (zoom về c\*)**                                               | **— (nhóm này thu hẹp)**                         | 0                                     | —              |

Dự đoán: mọi phương pháp zoom cho Δ âm có ý nghĩa **ở cột 3**; cột 1 không khác 0 (giới hạn Johnson); cột 2 (nhiều token cho ít pixel — ảnh đã được phóng) kỳ vọng Δ **nhỏ, không nhất thiết bằng 0**, vì upsampling không thêm thông tin nhưng ViT vẫn có thể được lợi khi vật thể trải nhiều patch — chính là hiệu ứng ViCrop khai thác; độ lớn của Δ ở cột 2 so với cột 3 đo phần lợi "cơ học" đó. Cột 4 kiểm tra nhánh từ chối mới của (3): nếu zoom vẫn giúp đáng kể ở đây, s_max quá bảo thủ và nên nới. Cột 5 gần 0 (còn lại là lỗi attention). Nếu zoom giảm thất bại ở cột 1, hoặc p₀^U ước lượng quá cao, hoặc mô hình khai thác ngữ cảnh mà ρ_px không nắm — phải báo và thảo luận.

### F. Ablation

**BẢNG XII. Ablation RGSP (Qwen3-VL-8B, α = 10 %).**

| Biến thể                                   | Nguồn ρ    | Chọn τ                 | Cov@10 % | Rủi ro thực | AURC | Gọi thêm | Ghi chú                    |
| ------------------------------------------ | ---------- | ---------------------- | -------- | ----------- | ---- | -------- | -------------------------- |
| RGSP-abstain (mặc định)                    | L_c + GSD  | Thuật toán 1, toàn cục | —        | —           | —    | 0        | triển khai                 |
| RGSP bốn nhánh, s_max = 1                  | L_c + GSD  | Thuật toán 1 toàn hệ   | —        | —           | —    | —        | với zoom                   |
| RGSP bốn nhánh, s_max = 2                  | L_c + GSD  | Thuật toán 1 toàn hệ   | —        | —           | —    | —        | cho phép phóng             |
| RGSP ba nhánh (bỏ từ chối "zoom không đủ") | L_c + GSD  | Thuật toán 1           | —        | —           | —    | —        | kỳ vọng rủi ro vượt α      |
| chỉ gate ρ_px (bỏ ρ_tok)                   | L_c + GSD  | Thuật toán 1           | —        | —           | —    | 0        | nút thắt cảm biến đủ chưa? |
| chỉ gate ρ_tok (bỏ p₀^U)                   | L_c + GSD  | Thuật toán 1           | —        | —           | —    | 0        |                            |
| τ cố định = ρ\* từ chẩn đoán               | —          | không hiệu chỉnh       | —        | —           | —    | 0        | giá trị của hiệu chỉnh     |
| τ theo point estimate (không bound)        | —          | El-Yaniv không bảo đảm | —        | —           | —    | 0        | giá coverage của bound     |
| Mondrian theo ô GSD                        | —          | Thuật toán 1 theo ô    | —        | —           | —    | 0        | dịch chuyển                |
| p₀^U ± 1 ô                                 | —          | Thuật toán 1           | —        | —           | —    | 0        | độ nhạy định nghĩa U       |
| ρ từ box GT †                              | box        | Thuật toán 1           | —        | —           | —    | 0        | trần trên                  |
| ρ từ mask iSAID †                          | √diện tích | Thuật toán 1           | —        | —           | —    | 0        | lớp thuôn dài              |
| L = max(w,h) †                             | box        | Thuật toán 1           | —        | —           | —    | 0        | hình dạng                  |
| ρ với GSD ước lượng từ ảnh                 | kiểu SSE-U | Thuật toán 1           | —        | —           | —    | 0        | thiếu siêu dữ liệu         |
| ρ_tok từ công thức thay lưới thực          | L_c + GSD  | Thuật toán 1           | —        | —           | —    | 0        | công thức đủ tốt không?    |
| Chỉ logit                                  | —          | Thuật toán 1           | —        | —           | —    | 0        | = baseline C               |

### G. Chuyển giao xuyên bộ dữ liệu, cảm biến và trần độ phân giải

**BẢNG XIII. Rủi ro thực và coverage khi hiệu chỉnh trên một điều kiện, test trên điều kiện khác (α = 10 %, Qwen3-VL-8B, RGSP-abstain).**

| Hiệu chỉnh → Test                    | Rủi ro thực (toàn cục) | Rủi ro thực (Mondrian) | Cov@10 % (toàn cục) | Cov@10 % (Mondrian) | Chồng lấn GSD (Jaccard) |
| ------------------------------------ | ---------------------- | ---------------------- | ------------------- | ------------------- | ----------------------- |
| DOTA → DOTA                          | —                      | —                      | —                   | —                   | 1,0                     |
| DOTA → DIOR                          | —                      | —                      | —                   | —                   | —                       |
| DOTA → xView                         | —                      | —                      | —                   | —                   | —                       |
| xView → DOTA                         | —                      | —                      | —                   | —                   | —                       |
| DOTA+DIOR → VisDrone (GSD ước lượng) | —                      | —                      | —                   | —                   | —                       |
| Gốc → lấy mẫu lại (ρ_px khớp)        | —                      | —                      | —                   | —                   | —                       |
| Trần 1 024² → trần {512², 2 048²}    | —                      | —                      | —                   | —                   | —                       |
| Crop 1 024 → crop {512, 1 536}       | —                      | —                      | —                   | —                   | —                       |

Hai dòng cuối kiểm tra τ có chuyển được giữa các điều kiện đầu vào — điều kiện cần để RGSP dùng được khi kích thước ảnh hoặc trần độ phân giải lúc triển khai khác lúc hiệu chỉnh; nếu ρ_tok là đại lượng đúng, τ theo token phải chuyển được trong khi ngưỡng theo px mô hình thì không.

### H. Đánh giá người và LLM-judge

Ngoài 500 câu cho ngưỡng người (IV-F), hai người chấm 500 câu mở phân tầng theo loại và ρ_px; báo κ người-người và κ người-judge theo ô. Judge khác họ với mô hình bị đánh giá. Ô có κ < 0,6 bị đánh dấu; kết luận chính chỉ dùng trắc nghiệm.

### I. Chi phí

Ước tính trước thực nghiệm: chẩn đoán cơ bản 40 nghìn câu x 8 mô hình ≈ 130-400 GPU-giờ; tập biến thiên trần độ phân giải 4 000 instance x ~4 trần x ~4 câu x 4 mô hình ≈ 250 nghìn suy luận ≈ 150-400 GPU-giờ; biến thiên crop tương tự; self-consistency x5 và zoom nhiều bước x3-4 trên hai mô hình đại diện; tổng dự kiến 1 000-1 800 GPU-giờ trên A6000/L40S. Người chú thích: 500 (ngưỡng người) + 500 (mở) + 2 000 (QC) + 600 (màu, trục) câu. Ngân sách xác nhận trước pilot.

---

## VII. Thảo luận

**Johnson cho MLLM.** Tiêu chí Johnson đo người và detector; chúng tôi đo MLLM và thấy [nếu H2-H3 đúng] chúng tuân cùng bậc thang nhiệm vụ với ngưỡng [cao hơn / thấp hơn] người `—` lần. Đây là điểm nối tự nhiên giữa hai cộng đồng: kỹ sư viễn thám có thể dự đoán MLLM thấy gì từ GSD và bảng lớp, như đã làm với NIIRS suốt ba thập niên; và cộng đồng MLLM có một chuẩn vật lý để so bộ mã hoá.

**Hai đại lượng bậc nhất, không phải đại lượng đủ.** ρ_px và ρ_tok chỉ đo kích thước so với pixel và so với token. GIQE có MTF và SNR; chúng tôi chưa có. Phần dư trong (4) và trong đường cong theo ρ_px là chỗ cho tiên nghiệm bậc hai.

**Chế độ attention-limited nằm ngoài khung.** Khi cả hai footprint đủ, thất bại còn lại do mô hình chú ý sai — không dự đoán được từ siêu dữ liệu. RGSP không claim xử lý chế độ này; khoảng cách RGSP+ − RGSP ước lượng phần lỗi thuộc chế độ này.

**Nhánh "zoom không đủ" là điều làm bảo đảm có ý nghĩa.** Không có nó, bộ định tuyến trả lời mọi câu ρ_px ≥ p₀^U và τ không kiểm soát được rủi ro; có nó, coverage được đổi lấy rủi ro đúng như selective prediction đòi hỏi, và s_max trở thành tham số vận hành: cho phép phóng bao nhiêu thì coverage tăng bao nhiêu, rủi ro trả giá bao nhiêu (Bảng XI, XII).

**Vị trí vật thể.** Định tuyến zoom xác định cần vị trí trong câu hỏi. Đây là giới hạn thật của claim "không cần chạy mô hình": với câu hỏi tồn tại, hoặc từ chối (RGSP-abstain), hoặc quét tile với chi phí tính trước, hoặc dùng attention và mất claim. Chúng tôi báo cả ba.

**Rò rỉ tiên nghiệm lớp.** L_c là hằng theo lớp; baseline Naive có cùng thông tin lớp nhưng không chuẩn hoá theo bộ mã hoá; chênh lệch RGSP − Naive là phần quy được cho kiến trúc.

**Mật độ có hai mặt.** Kết cấu tổng hợp giúp nhận ra _sự tồn tại_ dưới p₀ theo instance; chen chúc hại đếm và tham chiếu. Mọi phân tích mật độ tách theo loại câu hỏi.

**Điều RGSP không làm.** Không giảm lỗi ở ô resolvable. Nó trả lời câu hỏi hẹp: _trước khi tốn một lần suy luận, có nên trả lời, nên nhìn kỹ hơn bao nhiêu, hay nên từ chối_ — bằng hai con số từ siêu dữ liệu và một tham số hiệu chỉnh.

---

## VIII. Kết luận

Chúng tôi tách "cannot see clearly" thành hai nút thắt tính trước được — cảm biến (ρ_px, đại lượng của Johnson) và bộ mã hoá (ρ_tok, đại lượng Johnson không có) — và cho thấy [nếu H1-H4 đúng] nút thắt token tồn tại độc lập với nút thắt pixel trong thí nghiệm giữ ảnh cố định và chỉ đổi trần độ phân giải của mô hình; ngưỡng cảm biến gần độc lập mô hình và cùng bậc với người; ngưỡng bộ mã hoá ổn định theo token; thuộc tính suy giảm theo bậc thang Johnson; zoom chỉ giúp ở ô token-limited và chỉ khi đủ; và bộ định tuyến bốn nhánh theo hai đại lượng này đạt coverage `—` tại selective risk 10 % so với `—` của confidence nội tại tốt nhất và `—` của bảng tra lớp x GSD, với `—` lần gọi bổ sung trung bình. RS-Resolve, nhãn ρ, ngưỡng người và script được công khai.

---

## Tài liệu tham khảo

_(Kiểm chứng đến 04/09/2026 trừ mục ghi [cần xác minh].)_

1. [xác minh danh sách tác giả; corresponding: J. Zhang, D. Wang, B. Du] "Seeing Clearly without Training: Mitigating Hallucinations in MLLMs for Remote Sensing" (RADAR/RSHBench), arXiv:2603.02754, 2026.
2. "RSHallu," arXiv:2602.10799, 2026.
3. F. Wang et al., "XLRS-Bench," CVPR 2025; arXiv:2503.23771.
4. Y. Dang et al., "RSHR-Bench," arXiv:2512.17319, 2025.
5. "CHOICE," NeurIPS D&B 2025; arXiv:2411.18145.
6. "VHM," AAAI 2025; arXiv:2403.20213.
7. "UHR-Micro / MAP-Agent," arXiv:2605.12237, 2026. [cần xác minh]
8. "ScaleEarth," arXiv:2605.07562, 2026.
9. "GeoEyes," arXiv:2602.14201, 2026.
10. "GeoLLaVA-8K," arXiv:2505.21375, 2025.
11. "LRS-VQA," ICCV 2025.
12. "UHR-BAT," arXiv:2604.13565, 2026.
13. "UAV-MAS," arXiv:2608.11738, 2026.
14. A. Kanade, T. Ganu, "Do You See Me," EACL 2026; arXiv:2506.02022.
15. "Exploring Perceptual Limitation of MLLMs," arXiv:2402.07384. [cần xác minh]
16. "ViCrop," ICLR 2025. [cần xác minh]
17. "SIEVES," arXiv:2604.25855, 2026.
18. "FUSE," arXiv:2606.14728, 2026.
19. "Variational VQA," arXiv:2505.09591, 2025.
20. "OATS-RS," Remote Sensing, doi:10.3390/rs18122038, 2026.
21. "TUBench," arXiv:2410.04107.
22. "HaloQuest," arXiv:2407.15680, 2024.
23. "MoHoBench," arXiv:2507.21503, 2025. [cần xác minh]
24. "MM-AQA," arXiv:2604.14799; "UA-Bench," arXiv:2604.17293. [cần xác minh]
25. "From Pixels to Tokens," arXiv:2410.06795, 2024.
26. "RSVLM-QA," arXiv:2508.07918, 2025.
27. "DDFAV," Remote Sensing 17(4):719, 2025.
28. Y. Geifman, R. El-Yaniv, "Selective Classification for Deep Neural Networks," NeurIPS 2017.
29. A. N. Angelopoulos, S. Bates, E. J. Candès, M. I. Jordan, L. Lei, "Learn then Test," arXiv:2110.01052, 2021.
30. A. N. Angelopoulos et al., "Conformal Risk Control," ICLR 2024.
31. C. J. Clopper, E. S. Pearson, Biometrika 1934.
32. **J. Johnson, "Analysis of Image Forming Systems," Proc. Image Intensifier Symposium, 1958.** [tiêu chí Johnson]
33. **J. C. Leachtenauer, W. Malila, J. Irvine, L. Colburn, N. Salvaggio, "General Image-Quality Equation: GIQE," Applied Optics 36(32), 1997.** [NIIRS/GIQE]
34. **R. G. Driggers, P. Cox, T. Edwards, _Introduction to Infrared and Electro-Optical Systems_, Artech House.** [Johnson cập nhật; hàm nhiệm vụ] [cần xác minh ấn bản]
35. N. Keshava, J. F. Mustard, "Spectral Unmixing," IEEE Signal Processing Magazine 2002.
36. [RSSOD / small-object detection in RS — trích bài có phép tính "xe 4x1,5 m ở 0,5 m/px = 8x3 px"] [cần xác minh]
37. Qwen Team, "Qwen3-VL Technical Report," 2025. [cfg]
38. "InternVL3.5," 2025. [cfg]
39. B. Li et al., "LLaVA-OneVision," 2024.
40. H. Liu et al., "Improved Baselines with Visual Instruction Tuning," CVPR 2024.
41. K. Kuckreja et al., "GeoChat," CVPR 2024.
42. SkySenseGPT; EarthDial; LHRS-Bot-Nova. [trích từng bài]
43. J. Ding et al., "Object Detection in Aerial Images: A Large-Scale Benchmark and Challenges" (DOTA-v2.0), TPAMI 2021.
44. K. Li et al., "DIOR," ISPRS JPRS 2020.
45. D. Lam et al., "xView," arXiv:1802.07856, 2018.
46. P. Zhu et al., "VisDrone," TPAMI 2021.
47. S. Waqas Zamir et al., "iSAID," CVPRW 2019.
48. LAE-DINO. [trích bài + tập huấn luyện]
49. E. DeLong et al., Biometrics 1988.
50. D. Bates et al., "lme4," J. Stat. Softw. 2015.
51. A. R. Jonckheere, "A distribution-free k-sample test against ordered alternatives," Biometrika 1954. [kiểm định thứ tự H3]

---

## Phụ lục (danh mục)

- **A.** Bảng L_c^min và L_c^typ theo lớp với nguồn (thông số kỹ thuật, ICAO, xView metadata).
- **B.** Cách ghi lưới token thực từ bộ tiền xử lý của từng mô hình ở từng trần độ phân giải; đối chiếu công thức s_m(I); cách tính c\*(τ) từ chính sách tiền xử lý.
- **C.** Prompt đầy đủ (S10); tuỳ chọn từ chối trung tính.
- **D.** Biến thể tiling xác định cho Q1-Q2: quét tile cỡ c\*, chi phí = số tile, kết quả và so với RGSP-abstain.
- **E.** Giao thức ngưỡng người (IV-F): công cụ, hướng dẫn, κ.
- **F.** Bảng S1-S13.
- **G.** Phiên bản API mô hình đóng, ngày truy cập; khai báo sử dụng công cụ AI tạo sinh trong soạn thảo theo chính sách IEEE.
