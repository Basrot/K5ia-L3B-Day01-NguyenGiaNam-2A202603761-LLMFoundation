# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV: 2A202603761
- **Họ và tên: Nguyễn Gia Nam
- - **SEED** (in ở ô đầu notebook): 8113

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☐ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☐ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☐ |
Dữ liệu sau loại bản ghi trùng tuyệt đối: **5.169 tin** (653 spam, 4.516 ham). Chia tập: train **3.619**, validation **775**, test **775** (91 spam); tập SUB gồm **40** tin. Lưu ý: kết quả 9/9 là các kiểm tra chức năng chính trong JSON, không đồng nghĩa mọi ô và phần tính tay đã được xác nhận.
## 2. Kết quả chính

*Chép bảng so sánh cuối Phần 5 (F1 test, F1 SUB 40, p95 ms/tin, USD/1k tin) của chính bạn.*


| Mô hình | F1 test | F1 SUB 40 | p95 (ms/tin) | USD/1.000 tin |
| --- | ---: | ---: | ---: | ---: |
| BoW + Logistic Regression | 0.8696 | 0.9474 | 1.27 | 0.00000 |
| BoW + MLP | 0.9070 | 0.9474 | 5.30 | 0.00000 |
| MiniLM + kNN | 0.9198 | 1.0000 | 63.65 | 0.00000 |
| MiniLM + Logistic Regression | 0.9222 | 0.9744 | 45.13 | 0.00000 |
| MiniLM + MLP head | **0.9432** | 0.9744 | 55.81 | 0.00000 |
| Gemini 3.5 Flash Lite, zero-shot | — | 0.8889 | 1080.80 | 0.01862 |

## 3. Dự đoán và thực tế

*Chọn 3 dự đoán `predict(...)` lệch nhiều nhất so với kết quả. Mỗi cái: bạn đoán gì, ra gì, vì sao lệch.*
**(1) F1 của BoW + Logistic Regression.** Em dự đoán F1 khoảng **0.85**. Kết quả test **0.8696**, cao hơn dự đoán **0.0196**. Với văn bản SMS ngắn, các từ đặc trưng như `call`, `txt`, `free` giúp mô hình tuyến tính nhận diện spam khá hiệu quả. Tuy nhiên recall chỉ **0.7692**, cho thấy vẫn bỏ sót spam.

**(2) Tác dụng của ReLU trong MLP.** Em dự đoán bỏ ReLU làm F1 giảm về khoảng **0.89**. Kết quả mô hình **không ReLU đạt F1 = 0.913**, trong khi MLP có ReLU đạt **0.907**; tức mô hình không ReLU **cao hơn 0.006** trong lần chạy này. Điều này trái dự đoán, nhưng không chứng minh ReLU vô ích: dữ liệu BoW khá phù hợp với ranh giới tuyến tính, và một lần chạy có thể chịu ảnh hưởng của cấu hình huấn luyện.
**(3) Embedding trên 6 câu dễ gây nhầm.** Em dự đoán Encoder + LogReg sẽ đúng nhiều hơn BoW + LogReg. Kết quả thực tế **cả hai đúng 5/6**, cùng bỏ sót câu “Last chance! Your account rewards expire tonight, confirm details to keep them”. Embedding giúp biểu diễn ngữ nghĩa nhưng không bảo đảm phát hiện được mọi kiểu spam; việc chọn đầu phân loại và dữ liệu huấn luyện vẫn quan trọng.

Ngoài ra, em dự đoán MLP đạt F1 khoảng **0.90–0.92**; thực tế **0.907**, nằm trong khoảng dự đoán. Em cũng dự đoán LLM lớn vượt MiniLM + LogReg, nhưng **chưa có số liệu LLM lớn trong JSON** để kết luận.

## 4. Phân tích 50 case khó

*Model nào tụt nhiều nhất, ở nhóm case nào? Nêu 2 case cụ thể và cơ chế khiến model sai.*
Notebook có phần trả lời phân tích stress test, nhưng các ô chạy stress test **không còn output trong tệp `.ipynb`** và JSON **không chứa bảng kết quả theo case**. Vì vậy những số liệu dưới đây là **số liệu em đã ghi trong phần trả lời của notebook, chưa đối chiếu được với output thực thi**:

- BoW + LogReg: em ghi F1 trên bộ khó **0.333**, accuracy **52%**.
- BoW + MLP: em ghi F1 trên bộ khó **0.462**, accuracy **58%**.
- LLM zero-shot và prompt phòng thủ: em ghi **50/50 case đúng**; chưa có output lưu lại để xác nhận.

So với **F1 test thường** trong JSON, BoW + LogReg giảm từ **0.870 xuống 0.333**, tương ứng **0.537**; BoW + MLP giảm từ **0.907 xuống 0.462**, tương ứng **0.445**. Như vậy **trong hai mô hình BoW có số liệu ghi lại, BoW + LogReg tụt nhiều hơn**. Không thể khẳng định model tụt nhiều nhất trong tất cả các model khi chưa có đủ kết quả stress test. Cũng cần tránh so sánh F1 SUB 40 với F1 50 case khó như thể đó là cùng một tập đánh giá.

**Case cụ thể 1 — ham chứa từ khóa spam:** `Congrats!!! You WON the bet lol, I owe you a beer. Call me later` (case #1, nhãn thật `ham`). Từ `Congrats`, `WON`, `Call` thường gắn với spam quảng cáo, nhưng đây là lời nhắn cá nhân về một vụ cá cược. Mô hình BoW có nguy cơ báo nhầm spam vì dựa vào các từ riêng lẻ; **chưa có dự đoán từng model được lưu** để xác nhận model nào sai case này.

**Case cụ thể 2 — spam mạo danh:** `Hey, it's your brother. My phone broke, I'm using this number now. Can you transfer me 200 dollars urgently?` (case do em tự tạo, nhãn `spam`, nhóm `manipulation`). Tin nhắn có văn phong tự nhiên, ít từ khóa spam truyền thống và tạo cảm giác cấp bách. BoW dễ bỏ sót do thiếu đặc trưng từ khóa; embedding/LLM có thể nhận ra dấu hiệu lừa đảo nhưng vẫn có thể nhầm với lời nhờ thật. **Chưa có output dự đoán cho case này.**

Theo phần trả lời trong notebook, nhóm `obfuscation` (né từ khóa) và `natural_spam` (spam giọng tự nhiên) đặc biệt khó với BoW. Biện pháp có thể thử là chuẩn hóa biến thể ký tự (`FR33` → `FREE`) trước khi phân loại, song cần kiểm tra tác dụng phụ và tránh sửa nhầm tin bình thường. Với prompt injection, không được coi nội dung SMS là chỉ dẫn hệ thống; cần kiểm soát đầu vào và có lớp kiểm tra độc lập. Bộ 50 case viết tay thích hợp tìm điểm yếu, không đại diện cho tỷ lệ spam ngoài thực tế.

## 5. Thử thách cá nhân

*Đề theo SEED của bạn, cách làm, kết quả, giải thích.*
**Điểm cần xác minh:** Với SEED 8113, công thức `SEED % 6` cho ra **1**, tức **thử thách #2: thay đổi kích thước từ vựng BoW**. Notebook đã có code và phần giải thích cho thử thách này. Tuy nhiên, ô báo cáo cuối lại ghi `thử thách #0` do trạng thái `_LOG` không nhất quán; cần chạy lại ô chọn thử thách và ô báo cáo cuối trước khi nộp.

**Đề bài:** Giảm vocab còn 100 từ, tăng lên 5.000 từ, huấn luyện lại Logistic Regression và so sánh F1 test.

**Cách làm:** Với mỗi `max_size`, xây dựng lại từ vựng trên tập train, vector hóa train/test, huấn luyện Logistic Regression cùng cấu hình `lr=0.5`, `epochs=300`, rồi dự đoán trên test với ngưỡng 0.5.

**Kết quả đã ghi trong phần trả lời notebook (ô chạy chưa lưu output):**

| Kích thước vocab | F1 test | Thời gian train |
| --- | ---: | ---: |
| 100 | 0.7925 | 1.51 giây |
| 5.000 | 0.8902 | 21.97 giây |

F1 tăng khoảng **0.0977** điểm, nhưng thời gian huấn luyện tăng khoảng **14.5 lần**. Vocab lớn giúp giữ thêm các từ có ích để phát hiện spam, đổi lại chi phí tính toán và nguy cơ học các từ nhiễu. Kết quả này cần được xác minh lại bằng output của ô thử thách.

## 6. Phương án cho sản phẩm thật

*Với yêu cầu p95 < 50 ms/tin cho 10 triệu SMS/ngày: chọn phương án nào, dẫn số liệu nào, chấp nhận đánh đổi gì.*
**Yêu cầu:** 10 triệu SMS/ngày, trung bình khoảng **115.7 SMS/giây**, p95 **dưới 50 ms/tin**.

Em chọn **BoW + MLP làm mô hình nền cho bản triển khai đầu tiên**: F1 test **0.9070**, p95 **5.30 ms**, tốc độ đo cả lô khoảng **11.434 tin/giây**. Mô hình có khoảng dư độ trễ lớn, không cần gọi API bên ngoài, và chính xác hơn BoW + LogReg (F1 0.8696). Đánh đổi là F1 thấp hơn MiniLM + MLP head (0.9432) và có thể gặp khó trước spam né từ khóa.

**Phương án nâng cao:** MiniLM + LogReg đạt F1 **0.9222**, p95 **45.13 ms**, nhưng throughput đo cả lô chỉ khoảng **30.6 tin/giây**, dưới tải trung bình 115.7 tin/giây trên cấu hình benchmark hiện tại. Muốn dùng cần đo lại với batching, nhiều worker, phần cứng phù hợp và tải thực tế; chỉ dựa vào p95 đơn lẻ là chưa đủ. MiniLM + MLP head có p95 **55.81 ms** nên không đạt SLA đo được.

**Thiết kế lai đề xuất:** Dùng BoW + MLP xử lý mặc định; những tin có xác suất spam nằm trong khoảng **[0.3, 0.7]** chuyển sang tuyến xử lý bổ sung, ví dụ MiniLM + LogReg hoặc hàng đợi rà soát. Không đặt LLM đồng bộ trên đường xử lý bắt buộc: Gemini zero-shot có p95 **1.080,80 ms** và chỉ khoảng **1.48 tin/giây** khi gọi tuần tự, không đạt SLA. Có thể dùng LLM bất đồng bộ cho các tin cần phân tích sâu, với chính sách an toàn cho OTP/tin giao dịch.

Nếu gửi **toàn bộ 10 triệu SMS/ngày** qua Gemini zero-shot theo đơn giá đo được **0.018623 USD/1.000 tin**, chi phí API ước tính **186.23 USD/ngày**, khoảng **5.587 USD/30 ngày**, chưa tính máy chủ, mạng và chi phí khác. Không nên áp dụng con số này cho model hoặc cấu hình prompt khác khi chưa đo lại.

## 7. AI log và điều đã học

*Mỗi lần dùng AI: hỏi gì, nhận gì, bạn sửa gì. Sau đó 2–3 câu: điều bạn hiểu ra hôm nay mà trước đó chưa hiểu.*
**AI log (chỉ ghi những lần sử dụng AI có căn cứ từ cuộc trao đổi này; cần bổ sung nếu còn dùng AI ở nơi khác):**

| Phần | Em hỏi AI gì | AI hỗ trợ gì | Em kiểm tra/chỉnh sửa gì |
| --- | --- | --- | --- |
| Chuẩn bị nộp bài | Cách tải `day01_lab.ipynb` và `lab1_report.json` từ Google Colab | Hướng dẫn tải notebook và JSON | Em đã tải và cung cấp hai file thực tế. |
| Viết báo cáo | Nhờ AI đối chiếu notebook và JSON để điền báo cáo cá nhân theo mẫu | Tổng hợp bảng chỉ số, chỉ ra dữ liệu còn thiếu và sự không nhất quán về thử thách | Em cần đọc lại từng mục, đối chiếu số liệu và bổ sung output/ảnh tính tay trước khi nộp. |

**Điều em hiểu ra:** Một mô hình có F1 cao trên tập test thông thường chưa chắc chống chịu tốt trước các tin nhắn lừa đảo được viết khéo hoặc cố tình né từ khóa. Em cũng nhận ra cần đánh giá đồng thời F1, độ trễ p95, throughput và chi phí; model chính xác nhất chưa chắc phù hợp với SLA của sản phẩm. Cuối cùng, một con số ghi trong phần trả lời chỉ có giá trị kiểm chứng tốt khi đi kèm output và quy trình tái lập thí nghiệm.

---

**Việc cần hoàn tất trước khi nộp:** (1) bổ sung `hand_calc.jpg` có các bước tính tay; (2) chạy lại và lưu output các ô bảng so sánh, stress test và thử thách; (3) đồng bộ trường `challenge` trong `lab1_report.json` với thử thách theo SEED; (4) kiểm tra AI log có ghi đầy đủ mọi lần dùng AI theo quy định môn học.

