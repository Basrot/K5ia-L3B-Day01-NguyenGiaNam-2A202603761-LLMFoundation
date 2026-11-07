# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV:** 2A202603761
- **Họ và tên:** Nguyễn Gia Nam
- **SEED:** 8113
  
## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [day01_lab.ipynb](day01_lab.ipynb) — notebook đã chạy, có lưu output | ☑ |
| [lab1_report.json](lab1_report.json) — kết quả do ô "Báo cáo cuối" sinh ra | ☑ |
| [hand_calc.jpg](hand_calc.jpg) — ảnh tính tay có các bước thực hiện | ☑ |

Em sử dụng bộ dữ liệu SMS Spam Collection gồm **5.169 tin nhắn** sau khi loại bỏ các bản ghi trùng tuyệt đối, trong đó có **653 spam** và **4.516 ham**.

Dữ liệu được chia thành:

- **Train:** 3.619 tin nhắn.
- **Validation:** 775 tin nhắn.
- **Test:** 775 tin nhắn, gồm 91 spam.
- **SUB:** 40 tin nhắn dùng cho một số thí nghiệm, đặc biệt là LLM.

Notebook đã lưu **80 output**, bao gồm kết quả chạy mô hình, các phép kiểm tra và phần báo cáo cuối.

Theo file `lab1_report.json`, em đạt **14/14 kiểm tra**, đã ghi nhận **10/10 dự đoán** và có kết quả của **7 cấu hình mô hình**.

Các kiểm tra bao gồm những chức năng chính của ML, DL, Embedding, LLM và năm phép tính tay. Kết quả kiểm tra đạt không có nghĩa là mọi mô hình đều dự đoán chính xác tuyệt đối, mà cho thấy các chức năng được kiểm tra đã đáp ứng yêu cầu tương ứng.

## 2. Kết quả chính

| Mô hình | F1 test | F1 SUB 40 | p95 (ms/tin) | USD/1.000 tin |
| --- | ---: | ---: | ---: | ---: |
| BoW + LogReg | 0.8696 | 0.9474 | 0.075 | 0 |
| BoW + MLP | 0.9070 | 0.9474 | 0.245 | 0 |
| MiniLM + kNN | 0.9198 | 1.0000 | 124.17 | 0 |
| MiniLM + LogReg | 0.9222 | 0.9744 | 40.91 | 0 |
| MiniLM + MLP head | **0.9432** | 0.9744 | 62.34 | 0 |
| LLM nhỏ zero-shot | 0.9302* | 0.9302 | 4038.72 | 0.01862 |
| LLM nhỏ few-shot | 1.0000* | 1.0000 | 743.21 | 0.05382 |

*Lưu ý: F1 của hai cấu hình LLM được đánh giá trên tập SUB gồm 40 tin nhắn, không phải toàn bộ tập test 775 tin. Vì vậy, không nên so sánh trực tiếp các chỉ số này như thể chúng được đo trên cùng một tập dữ liệu. Chi phí 0 USD của các mô hình chạy cục bộ chưa tính tiền phần cứng và điện năng.*

Qua kết quả thực nghiệm, em nhận thấy **MiniLM + MLP head có F1 test cao nhất là 0.9432** trong các mô hình được đánh giá trên toàn bộ tập test. Điều này cho thấy biểu diễn văn bản bằng embedding kết hợp mạng nơ-ron có thể cải thiện chất lượng phân loại so với phương pháp chỉ sử dụng Bag of Words.

Trong khi đó, **BoW + LogReg** có độ trễ thấp nhất, với p95 khoảng 0.075 ms/tin. Mặc dù F1 thấp hơn các mô hình MiniLM, phương pháp này có lợi thế rõ rệt về tốc độ.

Đối với LLM, zero-shot đạt F1 = 0.9302 trên SUB, còn few-shot đạt F1 = 1.0000. Tuy nhiên, few-shot có chi phí cao hơn và kết quả trên 40 tin nhắn chưa đủ để kết luận rằng mô hình sẽ đạt độ chính xác tuyệt đối trên dữ liệu thực tế.

Em rút ra rằng **mô hình có F1 cao nhất chưa chắc là phương án phù hợp nhất để triển khai**, vì còn phải cân nhắc độ trễ, chi phí và khả năng xử lý số lượng lớn tin nhắn.

## 3. Dự đoán và thực tế

**1. BoW + Logistic Regression**

Ban đầu, em dự đoán F1 của BoW + LogReg khoảng **0.85**. Kết quả thực tế đạt **0.8696**, cao hơn dự đoán khoảng 0.0196.

Em nghĩ nguyên nhân là nhiều tin nhắn spam trong bộ dữ liệu có những từ khóa tương đối rõ ràng, giúp mô hình nhận diện tốt. Tuy nhiên, recall của mô hình chỉ đạt khoảng 0.7692, cho thấy vẫn có một số tin nhắn spam bị bỏ sót.

Qua đó, em hiểu rằng accuracy cao không đồng nghĩa với việc mô hình phát hiện được tất cả tin nhắn spam. Cần xem thêm precision, recall và F1 để đánh giá đầy đủ.

**2. Thử nghiệm bỏ hàm kích hoạt ReLU**

Em dự đoán khi bỏ ReLU, F1 của MLP sẽ giảm xuống khoảng **0.89**, vì hai lớp Linear liên tiếp không có hàm kích hoạt phi tuyến sẽ tương đương với một phép biến đổi tuyến tính.

Tuy nhiên, kết quả thử nghiệm trong notebook cho thấy mô hình không sử dụng ReLU đạt F1 khoảng **0.913**, nhỉnh hơn MLP có ReLU với F1 khoảng **0.907**.

Kết quả này khác với dự đoán ban đầu của em. Em nhận ra rằng việc tăng độ phức tạp của mô hình không phải lúc nào cũng làm kết quả tốt hơn, đặc biệt khi dữ liệu không quá lớn.

Điều đó không có nghĩa ReLU không hữu ích. Trên những bài toán phức tạp hơn, khả năng học quan hệ phi tuyến vẫn có thể mang lại lợi thế.

**3. Embedding với các câu dễ nhầm**

Em dự đoán MiniLM + LogReg sẽ phân loại tốt hơn BoW + LogReg trên các câu dễ nhầm, vì embedding có thể biểu diễn ngữ nghĩa của toàn bộ câu thay vì chỉ dựa vào sự xuất hiện của từ.

Theo phần đối chiếu em ghi trong notebook, cả hai mô hình đều phân loại đúng **5/6 câu** và cùng bỏ sót một tin nhắn giả thông báo phần thưởng sắp hết hạn.

Kết quả này cho thấy embedding không phải lúc nào cũng vượt trội trên từng trường hợp cụ thể. Mặc dù biểu diễn ngữ nghĩa tốt hơn, mô hình vẫn có thể dự đoán sai khi nội dung mang tính lừa đảo được diễn đạt giống tin nhắn thông thường.

Qua ba thử nghiệm, em nhận thấy cần kiểm chứng các dự đoán bằng kết quả thực tế thay vì chỉ dựa vào lý thuyết.

## 4. Phân tích 50 case khó

Ngoài tập test thông thường, em sử dụng phần stress test gồm **50 trường hợp khó** để đánh giá khả năng xử lý các tin nhắn không điển hình.

Các trường hợp bao gồm spam viết giống tin nhắn cá nhân, tin nhắn chứa từ khóa dễ gây hiểu nhầm, spam né từ khóa và những nội dung có dấu hiệu prompt injection.

Kết quả được lưu trong output stress test của notebook:

| Mô hình | F1 trên SUB thông thường | F1 trên 50 case khó | Accuracy trên 50 case khó |
| --- | ---: | ---: | ---: |
| BoW + LogReg | 0.947 | 0.410 | 54% |
| BoW + MLP | 0.947 | 0.565 | 60% |
| LLM nhỏ zero-shot | 0.930 | 0.983 | 98% |
| LLM nhỏ + prompt phòng thủ | 0.930 | 1.000 | 100% |

Trong các mô hình BoW, **BoW + LogReg giảm mạnh nhất**, từ F1 = 0.947 xuống 0.410, tức giảm khoảng 0.537. BoW + MLP cũng giảm từ 0.947 xuống 0.565.

Em cho rằng nguyên nhân là các mô hình BoW chủ yếu dựa vào những từ xuất hiện trong văn bản. Khi tin nhắn spam được viết tự nhiên hơn hoặc cố tình thay đổi cách viết từ khóa, mô hình dễ bỏ sót.

Trong khi đó, LLM có thể xem xét ngữ cảnh của cả câu nên xử lý tốt hơn các trường hợp khó trong thử nghiệm này. Prompt phòng thủ cũng cho kết quả tốt hơn zero-shot thông thường.

Tuy nhiên, **50 case này là tập kiểm thử nhỏ được xây dựng có chủ đích**, không đại diện đầy đủ cho dữ liệu SMS thực tế. Vì vậy, kết quả 100% không chứng minh LLM có thể phát hiện mọi loại spam hoặc chống được mọi hình thức prompt injection.

**Case 1 — Tin nhắn ham chứa từ khóa giống spam**

> “Congrats!!! You WON the bet lol, I owe you a beer. Call me later”

Nhãn thực tế: **ham**.

Các từ như *Congrats* và *WON* thường xuất hiện trong tin nhắn quảng cáo hoặc thông báo trúng thưởng. Tuy nhiên, xét toàn bộ ngữ cảnh, đây chỉ là cuộc trò chuyện thông thường giữa bạn bè.

Ví dụ này cho thấy nếu chỉ dựa vào từ khóa, mô hình có thể nhầm tin nhắn hợp lệ thành spam.

**Case 2 — Tin nhắn spam giả danh người thân**

> “Hey, it's your brother. My phone broke, I'm using this number now. Can you transfer me 200 dollars urgently?”

Nhãn thực tế: **spam**.

Tin nhắn sử dụng cách nói giống một người thân đang cần giúp đỡ. Tuy nhiên, nội dung yêu cầu chuyển tiền khẩn cấp từ một số điện thoại lạ là dấu hiệu đáng ngờ.

Mô hình dựa vào từ khóa có thể bỏ sót dạng tin nhắn này vì không xuất hiện những từ spam quen thuộc như *free*, *prize* hoặc *winner*.

Hai ví dụ trên minh họa các kiểu nội dung gây khó khăn cho mô hình. Em không xem chúng là bằng chứng một mô hình cụ thể đã dự đoán sai nếu chưa đối chiếu dự đoán riêng của mô hình đó.

**Bài học từ stress test**

Em nhận thấy cần kiểm tra mô hình trên cả những trường hợp bất thường, thay vì chỉ sử dụng tập test được chia ngẫu nhiên từ cùng bộ dữ liệu huấn luyện.

Đối với spam né từ khóa, có thể thử bước tiền xử lý để chuẩn hóa một số ký tự bị thay thế, chẳng hạn `FR33` thành `FREE`. Tuy nhiên, cách này cũng có nguy cơ làm thay đổi nội dung hợp lệ, nên cần kiểm thử cẩn thận.

Đối với LLM, ngoài prompt phòng thủ, hệ thống nên có thêm cơ chế kiểm tra đầu ra, giới hạn dữ liệu đầu vào và không phụ thuộc hoàn toàn vào một lần dự đoán của LLM.

## 5. Thử thách cá nhân

Với **SEED 8113**, em được giao **thử thách số 2: Thay đổi kích thước từ vựng của mô hình BoW + Logistic Regression**.

Mục tiêu của thử thách là giảm kích thước vocabulary xuống 100 từ và tăng lên 5.000 từ, sau đó huấn luyện lại mô hình để so sánh F1 test và thời gian huấn luyện.

Kết quả em đã ghi nhận trong phần giải thích thử thách:

| Số từ trong vocabulary | F1 test | Thời gian train |
| --- | ---: | ---: |
| 100 | 0.7925 | 1.51 giây |
| 5.000 | 0.8902 | 21.97 giây |

Khi tăng vocabulary từ 100 lên 5.000 từ, F1 tăng khoảng **0.0977**, nhưng thời gian huấn luyện cũng tăng hơn 14 lần.

Theo em, vocabulary chỉ có 100 từ là khá nhỏ, khiến mô hình bỏ qua nhiều từ có ích cho việc phân biệt spam và ham.

Khi tăng lên 5.000 từ, mô hình có thêm đặc trưng để nhận diện các dạng tin nhắn khác nhau, nhờ đó F1 cải thiện.

Tuy nhiên, vocabulary lớn cũng khiến số chiều của dữ liệu tăng, dẫn đến thời gian huấn luyện và tài nguyên tính toán cao hơn.

Em cho rằng không phải cứ tăng số lượng từ là F1 sẽ tiếp tục tăng. Khi vocabulary quá lớn, mô hình có thể bổ sung nhiều từ ít hữu ích hoặc nhiễu.

Nếu ưu tiên chất lượng phân loại, em sẽ cân nhắc vocabulary lớn hơn. Nếu ưu tiên tài nguyên và tốc độ, em sẽ lựa chọn kích thước từ vựng phù hợp sau khi kiểm thử.

*Lưu ý về bằng chứng: Notebook đã có code thử thách và phần giải thích kết quả, nhưng output của riêng ô chạy thử thách chưa được lưu. Vì vậy, các số liệu trong bảng là kết quả em đã ghi trong phần trả lời, chưa được xác minh lại bằng output của lần chạy cuối.*

## 6. Phương án cho sản phẩm thật

Giả sử một nhà mạng cần xử lý **10 triệu SMS/ngày**, tương đương trung bình khoảng **116 tin nhắn/giây**, với yêu cầu **p95 dưới 50 ms/tin**.

Dựa trên bảng kết quả, em chọn **BoW + MLP** làm mô hình phân loại ban đầu.

Mô hình có các chỉ số:

- **F1 test:** 0.9070.
- **p95:** 0.245 ms/tin.
- **Tốc độ đo được:** khoảng 42.078 tin nhắn/giây.
- **Chi phí API:** 0 USD, chưa tính phần cứng và vận hành.

So với BoW + LogReg, MLP có F1 cao hơn nhưng vẫn giữ độ trễ thấp.

Trong khi đó, MiniLM + MLP head có F1 test cao hơn, đạt 0.9432, nhưng p95 khoảng 62.34 ms/tin, vượt mức yêu cầu 50 ms trong phép đo hiện tại.

MiniLM + LogReg có F1 test 0.9222 và p95 khoảng 40.91 ms/tin, cũng là một phương án đáng cân nhắc. Tuy nhiên, em ưu tiên BoW + MLP ở giai đoạn đầu vì có độ trễ thấp hơn nhiều.

**Đề xuất hệ thống kết hợp**

Em đề xuất xây dựng hệ thống gồm hai tầng:

**Tầng 1 — BoW + MLP**

Phân loại phần lớn tin nhắn bằng mô hình có tốc độ nhanh. Những tin có xác suất dự đoán rõ ràng sẽ được xử lý ngay.

**Tầng 2 — MiniLM + LogReg**

Những tin nhắn có xác suất chưa chắc chắn, ví dụ trong khoảng 0.3 đến 0.7, sẽ được chuyển sang MiniLM + LogReg để kiểm tra thêm.

Cách kết hợp này giúp hệ thống tận dụng tốc độ của BoW + MLP và khả năng biểu diễn ngữ nghĩa của MiniLM.

Tuy nhiên, khoảng ngưỡng 0.3–0.7 mới là đề xuất ban đầu, cần được hiệu chỉnh trên tập validation. Việc chuyển sang tầng thứ hai cũng làm tăng độ trễ tổng, nên cần đo lại p95 của toàn bộ hệ thống.

**Đánh giá phương án LLM**

Theo kết quả đo trên SUB:

- LLM zero-shot có chi phí khoảng 0.01862 USD/1.000 tin.
- LLM few-shot có chi phí khoảng 0.05382 USD/1.000 tin.

Nếu giả định mức chi phí trên giữ nguyên và phải xử lý toàn bộ 10 triệu tin nhắn mỗi ngày, chi phí API ước tính cho 30 ngày là:

- **Zero-shot:** khoảng 5.587 USD/tháng.
- **Few-shot:** khoảng 16.147 USD/tháng.

Đây chỉ là phép ngoại suy từ thử nghiệm nhỏ, chưa bao gồm biến động token, giá API và chi phí vận hành khác.

Ngoài chi phí, độ trễ của LLM cũng là vấn đề lớn. Cả hai cấu hình đều vượt yêu cầu p95 dưới 50 ms/tin trong phép đo hiện tại.

Vì vậy, em không chọn LLM để phân loại trực tiếp toàn bộ tin nhắn. Nếu sử dụng, em chỉ cân nhắc LLM cho một tỷ lệ nhỏ các trường hợp cần phân tích bổ sung, không nằm trên đường xử lý bắt buộc của mọi tin nhắn.

**Kết luận triển khai**

Em ưu tiên BoW + MLP cho phiên bản ban đầu và cân nhắc kết hợp MiniLM + LogReg để xử lý các trường hợp khó.

Trước khi đưa vào thực tế, cần tiếp tục kiểm thử khả năng xử lý đồng thời, mức sử dụng RAM/CPU, độ trễ khi tải cao và tỷ lệ chặn nhầm tin nhắn hợp lệ.

Kết quả đo trong Google Colab chưa đủ để khẳng định hệ thống có thể đáp ứng 10 triệu SMS/ngày trong môi trường production.

## 7. AI log và điều đã học

| Em dùng AI để hỏi | AI hỗ trợ | Em kiểm tra/chỉnh sửa |
| --- | --- | --- |
| Cách tải notebook và JSON từ Google Colab | Hướng dẫn tải file và giữ lại output | Em kiểm tra notebook có 80 output và JSON có đầy đủ các trường kết quả |
| Cách xử lý lỗi khi push GitHub | Giải thích lỗi GitHub Push Protection, Git và cách đồng bộ file | Em điều chỉnh cấu hình API key, kiểm tra lại file và push thành công |
| Cách tổ chức báo cáo từ kết quả LAB | Gợi ý bố cục, đối chiếu số liệu và cách trình bày | Em đối chiếu các chỉ số với notebook và JSON, giữ lại các phần chưa có đủ bằng chứng dưới dạng ghi chú |
| Cách kiểm tra yêu cầu nộp bài | Hướng dẫn kiểm tra tên file, đường dẫn và cấu trúc thư mục | Em sắp xếp các file trong `report/individual/` và kiểm tra kết quả trước khi nộp |

Qua bài LAB, em học được rằng **Machine Learning, Deep Learning, Embedding và LLM có những ưu điểm và hạn chế khác nhau**.

BoW + Logistic Regression đơn giản và chạy nhanh nhưng có thể bỏ sót các dạng spam viết tự nhiên hoặc cố tình né từ khóa.

MLP có thể cải thiện chất lượng phân loại, nhưng việc tăng độ phức tạp của mô hình không đảm bảo kết quả luôn tốt hơn.

Embedding giúp biểu diễn ngữ nghĩa của câu, trong khi LLM có thể xử lý nhiều trường hợp khó mà mô hình dựa vào từ khóa gặp hạn chế. Tuy nhiên, LLM có chi phí và độ trễ lớn hơn, đồng thời vẫn có thể bị ảnh hưởng bởi những đầu vào được thiết kế để gây nhầm lẫn.

Điều khiến em thay đổi suy nghĩ nhiều nhất là **mô hình phức tạp hơn không nhất thiết đạt kết quả tốt hơn trong mọi thử nghiệm**. Ví dụ, em dự đoán bỏ ReLU sẽ làm F1 giảm, nhưng kết quả thực tế lại tăng nhẹ.

Em cũng hiểu rõ hơn rằng khi đánh giá mô hình, không nên chỉ nhìn vào accuracy hoặc F1. Cần xem thêm precision, recall, tốc độ, độ trễ p95, chi phí và khả năng hoạt động với dữ liệu khác phân phối huấn luyện.

Trong quá trình thực hiện, em có gặp lỗi khi gọi API và đồng bộ file lên GitHub. Qua đó, em học thêm cách đọc thông báo lỗi, kiểm tra kết quả đã lưu và bảo vệ API key trước khi công khai mã nguồn.

**Bài học quan trọng nhất của em là phải đối chiếu dự đoán với kết quả thực nghiệm, hiểu nguyên nhân khi mô hình sai và lựa chọn giải pháp dựa trên yêu cầu thực tế thay vì chỉ chọn mô hình có điểm số cao nhất.**
