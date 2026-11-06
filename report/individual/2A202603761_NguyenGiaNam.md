# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV: 2A202603761
- **Họ và tên: Nguyễn Gia Nam
- - **SEED** (in ở ô đầu notebook): 8113

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`]((../day01_lab.ipynb) — notebook đã chạy, còn output | ☐ |
| [`lab1_report.json`](../lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☐ |
| [`hand_calc.jpg`](../hand_calc.png) — ảnh tính tay có các bước | ☐ |
Dữ liệu sau loại bản ghi trùng tuyệt đối: **5.169 tin** (653 spam, 4.516 ham). Chia tập: train **3.619**, validation **775**, test **775** (91 spam); tập SUB gồm **40** tin. Lưu ý: kết quả 9/9 là các kiểm tra chức năng chính trong JSON, không đồng nghĩa mọi ô và phần tính tay đã được xác nhận.
m sử dụng bộ dữ liệu gồm **5.169 SMS** sau khi loại tin trùng, trong đó có 653 spam và 4.516 ham. Tập test có 775 tin và tập SUB có 40 tin.

## 2. Kết quả chính

| Mô hình | F1 test | F1 SUB 40 | p95 (ms/tin) | USD/1.000 tin |
| --- | ---: | ---: | ---: | ---: |
| BoW + LogReg | 0.8696 | 0.9474 | 1.27 | 0 |
| BoW + MLP | 0.9070 | 0.9474 | 5.30 | 0 |
| MiniLM + kNN | 0.9198 | 1.0000 | 63.65 | 0 |
| MiniLM + LogReg | 0.9222 | 0.9744 | 45.13 | 0 |
| MiniLM + MLP head | **0.9432** | 0.9744 | 55.81 | 0 |
| LLM nhỏ zero-shot | 0.8889* | 0.8889 | 1080.80 | 0.01862 |

*F1 = 0.8889 của LLM là kết quả trên tập SUB, không phải F1 của toàn bộ tập test. Các mức chi phí 0 ở mô hình chạy cục bộ chưa tính tiền máy và điện.*

Nhìn chung, **MiniLM + MLP head** có F1 test cao nhất, còn **BoW + LogReg** chạy nhanh nhất.

## 3. Dự đoán và thực tế

1. **BoW + LogReg:** Em đoán F1 khoảng **0.85**, thực tế **0.8696**. Kết quả tốt hơn một chút vì nhiều tin spam có từ khóa khá rõ, nhưng mô hình vẫn bỏ sót một số tin spam.
2. **Bỏ ReLU:** Em đoán F1 sẽ giảm còn **0.89**, nhưng lần thử không ReLU lại được khoảng **0.913**, nhỉnh hơn MLP có ReLU (**0.907**). Em nhận ra mô hình phức tạp hơn không phải lúc nào cũng tốt hơn trên một bộ dữ liệu nhỏ.
3. **Embedding với 6 câu dễ nhầm:** Em nghĩ MiniLM + LogReg sẽ đúng nhiều hơn BoW + LogReg. Thực tế cả hai đều đúng **5/6 câu** và cùng bỏ sót tin nhắn giả thông báo phần thưởng sắp hết hạn. Embedding hiểu ngữ nghĩa tốt hơn nhưng vẫn có thể sai.

## 4. Phân tích 50 case khó

Theo phần em ghi trong notebook, F1 của **BoW + LogReg** giảm từ **0.870 xuống 0.333**, còn **BoW + MLP** giảm từ **0.907 xuống 0.462**. Trong hai mô hình này, BoW + LogReg giảm nhiều hơn. Các nhóm khó là tin spam viết tự nhiên và tin cố tình né từ khóa. **Lưu ý:** output của ô stress test chưa được lưu nên em cần chạy lại để xác nhận các số liệu này.

- **Case 1:** “Congrats!!! You WON the bet lol, I owe you a beer. Call me later” là **ham**. Các từ như *WON*, *Congrats* dễ làm mô hình nhầm thành spam, dù nội dung chỉ là nhắn tin giữa bạn bè.
- **Case 2:** “Hey, it's your brother. My phone broke ... Can you transfer me 200 dollars urgently?” là **spam**. Câu này giống lời nhờ giúp đỡ thông thường nên mô hình dựa vào từ khóa có thể bỏ sót.

Hai ví dụ trên giải thích nguy cơ nhầm lẫn; notebook chưa lưu dự đoán của từng mô hình cho từng câu.

## 5. Thử thách cá nhân

Với **SEED 8113**, thử thách em làm là **thay đổi kích thước từ vựng BoW** rồi huấn luyện lại Logistic Regression.

| Số từ trong vocab | F1 test | Thời gian train |
| --- | ---: | ---: |
| 100 | 0.7925 | 1.51 giây |
| 5.000 | 0.8902 | 21.97 giây |

Theo kết quả em đã ghi trong notebook, tăng vocab giúp F1 tốt hơn khoảng **0.098**, nhưng thời gian huấn luyện cũng tăng nhiều. Em thấy cần cân bằng giữa số lượng đặc trưng và tốc độ. *Cần chạy lại ô thử thách để lưu output và đồng bộ mục thử thách trong JSON.*

## 6. Phương án cho sản phẩm thật

Với yêu cầu **10 triệu SMS/ngày** (trung bình khoảng **116 tin/giây**) và **p95 dưới 50 ms/tin**, em chọn **BoW + MLP** để triển khai ban đầu. Mô hình có **F1 = 0.907**, **p95 = 5.30 ms** và tốc độ đo được khoảng **11.434 tin/giây**, nên có khoảng dư tốt về tốc độ. Em chấp nhận F1 thấp hơn MiniLM + MLP head (**0.943**) để giảm độ trễ.

Nếu tin nhắn khó phân loại, em sẽ cân nhắc dùng MiniLM + LogReg để kiểm tra thêm. Trước khi đưa vào thực tế vẫn cần thử tải đồng thời, vì kết quả trong Colab chưa chứng minh hệ thống chịu được 10 triệu tin/ngày.

## 7. AI log và điều đã học

| Em dùng AI để hỏi | AI hỗ trợ | Em kiểm tra/chỉnh sửa |
| --- | --- | --- |
| Cách tải file từ Google Colab | Hướng dẫn tải notebook và JSON | Em tải đúng hai file kết quả |
| Cách viết báo cáo từ kết quả LAB | Tổng hợp số liệu và gợi ý cách diễn đạt | Em đối chiếu bảng kết quả, rút ngắn phần giải thích và giữ các chỗ cần xác minh |

Qua bài LAB, em hiểu rằng **F1 cao chưa chắc mô hình đã phù hợp để triển khai**, vì còn phải xem tốc độ và chi phí. Em cũng thấy các tin nhắn spam viết giống người thật có thể làm mô hình dựa vào từ khóa dự đoán sai. Lần sau em sẽ chú ý lưu đầy đủ output để dễ đối chiếu kết quả.
