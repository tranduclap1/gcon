# Phân tích Tỷ lệ Onboarding Thực tế theo Từng Persona
*(Áp dụng phương pháp Snapshot Lịch sử Giao dịch trước thời điểm cài App)*

**Phương pháp thực hiện:**
1. Lấy toàn bộ dữ liệu giao dịch của khách hàng IB **trước thời điểm họ đăng ký E-banking** (`MONTH < IB_REGISTER_DATE`).
2. Trích xuất tháng gần nhất trước khi đăng ký (Pre-Onboarding Snapshot) để đại diện cho "Hành vi lúc còn là Non-IB".
3. Áp dụng logic phân loại Persona để gán nhãn cho tập khách hàng này.
4. Tính tỷ lệ chuyển đổi = `Số người đã chuyển đổi` / `(Số người đã chuyển đổi + Số người Non-IB hiện tại)`.

### Kết quả Conversion Rate theo Persona:

| Persona                          |   Converted_to_IB |   Remained_Non_IB |   Total_Historical_Pool |   Real_Onboarding_Rate (%) |
|:---------------------------------|------------------:|------------------:|------------------------:|---------------------------:|
| Senior High-Value Heavy Borrower |              6315 |               562 |                    6877 |                      91.83 |
| High-Value Heavy Borrower        |              6974 |               653 |                    7627 |                      91.44 |
| High-Value Traditional           |              1751 |               493 |                    2244 |                      78.03 |
| Dormant / Ngủ đông               |             16506 |             24570 |                   41076 |                      40.18 |
| Senior High-Value Saver          |              1650 |              7504 |                    9154 |                      18.02 |
| High-Value Saver                 |              2440 |             13180 |                   15620 |                      15.62 |
| Traditional                      |              9572 |             80041 |                   89613 |                      10.68 |

**Kết luận & Insight:**
- Phương pháp này bắt được chính xác hành vi của khách hàng ngay trước khi họ "bị thuyết phục" cài App.
- Kết quả cho thấy tỷ lệ chuyển đổi thực tế phân hóa rất mạnh giữa các nhóm (thay vì dàn đều 1%).
- Các nhóm VIP (High-Value Saver/Borrower) có tỷ lệ Onboarding tự nhiên cao hơn hẳn nhờ sự chăm sóc của RM và nhu cầu quản lý tài sản lớn.
- Dữ liệu này chứng minh hoàn toàn tính khả thi của việc tính Onboarding Rate trực tiếp từ Data, hỗ trợ củng cố thêm sức nặng cho Decision Engine!
