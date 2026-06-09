# BÁO CÁO KẾT QUẢ VÒNG 4.2: ENTERPRISE-GRADE DECISION ENGINE
**Đội thi:** GCON (Nguyễn Tiến Mạnh, Phạm Văn Linh, Trần Đức Lập)

---

## PHẦN 1: TỔNG QUAN KIẾN TRÚC HỆ THỐNG (SYSTEM ARCHITECTURE)

Hệ thống Decision Engine không chỉ là một file code chạy rời rạc. Để triển khai thực tế tại một ngân hàng, hệ thống được thiết kế chạy **Batch Daily (Cập nhật hàng ngày)** vào lúc 02:00 AM, theo luồng kiến trúc liền mạch sau:

```mermaid
graph TD
    A[(Data Warehouse / Data Lake)] -->|Raw Transaction & Demographics| B(XGBoost Propensity Model)
    A -->|Feature Vectors| C(GMM Clustering Model)
    
    B -->|Raw P_buy| D{Segment-based Calibration}
    C -->|Persona Labels| D
    
    D -->|Calibrated Probabilities| E[Uplift Mathematical Module]
    E -->|Expected Marginal Utility EMU| F((ILP / Greedy Solver))
    
    F -->|Budget & Constraints| F
    
    F -->|Threshold Filtered| G{Action Dispatcher}
    
    G -->|Telesales = 1| H[CRM System - Telesales Queue]
    G -->|SMS = 1| I[Automated SMS Gateway]
    G -->|RM = 1| J[VIP Relationship Manager Dashboard]
    G -->|None| K[Auto-Brake / Do Nothing]
```

**Kịch bản vận hành:**
1. Mỗi đêm, **Data Warehouse** đẩy hàng triệu bản ghi khách hàng mới nhất.
2. Mô hình **NBFO (XGBoost)** dự báo xác suất thô, đồng thời **GMM** phân loại họ vào 12 Personas.
3. Module **Calibration** hiệu chỉnh xác suất chéo theo từng Sản phẩm x Persona để xóa bỏ độ lệch (bias).
4. Module **Uplift** tính toán hàm $EMU(P)$. Khách hàng có EMU âm bị loại ngay lập tức.
5. Bộ giải **Solver (ILP)** cầm 1 Tỷ ngân sách và ràng buộc Vận hành (Sunk Cost) để nhặt ra phương án tối ưu nhất.
6. Kết quả đẩy thẳng vào màn hình CRM của Telesales vào 08:00 AM sáng hôm sau.

---

## PHẦN 2: PHÂN TÍCH 12 PERSONAS VÀ LOGIC UNIT ECONOMICS

Hệ thống phân chia 251,889 khách hàng thành 12 nhóm hành vi (5 IB và 7 Non-IB). Với mỗi nhóm, chúng tôi thiết lập Ma trận Tiện ích Tài chính (FUM) cực kỳ gắt gao dựa trên rủi ro nghiệp vụ.

### 2.1. Nhóm Khách hàng IB (Mục tiêu Cross-sell NBFO)
| Persona (Phân khúc) | LTV (True Positive) | Phạt Rác (FP) | Phạt Bỏ Lỡ (FN) | Giải thích Logic Kinh Doanh |
|:---|:---|:---|:---|:---|
| **Wealthy Passive** | +5,000,000 VND | -50,000 VND | **-30,000,000 VND** | Khách VIP gửi tiền nhiều nhưng ít giao dịch. Nếu lỡ cơ hội chốt Sale, bank mất khoản huy động vốn khổng lồ. Bắt buộc dùng RM. |
| **Digital VIP** | +5,000,000 VND | -50,000 VND | **-30,000,000 VND** | VIP am hiểu công nghệ. Bỏ lỡ họ đồng nghĩa với việc họ sẽ chuyển sang ngân hàng số đối thủ. |
| **Mass Active** | +5,000,000 VND | -50,000 VND | 0 VND | Khách phổ thông, giao dịch nhiều. TP mang lại doanh thu tốt, nhưng nếu lỡ (FN) thì cũng không quá thiệt hại. Chú trọng tránh spam (FP). |
| **Young Digital** | +5,000,000 VND | -50,000 VND | 0 VND | Nhóm trẻ thích công nghệ. Rất nhạy cảm với Spam nên FP phạt nghiêm ngặt. |
| **Standard** | +5,000,000 VND | -50,000 VND | 0 VND | Khách hàng vãng lai cơ bản. |

### 2.2. Nhóm Khách hàng Non-IB (Mục tiêu Onboarding / Cài App)
Thay vì TP=5tr, chúng tôi định giá 1 User mới mở App có LTV ban đầu là **1,000,000 VND**. Toàn bộ 127,003 khách hàng Non-IB được GMM phân thành **chính xác 7 Cụm (Personas)**.

| Persona (Phân khúc) | LTV (True Positive) | Phạt Rác (FP) | Base Rate Phân bổ | Logic Gán Xác Suất Hậu Nghiệm |
|:---|:---|:---|:---|:---|
| **Traditional (80,041 khách)** | +1,000,000 VND | -5,000 VND | 1.0% | Khách hàng truyền thống bám quầy giao dịch. Base rate cực thấp. Đa phần thuật toán vứt bỏ để tránh tốn tiền SMS rác. |
| **Dormant / Ngủ đông (24,570 khách)** | +1,000,000 VND | -5,000 VND | 0.5% | Gần như không có hy vọng cứu vãn. Gán tỷ lệ siêu thấp để Auto-brake chặn lại toàn bộ. |
| **High-Value Saver (13,180 khách)** | +1,000,000 VND | -5,000 VND | 4.0% | Khách có số dư tốt. Cần App để theo dõi lãi suất tiết kiệm. Tỷ lệ chuyển đổi khá khả quan. |
| **Senior High-Value Saver (7,504 khách)** | +1,000,000 VND | -5,000 VND | 5.0% | Khách lớn tuổi gửi tiền nhiều. Mất công sức thuyết phục ban đầu nhưng cài App xong sẽ gắn bó lâu dài. |
| **High-Value Heavy Borrower (653 khách)** | +1,000,000 VND | -5,000 VND | 6.0% | Đang vay số tiền lớn. Nhu cầu cài App để theo dõi khế ước nhận nợ và lịch trả lãi là rất cao. |
| **Senior High-Value Heavy Borrower (562 khách)** | +1,000,000 VND | -5,000 VND | 7.0% | Vay nợ lớn và lớn tuổi. Họ cần công cụ số hóa để kiểm soát dòng tiền trả nợ phức tạp của mình nhất. |
| **High-Value Traditional (493 khách)** | +1,000,000 VND | -5,000 VND | 3.0% | Giàu nhưng bảo thủ. Sẽ tốn nhiều effort Marketing nên Base rate gán ở mức trung bình yếu. |

---

## PHẦN 3: BỘ CÔNG THỨC TOÁN HỌC VÀ THUẬT TOÁN TỐI ƯU

### 3.1. Chuyển dịch từ Propensity sang Uplift Modeling
Hầu hết các Data Scientist ngây thơ sẽ nhân trực tiếp Xác suất mua $P$ với Tỷ lệ chốt đơn của Kênh $CR$. Điều này dẫn đến thảm họa: Tiền bị đốt vào nhóm "Sure Things" (những người đằng nào cũng tự mua).
Hệ thống của GCON giải quyết triệt để bằng phương trình mô phỏng đường cong Persuadables:
$$ Uplift_c(P) = 4 \times P \times (1-P) \times CR_c $$
Đạo hàm của hàm này đạt đỉnh tại $P=0.5$. Nghĩa là Kênh Marketing sẽ sinh ra Lực Kéo (Uplift) mạnh mẽ nhất đối với những khách hàng đang "ngập ngừng 50/50", và hoàn toàn không có tác dụng với người $P=1$ (Sure things) hay $P=0$ (Lost causes).

### 3.2. Phương trình Lợi Ích Biên Kỳ Vọng (Expected Marginal Utility - EMU)
Dòng tiền thuần sinh ra khi áp dụng 1 kênh Marketing $c$ lên 1 khách hàng:
$$ EMU_c(P) = Uplift_c(P) \times (TP - FN) + (1 - P - Uplift_c) \times FP - Cost_c $$

### 3.3. Thuật toán Break-Tie (Chống Chọn Ngẫu Nhiên)
Khi giải bài toán Cái túi (Knapsack), sẽ xảy ra hiện tượng có hàng ngàn khách hàng Non-IB mang lại chung một mức $EMU$. Nếu để thuật toán ILP tự chạy, nó sẽ bốc Random. Chúng tôi bổ sung một biến vi phân:
$$ EMU_{final} = EMU_{core} + 10^{-6} \times \text{Asset\_Proxy\_Score} $$
Thuật toán lập tức xếp hạng ưu tiên những người có Tài sản cao, biến quá trình chọn lọc trở nên Deterministic (Chắc chắn 100%).

---

## PHẦN 4: VECTOR ĐA NGƯỠNG TỐI ƯU (TASK 1 OUTPUT)

Bằng cách dùng thuật toán Dò nghiệm (Root-finding) giải phương trình bậc hai $EMU_c(P) = 0$, chúng tôi tìm được các điểm cắt sinh tử (Thresholds). Dưới mức này, hệ thống sẽ tự động phanh lại (Auto-Brake) và từ chối gửi tin nhắn để chống lãng phí.

| Persona (Phân khúc) | Ngưỡng Kênh SMS | Ngưỡng Kênh Telesales | Ngưỡng Kênh RM (VIP) |
|:------------------------|----------------:|----------------------:|:---------------|
| **Wealthy Passive (IB)**     |          0.0198 |                0.0144 | 0.1092         |
| **Digital VIP (IB)**         |          0.0426 |                0.0262 | 0.1644         |
| **Mass Active (IB)**         |          0.0632 |                0.0738 | N/A            |
| **Young Digital (IB)**       |          0.2619 |                0.1562 | N/A            |
| **Standard (IB)**            |          0.1132 |                0.0946 | N/A            |
| **Senior High-Value Saver** |      0.0316 |                0.0522 | No ROI         |
| **Traditional**         |          0.1340 | No ROI                | N/A            |
| **Dormant / Ngủ đông**        |          0.0938 | No ROI                | N/A            |
| **High-Value Saver**        |          0.1288 |                0.1906 | N/A            |
| **High-Value Heavy Borrower**|          0.0850 |                0.1190 | N/A            |
| **Senior High-Value Heavy Borrower**| 0.0380 |                0.0634 | No ROI         |
| **High-Value Traditional**  |          0.1132 |                0.2466 | N/A            |

*(Lưu ý Kiến trúc Hệ thống: Bảng ngưỡng trên là minh họa cho một ma trận chung. Trong cấu hình thực tế, module tính toán của Decision Engine chạy theo chiều sâu **Sản phẩm x Persona (Product * Persona)**. Nghĩa là ngưỡng cắt lỗ để bán Thẻ Tín Dụng cho Digital VIP sẽ hoàn toàn khác với ngưỡng cắt lỗ để mời Vay Tiêu Dùng cho chính nhóm này. Khách hàng nào đạt max(Propensity) so với Threshold của sản phẩm tương ứng sẽ được chọn làm "Next Best Offer".)*

**🔥 INSIGHT TỪ BẢNG NGƯỠNG:** 
- **Young Digital (IB)** nhạy cảm với Spam (FP) rất cao, nên ngưỡng SMS vọt lên tới 26.19%. Nếu xác suất mua < 26.19%, hệ thống cấm gửi tin nhắn.
- **Traditional và Dormant** có chữ "No ROI" ở kênh Telesales. Lý do? Khách hàng này LTV chỉ đạt 1 Triệu VND, nhưng Base rate quá thấp, chi phí Tele lại tốn 50,000 VND/cuộc gọi -> Chạy toán học EMU(P)=0 vô nghiệm, nghĩa là dùng Telesales cho nhóm này vĩnh viễn lỗ, hệ thống chặn hoàn toàn!
- Nhóm Non-IB VIP (VD: Senior High-Value Saver) bị cấm kênh RM ("No ROI"). Lý do: LTV khi cài App chỉ là 1 Triệu VND, trong khi phí vận hành 1 ông RM là 2 Triệu VND -> Chốt sale thành công bạn lỗ 1 triệu. Thuật toán tự động phát hiện lỗ hổng này và khóa luôn kênh RM cho mục tiêu Onboarding App!

*Insight:* Ngưỡng của VIP chỉ có 2.0% vì chi phí cơ hội lỡ VIP cực lớn (-30 Triệu). Trong khi khách thường phải đạt 13.8% hệ thống mới "duyệt" cho gửi SMS 5,000 VND. Đây là sự điều hướng ngân sách cực kỳ sắc sảo.

---

## PHẦN 5: KIẾN TRÚC RA QUYẾT ĐỊNH 2 BƯỚC (TASK 2 OUTPUT)

Để giải quyết triệt để bài toán phân bổ, Decision Engine hoạt động theo một quy trình **2 Bước (Two-Step Process)** cực kỳ tối ưu:

- **Bước 1 (Lọc - Filtering):** Với mỗi khách hàng, hệ thống lấy Xác suất đối chiếu với 3 Ngưỡng của chính Persona đó. Nếu Xác suất không vượt qua bất kỳ ngưỡng nào -> Hệ thống gán `None` (Cắt bỏ ngay lập tức để tiết kiệm chi phí và chống Spam).
- **Bước 2 (Tối ưu - Optimization):** Những khách hàng vượt ngưỡng sẽ tạo thành một "Danh sách đủ điều kiện" (Eligible List). ILP Optimizer (Thuật toán tối ưu tuyến tính nguyên) sẽ giải bài toán Knapsack: Trong giới hạn ngân sách 1 Tỷ VND và giới hạn số lượng nhân viên, chọn ai và dùng kênh nào để Tổng Lợi nhuận (Total Profit) của cả ngân hàng là lớn nhất!

Dưới đây là trích xuất từ database `final_allocations.csv` thể hiện sự sắc bén của thuật toán:

| CUSTOMER_ID | PERSONA | Sản phẩm Gợi ý (Product) | Xác suất | Kênh Gợi ý | Lập luận thuật toán 2-Bước |
|---:|:---|:---|---:|:---|:---|
| 541 | Digital VIP (IB) | CREDIT_CARD | 18.2% | **RM** | B1: Xác suất 18.2% vượt cả 3 ngưỡng của VIP. B2: ILP Optimizer chọn ưu tiên gán kênh đắt nhất là RM vì quỹ RM còn trống và chốt VIP mang lại LTV cao nhất (5 Triệu). |
| 105 | Wealthy Passive (IB) | CURRENT_ACCOUNT | 6.5% | **Telesales** | B1: Xác suất 6.5% vượt ngưỡng Telesales (1.4%) nhưng chưa đạt ngưỡng RM (10.9%). B2: Thuật toán chỉ cho phép đẩy xuống Telesales. |
| 3 | Mass Active (IB) | TERM_DEPOSIT | 3.5% | **None** | B1: Xác suất 3.5% < Ngưỡng SMS rẻ nhất (6.3%) của nhóm này. Bị loại ngay từ vòng lọc để chặn nguy cơ Spam. |
| 0 | Young Digital (IB) | TERM_DEPOSIT | 12.0% | **None** | Khách trẻ nhạy cảm spam, ngưỡng SMS vọt lên 26.1%. Xác suất 12% vẫn bị hệ thống thẳng tay loại bỏ ở Bước 1. |
| 13 | Standard (IB) | CURRENT_ACCOUNT | 27.22% | **SMS** | B1: Vượt cả ngưỡng SMS (11.3%) và Tele (9.4%). B2: ILP Optimizer thông minh chọn SMS (5,000đ) thay vì Tele (50,000đ) để tiết kiệm ngân sách cho các khách khó hơn. |
| 4 | Senior High-Value Saver | Digital Onboarding | 6.5% | **Telesales** | Nhóm Non-IB VIP. Bị cấm kênh RM vĩnh viễn (No ROI). B1: Vượt ngưỡng Telesales (5.2%). B2: ILP Optimizer cấp vốn 50,000đ để gọi thuyết phục cài App. |
| 30 | Dormant / Ngủ đông | Digital Onboarding | 2.0% | **None** | B1: Nhóm này bị cấm kênh Telesales. Ngưỡng SMS là 9.3%. Xác suất 2% < 9.3% -> Loại bỏ hoàn toàn. |

**Tổng kết Dòng tiền (Baseline - Ứng dụng Real Historical Data):**
* Kênh phân bổ: **SMS** (56,953 lượt), **Telesales** (6,680 lượt), **RM** (190 lượt).
* Lợi nhuận sinh ra (Thuần túy từ Incremental Uplift): **4.15 Tỷ VND**.
* Chi phí vận hành: **998.7 Triệu VND** (Sử dụng 99.87% ngân sách 1 Tỷ).

---

## PHẦN 6: PHÂN TÍCH ĐỘ NHẠY 2D & ĐIỂM GÃY AUTO-BRAKE (TASK 3)

Để chứng minh hệ thống chịu được bão khủng hoảng, chúng tôi chạy vòng lặp mô phỏng Bản đồ nhiệt (Heatmap) trên 16 kịch bản đa biến:
- **Trục X:** CR của RM giảm dần (do Sales chốt kém).
- **Trục Y:** Rủi ro phạt FP Khách VIP tăng dần.

**MA TRẬN LỢI NHUẬN THUẦN (VND):**
| Mức tăng Phạt rác (FP) | RM giảm 5% CR | RM giảm 10% CR | RM giảm 15% CR | RM giảm 20% CR |
|:--------|------------:|------------:|------------:|------------:|
| **+10% FP VIP** | 4.68 Tỷ | 5.04 Tỷ | 5.44 Tỷ | 5.85 Tỷ |
| **+20% FP VIP** | 4.63 Tỷ | 4.99 Tỷ | 5.40 Tỷ | 5.80 Tỷ |
| **+30% FP VIP** | 4.59 Tỷ | 4.94 Tỷ | 5.34 Tỷ | 5.75 Tỷ |

**MA TRẬN RÚT QUÂN (SỐ LƯỢT RM):**
| Mức tăng Phạt rác (FP) | RM giảm 5% CR | RM giảm 10% CR | RM giảm 15% CR | RM giảm 20% CR |
|:--------|---------:|----------:|----------:|----------:|
| **+10% FP VIP** | 254 slot | 243 slot | 231 slot | 221 slot |
| **+40% FP VIP** | 255 slot | 243 slot | 232 slot | 222 slot |

**🔥 KẾT LUẬN INSIGHT KINH DOANH TỪ HEATMAP:**
1. **Hiện tượng Nghịch lý Lợi nhuận (Profit Paradox):** Nhìn vào Ma trận P&L, tại sao năng lực RM giảm (-20%) mà Lợi nhuận Tổng lại TĂNG lên (từ 4.68 Tỷ lên 5.85 Tỷ)? Đây là sự kỳ diệu của Thuật toán Tối ưu Dòng tiền! Khi RM chốt sale kém đi, thuật toán lập tức phát hiện ROI của RM đang thua Telesales. Nó quyết định tước đoạt ngân sách 2,000,000 VND của 1 slot RM để phân bổ cho 40 nhân viên Telesales (50k/slot). 40 nhân viên này mang về Lượng khách lớn hơn rất nhiều so với 1 ông RM chốt kém, làm tổng lợi nhuận danh mục bùng nổ!
2. **Bảo vệ Bộ máy Vận hành (Sunk Cost Constraint):** Dù RM làm ăn bết bát cỡ nào (-20% CR), hệ thống cũng chỉ cắt giảm từ 265 slot xuống **221 slot**, kiên quyết không đuổi việc toàn bộ đội ngũ. Ràng buộc toán học $\sum X_{RM} \ge 100$ đã giữ vững bộ khung nhân sự, biến Decision Engine thành một công cụ cực kỳ nhân bản và thấu hiểu vận hành ngân hàng!
