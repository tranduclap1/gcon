# BÁO CÁO TỔNG HỢP VÒNG 4: DECISION ENGINE CHO NBFO VÀ RETENTION

**Đội thi:** GCON (Nguyễn Tiến Mạnh, Phạm Văn Linh, Trần Đức Lập)  
**Phạm vi:** Tổng hợp hai engine IB và Non-IB trong folder `Vòng 4 (Decision Engine)`  
**Hướng tiếp cận:** Next Best Financial Offer, phân khúc khách hàng, tối ưu giữ chân và ra quyết định dưới ràng buộc ngân sách.

---

## 1. Tóm tắt điều hành

Trong bối cảnh ngân hàng số, đội GCON xây dựng một hệ thống decision engine nhằm chuyển kết quả phân tích dữ liệu và mô hình dự đoán thành khuyến nghị hành động có thể triển khai thực tế. Hệ thống không chỉ dự đoán xác suất khách hàng phù hợp với sản phẩm tài chính, mà còn tối ưu kênh tiếp cận, chi phí chiến dịch và năng lực vận hành của Telesales/RM.

Theo yêu cầu đề bài, giải pháp tập trung vào ba mục tiêu kinh doanh chính:

- Dự đoán nhu cầu sử dụng sản phẩm tài chính để phục vụ cross-selling.
- Xây dựng chân dung/persona khách hàng dựa trên hành vi và giá trị tài chính.
- Cá nhân hóa chiến lược chăm sóc, giữ chân và phát triển khách hàng số.

Hệ thống được chia thành hai nhánh:

| Nhánh | Population | Bài toán | Mục tiêu tối ưu |
|---|---:|---|---|
| IB | 124,886 khách đã có Internet Banking | Cross-sell NBFO | Tối đa Expected Marginal Utility từ việc chọn sản phẩm và kênh tiếp cận |
| Non-IB | 127,000 khách chưa có IB | Retention/churn prevention | Tối đa EMU giữ chân trên nhóm có rủi ro rời bỏ |

Tổng ngân sách chiến dịch là 1 tỷ VND, được phân bổ sau grid search:

| Nhánh | Budget | Human-touch cap |
|---|---:|---:|
| IB | 450,000,000 VND | 6,000 lượt |
| Non-IB | 550,000,000 VND | 4,000 lượt |
| Tổng | 1,000,000,000 VND | 10,000 lượt |

### 1.1 Grid search phân bổ budget và human-touch

Trước khi chạy MILP cho từng nhánh, đội thực hiện coarse-to-refine grid search để chọn cách chia ngân sách và năng lực human-touch giữa IB và Non-IB. Objective của grid search là cân bằng hiệu quả hai bài toán:

```text
Score = 0.5 * normalized IB EMU + 0.5 * normalized Non-IB EMU
```

Cấu hình tốt nhất trong grid search:

| Stage | IB Budget | Non-IB Budget | IB Human | Non-IB Human | IB EMU | Non-IB EMU | Normalized Score |
|---|---:|---:|---:|---:|---:|---:|---:|
| refine | 450M | 550M | 6,000 | 4,000 | 5.11B | 23.84B | 0.7451 |

Cấu hình này được chọn vì ba lý do chính:

- IB cần nhiều human-touch hơn để khai thác tập khách đã có digital relationship và propensity cross-sell rõ hơn.
- Non-IB được cấp budget lớn hơn vì bài toán retention có population at-risk lớn và giá trị giữ chân cao.
- Score được chuẩn hóa giữa hai nhánh, nên phương án được chọn không chỉ tối đa hóa riêng IB hoặc riêng Non-IB mà cân bằng cả cross-sell và retention.

Lưu ý: grid search dùng để chọn cấu hình budget/human cap. EMU Non-IB chính thức trong baseline là 22.91B, được tính trên allocation cuối cùng của engine Non-IB.

Kết quả baseline:

| Metric | IB | Non-IB | Tổng |
|---|---:|---:|---:|
| Population | 124,886 | 127,000 | 251,886 |
| Cost/COGS chiến dịch | 448,655,000 | 550,000,000 | 998,655,000 |
| EMU kỳ vọng | 5,109,748,849 | 22,910,966,121 | 28,020,714,970 |
| SMS | 27,401 | 66,680 | 94,081 |
| Telesales | 5,993 | 292 | 6,285 |
| RM | 6 | 101 | 107 |
| Human-touch used | 5,999 | 393 | 6,392 |

---

## 2. Liên kết với đề bài

Đề bài yêu cầu ứng dụng Data Analytics trong ngân hàng số để phân tích hành vi khách hàng, phát hiện rủi ro và cá nhân hóa dịch vụ. Trong phạm vi Vòng 4, giải pháp của đội tập trung vào việc đưa các kết quả phân tích trước đó vào một engine ra quyết định có thể hành động.

Mapping với đề bài:

| Yêu cầu đề bài | Cách giải pháp đáp ứng |
|---|---|
| Dự đoán nhu cầu sản phẩm tài chính | IB engine sử dụng propensity/NBFO probability để xếp ưu tiên cross-sell |
| Phân tích xu hướng tiền gửi, tín dụng, thẻ | IB segmentation dựa trên loan, term deposit, card và độ sâu sản phẩm |
| Xây dựng chân dung khách hàng | Non-IB clustering tạo các persona như Ultra Saver, Traditional, Stable Senior, HV Borrower |
| Cá nhân hóa trải nghiệm và giữ chân | Mỗi segment/cluster có utility, ngưỡng và kênh tiếp cận riêng |
| Ứng dụng business logic thực tế | Engine có ràng buộc budget, human capacity, RM eligibility và one-channel-per-customer |
| Giải thích kết quả mô hình | Báo cáo trình bày công thức EMU, định nghĩa churn, threshold và lý do channel allocation |

Thay vì dùng mô hình chỉ để chấm điểm, hệ thống chuyển score thành quyết định: khách nào được tiếp cận, bằng kênh nào, với mức chi phí nào, và giá trị kỳ vọng ra sao.

---

## 3. Kiến trúc giải pháp

Decision engine gồm bốn lớp:

1. Lớp dữ liệu và feature: tổng hợp thông tin nhân khẩu học, hành vi IB, tiền gửi, tín dụng, thẻ và digital activity.
2. Lớp segment/persona: tách IB theo product profile và digital activity; tách Non-IB theo cụm giá trị tài chính và rủi ro churn.
3. Lớp utility tài chính: chuyển probability/churn risk thành Expected Marginal Utility cho từng channel.
4. Lớp tối ưu: dùng Mixed Integer Linear Programming để chọn allocation thỏa ràng buộc ngân sách và năng lực vận hành.

Kênh tiếp cận chung:

| Kênh | Cost/COGS mỗi lượt | Conversion rate giả định | Vai trò |
|---|---:|---:|---|
| SMS | 5,000 VND | 2% | Kênh scale, chi phí thấp |
| Telesales | 50,000 VND | 5% | Kênh human-touch cho khách cần tư vấn |
| RM | 2,000,000 VND | 15% | Kênh cao cấp cho nhóm VIP/high value |

### 3.1 Tiêu chí chia nhóm IB

IB segmentation được gán bằng rule-based profile để đảm bảo dễ giải thích trong vận hành. Rule được xét theo thứ tự ưu tiên từ trên xuống: nếu khách thỏa một điều kiện sớm hơn thì được gán vào segment đó.

| Segment | Ý nghĩa business | Tiêu chí gán nhóm |
|---|---|---|
| V1_HV_Borrower | Khách vay giá trị cao, có tiềm năng cross-sell và cần chăm sóc kỹ | `AVG_LOAN_AMOUNT > 500M` |
| V2_Conservative | Khách có số dư tiền gửi cao, thiên về an toàn | `AVG_TD_BALANCE > 100M` và `has_loan = 0` |
| V3_Multi_Premium | Khách đa sản phẩm, giá trị cao | `product_depth >= 3` và `AVG_TD_BALANCE > 200M` |
| N1_Active_Digital | Khách digital active, có thể tiếp cận bằng kênh số/human | `has_card = 1` và `has_loan = 1` sau khi không thuộc V1/V2/V3 |
| N2_Semi_Digital | Khách dùng số vừa phải, cần kích hoạt thêm | Nhóm còn lại: đã có tín hiệu IB nhưng chưa đủ điều kiện vào các nhóm trên |
| N3_Dormant | Khách đã có IB nhưng không phát sinh login | Với khách đã đăng ký IB trước 2020: `login_count = 0` |

Lưu ý cohort: với khách đăng ký IB trong 2020/2021, `login_count = 0` trong snapshot 2019 được xem là artifact thời gian, nên không tự động gán Dormant. Với khách đã có IB trước 2020, `login_count = 0` là tín hiệu dormant thật.

Công thức EMU chung:

```text
uplift_c(P) = 4 * P * (1 - P) * CR_c

EMU_c(P) = uplift_c(P) * (TP - FN)
           + (1 - P - uplift_c(P)) * FP
           - channel_cost_c
```

Trong đó:

- `P` là xác suất adoption với IB hoặc churn risk với Non-IB.
- `CR_c` là conversion rate của channel.
- `TP` là giá trị kỳ vọng khi tác động đúng.
- `FP` là chi phí/tổn thất khi tác động sai.
- `FN` là opportunity loss khi bỏ lỡ khách giá trị cao.
- `channel_cost_c` là COGS trực tiếp của kênh.

---

## 4. Engine IB: Cross-sell NBFO

### 4.1 Mục tiêu và population

IB engine được thiết kế cho 124,886 khách đã có Internet Banking. Mục tiêu là đề xuất sản phẩm tài chính phù hợp và chọn kênh tiếp cận tối ưu để tối đa hóa EMU.

```text
Population: 124,886 khách IB
Budget: 450,000,000 VND
Human-touch cap: 6,000 lượt Telesales/RM
Objective: maximize total Expected Marginal Utility
```

Output chính của nhánh IB là `final_allocations.csv`.

### 4.2 Segment IB

IB segmentation được xây dựng từ cohort đăng ký IB, tần suất đăng nhập và product profile. Logic này giúp engine phân biệt khách giá trị cao, khách bảo thủ, khách đa sản phẩm và khách ít tương tác số.

| Segment | Ý nghĩa business |
|---|---|
| V1_HV_Borrower | Khách vay giá trị cao, có tiềm năng cross-sell và cần chăm sóc kỹ |
| V2_Conservative | Khách có số dư tiền gửi cao, thiên về an toàn |
| V3_Multi_Premium | Khách đa sản phẩm, giá trị cao |
| N1_Active_Digital | Khách digital active, có thể tiếp cận bằng kênh số/human |
| N2_Semi_Digital | Khách dùng số vừa phải, cần kích hoạt thêm |
| N3_Dormant | Khách đã có IB nhưng không phát sinh login |

Rule quan trọng là nhóm `N3_Dormant` chỉ áp dụng cho khách đã đăng ký IB trước hoặc trong 2019 và có `login_count = 0`. Khách đăng ký IB năm 2020/2021 không bị gán dormant dựa trên snapshot 2019, vì lúc đó họ chưa có IB.

### 4.3 Utility matrix và threshold

Với IB, TP cho adoption thành công là 5,000,000 VND. Khách VIP-like có FN -30,000,000 VND vì bỏ lỡ có thể làm mất cơ hội doanh thu lớn hơn.

| Segment | TP | FP | FN |
|---|---:|---:|---:|
| V1_HV_Borrower | 5,000,000 | -50,000 | -30,000,000 |
| V2_Conservative | 5,000,000 | -50,000 | -30,000,000 |
| V3_Multi_Premium | 5,000,000 | -50,000 | -30,000,000 |
| N1_Active_Digital | 5,000,000 | -50,000 | 0 |
| N2_Semi_Digital | 5,000,000 | -50,000 | 0 |
| N3_Dormant | 5,000,000 | -50,000 | 0 |

Ngưỡng tối thiểu để channel có EMU dương:

| Segment | SMS | Telesales | RM |
|---|---:|---:|---:|
| V1_HV_Borrower | 0.0198 | 0.0144 | 0.1092 |
| V2_Conservative | 0.0198 | 0.0144 | 0.1092 |
| V3_Multi_Premium | 0.0198 | 0.0144 | 0.1092 |
| N1_Active_Digital | 0.1382 | 0.1048 | N/A |
| N2_Semi_Digital | 0.1382 | 0.1048 | N/A |
| N3_Dormant | 0.1382 | 0.1048 | N/A |

VIP-like segment có threshold thấp hơn vì FN lớn: bỏ lỡ khách giá trị cao tạo mất mát kỳ vọng cao hơn, nên engine chấp nhận tiếp cận ở mức probability thấp hơn.

### 4.4 Tối ưu MILP

IB allocation được mô hình hóa ở cấp `customer-channel`. Với mỗi khách `i` và kênh `c`, biến quyết định là:

```text
x[i,c] = 1 nếu khách i được gán kênh c
x[i,c] = 0 nếu không
```

Hàm mục tiêu vẫn là tối đa hóa tổng EMU kỳ vọng:

```text
maximize  Σ_i Σ_c EMU[i,c] * x[i,c]
```

Các constraints chính:

```text
Σ_c x[i,c] <= 1
```

Mỗi khách chỉ được nhận tối đa một kênh tiếp cận. Constraint này tránh việc một khách bị tính nhiều action cùng lúc, ví dụ vừa nhận SMS vừa được gọi Telesales.

```text
Σ_i Σ_c cost[c] * x[i,c] <= 450,000,000
```

Tổng COGS của toàn bộ chiến dịch IB không vượt ngân sách 450M VND. Vì mỗi channel có cost khác nhau, constraint này buộc mô hình cân bằng giữa số lượng khách tiếp cận và giá trị kỳ vọng của từng kênh.

```text
Σ_i (x[i,Telesales] + x[i,RM]) <= 6,000
```

Telesales và RM là hai kênh cần nhân sự xử lý, nên được gom vào cùng một giới hạn human-touch. Dù budget còn dư, mô hình vẫn không thể phân bổ quá 6,000 lượt cần can thiệp thủ công.

```text
x[i,c] = 0 nếu EMU[i,c] <= 0
x[i,c] = 0 nếu khách i không eligible với kênh c
```

Các cặp customer-channel có EMU âm hoặc không đạt threshold theo segment sẽ bị khóa về 0. Đây là constraint auto-brake, giúp engine không cố tiêu budget cho những khách/kênh có utility kỳ vọng không đủ tốt.

### 4.5 Kết quả IB

| Metric | Kết quả |
|---|---:|
| Total customers | 124,886 |
| EMU kỳ vọng | 5,109,748,849 VND |
| Cost/COGS | 448,655,000 VND |
| Budget limit | 450,000,000 VND |
| SMS | 27,401 |
| Telesales | 5,993 |
| RM | 6 |
| None/Auto-brake | 91,486 |
| Human-touch used | 5,999 / 6,000 |

Nhận xét:

- Ngân sách gần binding: engine dùng 448.655M trên 450M.
- Human-touch gần binding: 5,999 trên 6,000 lượt.
- Telesales là kênh human chủ lực của IB vì có tỷ lệ EMU/COGS tốt hơn RM trong điều kiện budget 450M.
- RM chỉ được gán 6 khách do chi phí mỗi lượt cao, chỉ phù hợp với một số trường hợp EMU rất cao.

### 4.6 Stress test IB

Kịch bản stress:

```text
FP VIP +20%
CR Telesales/RM -15%
```

| Metric | Baseline | Stress |
|---|---:|---:|
| EMU | 5,109,748,849 | 4,423,648,060 |
| Cost/COGS | 448,655,000 | 448,655,000 |
| SMS | 27,401 | 27,401 |
| Telesales | 5,993 | 5,993 |
| RM | 6 | 6 |

EMU giảm 13.4% nhưng vẫn dương lớn, cho thấy allocation IB có khả năng chịu được suy giảm conversion và tăng chi phí sai mục tiêu.

---

## 5. Engine Non-IB: Retention và churn prevention

### 5.1 Mục tiêu và population

Non-IB engine chuyển bài toán từ activation/onboarding sang retention. Lý do là nhóm chưa có IB vẫn có giá trị tài chính ngoài kênh số; mục tiêu quan trọng là giữ chân các khách có rủi ro rời bỏ trước khi đề xuất adoption IB.

```text
Population: 127,000 khách Non-IB
At-risk base: 77,824 khách
Budget: 550,000,000 VND
Human-touch cap: 4,000 lượt Telesales/RM
Objective: maximize expected retention EMU
```

Output chính của nhánh Non-IB là `final_allocations_nonib.csv`.

### 5.2 Định nghĩa churn

Churn được xác định dựa trên diễn biến tài chính hàng tháng trong năm 2019, bao gồm tiền gửi, dư nợ vay và sử dụng thẻ. Mục tiêu không phải là gán nhãn churn cho toàn bộ khách Non-IB, mà là nhận diện những khách từng có quan hệ tài chính với ngân hàng nhưng đến cuối năm có dấu hiệu suy giảm hoặc rời bỏ.

Trước hết, engine xác định nhóm at-risk: đây là các khách từng có hoạt động tài chính trong giai đoạn Jan-Sep 2019. Nhóm này được xem là có quan hệ thực tế với ngân hàng, nên nếu họ biến mất hoặc giảm mạnh giá trị tài chính trong Q4 thì cần được đưa vào bài toán retention.

Trong nhóm at-risk, churn được chia thành hai mức:

- Hard churn: khách gần như không còn hoạt động trong Q4. Đây là tín hiệu rời bỏ rõ nhất, vì khách đã từng active trước đó nhưng đến cuối năm gần như không còn giá trị tài chính hoặc hoạt động thẻ.
- Runoff risk: khách vẫn còn hoạt động trong Q4 nhưng giá trị tài chính giảm rất mạnh so với tháng 9. Nhóm này chưa được xem là churn hoàn toàn, vì với khách gửi tiết kiệm/term deposit, số dư giảm có thể đến từ đáo hạn hoặc tái cơ cấu kỳ hạn.

Vì runoff risk không chắc chắn là churn thật sự, engine chỉ tính nhóm này với trọng số 30% khi ước lượng rủi ro churn hiệu dụng. Sau đó, rủi ro churn được tổng hợp theo cluster/persona để tạo `P_CHURN_cluster`. Giá trị này đại diện cho mức rủi ro của cả cụm, và được dùng để tính EMU cho từng kênh tiếp cận.

### 5.3 Persona và churn risk

| Cluster | At Risk | Hard Churn | Runoff Risk | Effective Churn | P Churn |
|---|---:|---:|---:|---:|---:|
| C3_Ultra_Saver | 393 | 229 | 25 | 236.5 | 60.18% |
| C5_HV_Saver | 10,542 | 1,325 | 1,618 | 1,810.4 | 17.17% |
| C4_Multi_Saver | 591 | 21 | 260 | 99.0 | 16.75% |
| C2_Senior_HV | 511 | 44 | 103 | 74.9 | 14.66% |
| C1_HV_Traditional | 470 | 12 | 80 | 36.0 | 7.66% |
| C0_Traditional | 59,861 | 2,198 | 5,775 | 3,930.5 | 6.57% |
| C6_Stable_Senior | 4,860 | 47 | 541 | 209.3 | 4.31% |
| C7_HV_Borrower | 596 | 0 | 70 | 21.0 | 3.52% |
| P0_Dormant | 0 | 0 | 0 | 0.0 | 0.10% |

Insight chính:

- `C3_Ultra_Saver` có churn risk cao nhất và được ưu tiên human-touch/RM.
- `C0_Traditional` có churn rate thấp hơn nhưng population rất lớn, nên SMS tạo tổng EMU cao.
- `C4_Multi_Saver` có runoff cao, nhưng sau điều chỉnh 30% thì risk thực tế ở mức vừa phải.
- `P0_Dormant` không nằm trong paid retention target.

### 5.4 Utility matrix và channel eligibility

Với Non-IB retention, TP được đặt cố định 50,000,000 VND cho mỗi trường hợp giữ chân thành công.

```text
TP = +50,000,000 VND
FP = -50,000 VND
FN = -30,000,000 VND cho VIP-like clusters
FN = 0 cho normal clusters
```

VIP-like clusters:

```text
C1_HV_Traditional
C2_Senior_HV
C3_Ultra_Saver
C4_Multi_Saver
C5_HV_Saver
C6_Stable_Senior
C7_HV_Borrower
```

Threshold:

| Cluster | SMS | Telesales | RM |
|---|---:|---:|---:|
| C3_Ultra_Saver | 0.0087 | 0.0063 | 0.0447 |
| C5_HV_Saver | 0.0087 | 0.0063 | 0.0447 |
| C4_Multi_Saver | 0.0087 | 0.0063 | 0.0447 |
| C2_Senior_HV | 0.0087 | 0.0063 | 0.0447 |
| C1_HV_Traditional | 0.0087 | 0.0063 | 0.0447 |
| C0_Traditional | 0.0138 | 0.0101 | N/A |
| C6_Stable_Senior | 0.0087 | 0.0063 | 0.0447 |
| C7_HV_Borrower | 0.0087 | 0.0063 | 0.0447 |
| P0_Dormant | 0.0138 | 0.0101 | N/A |

### 5.5 Tối ưu MILP

Non-IB được tối ưu ở cấp `cluster-channel` vì các khách trong cùng cluster có cùng risk/utility theo kênh. Biến quyết định là:

```text
y[g,c] = số khách cluster g được gán kênh c
```

Hàm mục tiêu là tối đa hóa EMU giữ chân:

```text
maximize  Σ_g Σ_c EMU[g,c] * y[g,c]
```

Các constraints chính:

```text
Σ_c y[g,c] <= AT_RISK_CUSTOMERS[g]
```

Số khách được phân bổ action trong mỗi cluster không vượt quá số khách at-risk của cluster đó. Constraint này giữ allocation nằm trong tập khách thật sự cần can thiệp.

```text
Σ_g Σ_c cost[c] * y[g,c] <= 550,000,000
```

Tổng COGS của chiến dịch Non-IB không vượt ngân sách 550M VND.

```text
Σ_g (y[g,Telesales] + y[g,RM]) <= 4,000
```

Telesales và RM dùng chung giới hạn human-touch 4,000 lượt. Constraint này phản ánh năng lực vận hành, không chỉ giới hạn chi phí.

```text
0 <= y[g,c] <= AT_RISK_CUSTOMERS[g]
y[g,c] là số nguyên
y[g,c] = 0 nếu EMU[g,c] <= 0 hoặc channel không eligible
```

Vì `y[g,c]` là số khách, biến phải là số nguyên và không âm. Các channel không có EMU dương hoặc không phù hợp với cluster sẽ không được chọn.

```text
Σ_g y[g,RM] >= MIN_RM_ALLOCATIONS
```

Non-IB có thêm constraint tối thiểu cho RM để đảm bảo nhóm khách high-value vẫn có một lượng chăm sóc chuyên sâu. Nếu không có constraint này, mô hình có thể nghiêng quá mạnh về SMS/Telesales do RM có chi phí cao.

### 5.6 Kết quả Non-IB

| Metric | Kết quả |
|---|---:|
| Population | 127,000 |
| At-risk base | 77,824 |
| Hard churn proxy | 3,876 |
| Runoff risk proxy | 8,472 |
| Effective churn score | 6,417.6 |
| SMS allocated | 66,680 |
| Telesales | 292 |
| RM | 101 |
| None | 59,927 |
| Cost/COGS | 550,000,000 VND |
| Campaign EMU | 22,910,966,121 VND |
| Expected retained customers | 422.82 |
| CLV at risk targeted | 116,589,517,118 VND |
| Human-touch used | 393 / 4,000 |

Budget binding:

```text
66,680 * 5,000 + 292 * 50,000 + 101 * 2,000,000 = 550,000,000 VND
```

### 5.7 Allocation theo cluster

| Cluster | N | SMS | Telesales | RM | None | Cost | EMU |
|---|---:|---:|---:|---:|---:|---:|---:|
| C3_Ultra_Saver | 912 | 0 | 292 | 101 | 519 | 216.60M | 2,058.4M |
| C5_HV_Saver | 13,178 | 10,542 | 0 | 0 | 2,636 | 52.71M | 9,113.5M |
| C4_Multi_Saver | 800 | 591 | 0 | 0 | 209 | 2.96M | 500.2M |
| C2_Senior_HV | 562 | 511 | 0 | 0 | 51 | 2.56M | 385.0M |
| C1_HV_Traditional | 493 | 470 | 0 | 0 | 23 | 2.35M | 188.8M |
| C0_Traditional | 80,038 | 49,706 | 0 | 0 | 30,332 | 248.53M | 9,639.2M |
| C6_Stable_Senior | 5,794 | 4,860 | 0 | 0 | 934 | 24.30M | 1,025.8M |
| C7_HV_Borrower | 653 | 0 | 0 | 0 | 653 | 0 | 0 |
| P0_Dormant | 24,570 | 0 | 0 | 0 | 24,570 | 0 | 0 |

### 5.8 Stress test Non-IB

Stress test Non-IB được chạy lại MILP riêng cho từng kịch bản, không chỉ re-score allocation baseline.

| Scenario | COGS | EMU | Expected retained | SMS | Telesales | RM | Incremental ROI |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 550.00M | 22.91B | 422.82 | 66,680 | 292 | 101 | 40.66x |
| Adverse CR/FP re-optimized | 550.00M | 21.96B | 418.54 | 66,680 | 292 | 101 | 38.93x |
| Budget cut -20% | 440.00M | 18.64B | 314.85 | 44,680 | 292 | 101 | 41.37x |
| SMS CR -25% | 550.00M | 18.14B | 276.58 | 30,610 | 3,899 | 101 | 31.97x |

Insight chính: khi SMS conversion giảm 25%, chiến lược tối ưu mới chuyển bớt scale từ SMS sang Telesales. Khi budget bị cắt 20%, solver cắt SMS trước vì đây là channel scale linh hoạt nhất, trong khi vẫn giữ RM minimum cho nhóm VIP/high-value.

---

## 6. Dự báo revenue và cost

Trong framework này, `TP`, `FN`, `FP` và `channel_cost` được quy đổi thành Expected Marginal Utility. Có thể đọc kết quả theo góc nhìn revenue và COGS như sau:

- Revenue/giá trị kỳ vọng: giá trị tài chính tạo ra từ adoption thành công hoặc giữ chân thành công, đã điều chỉnh theo xác suất uplift.
- COGS: chi phí trực tiếp của kênh tiếp cận, gồm SMS, Telesales và RM.
- EMU: lợi ích biên kỳ vọng sau khi trừ chi phí kênh và điều chỉnh chi phí sai mục tiêu/bỏ lỡ.

Bảng tổng hợp:

| Nhánh | Giá trị kinh doanh chính | Cost | EMU kỳ vọng |
|---|---|---:|---:|
| IB | Cross-sell NBFO trên khách đã có IB | 448.655M | 5.110B |
| Non-IB | Giữ chân khách at-risk, bảo vệ CLV | 550.000M | 22.911B |
| Tổng | Cross-sell + retention | 998.655M | 28.021B |

Expected retained customers của Non-IB là 422.82 khách. Với IB, output là allocation cross-sell theo xác suất sản phẩm và channel; giá trị kỳ vọng được phản ánh trực tiếp trong EMU 5.11B.

### 6.1 KPI bắt buộc cho Pitch Deck

CAC được tính bằng `COGS / expected conversions`. Incremental ROI được tính bằng `(EMU - COGS) / COGS`.

| Nhánh | COGS | EMU | Expected conversions | CAC | Incremental ROI | EMU/COGS |
|---|---:|---:|---:|---:|---:|---:|
| IB | 448.655M | 5.110B | 629.04 | 0.713M | 10.39x | 11.39x |
| Non-IB | 550.000M | 22.911B | 422.82 | 1.301M | 40.66x | 41.66x |
| Tổng | 998.655M | 28.021B | 1,051.86 | 0.949M | 27.06x | 28.06x |

### 6.2 Dự phóng P&L theo thời gian

Dự phóng dùng rollout 12 tháng: COGS được ghi nhận theo nhịp triển khai chiến dịch, còn EMU được ghi nhận trễ hơn theo thời điểm conversion/retention phát sinh giá trị. Đây là projection vận hành, không thay đổi allocation baseline.

| Quarter | COGS | Incremental EMU | Net incremental profit | Cumulative profit |
|---|---:|---:|---:|---:|
| 2020Q1 | 449.39M | 5.04B | 4.59B | 4.59B |
| 2020Q2 | 349.53M | 8.13B | 7.78B | 12.37B |
| 2020Q3 | 129.83M | 7.85B | 7.72B | 20.09B |
| 2020Q4 | 69.91M | 7.01B | 6.94B | 27.02B |

Monthly detail được xuất ở `pnl_projection.csv`.

Tỷ lệ EMU/COGS:

| Nhánh | EMU/COGS |
|---|---:|
| IB | 11.39x |
| Non-IB | 41.66x |
| Tổng | 28.06x |

Diễn giải business:

- Non-IB tạo EMU/COGS cao hơn vì TP retention được đặt 50M và tập trung vào nhóm at-risk.
- IB vẫn cần thiết vì đây là bài toán tăng trưởng sản phẩm và khai thác relationship hiện hữu.
- COGS gần hết ngân sách nhưng vẫn nằm trong cap, cho thấy solver sử dụng ngân sách hiệu quả.

---

## 7. Ý nghĩa kinh doanh

### 8.1 Từ prediction sang decision

Điểm mạnh của giải pháp là không dùng model score như một bảng xếp hạng đơn giản. Score được đưa vào utility function, sau đó qua MILP để tạo allocation có thể triển khai. Cách làm này gần với vận hành thực tế hơn vì mọi ngân hàng đều có giới hạn ngân sách, nhân sự và channel.

### 8.2 Cá nhân hóa theo giá trị và rủi ro

Khách không được đối xử giống nhau:

- Khách IB giá trị cao được ưu tiên khi probability dù thấp vì opportunity loss lớn.
- Khách Non-IB có churn risk cao được ưu tiên retention trước activation.
- Khách population lớn nhưng risk vừa phải được tiếp cận bằng SMS để tối ưu scale.
- RM được dành cho cụm có giá trị/rủi ro cao và có yêu cầu business coverage.

### 8.3 Giảm lãng phí chiến dịch

Engine có cơ chế auto-brake: nếu EMU của một channel không dương, khách không được allocate vào channel đó. Điều này giúp tránh việc gửi chiến dịch đại trà gây tốn chi phí và giảm trải nghiệm khách hàng.

### 8.4 Minh bạch và giải thích được

Mỗi quyết định có thể giải thích bằng:

- Segment/cluster của khách.
- Probability hoặc churn risk.
- Threshold của channel.
- EMU kỳ vọng.
- Ràng buộc budget/human cap.

Đây là yếu tố quan trọng để đưa mô hình AI/Data Analytics vào quy trình kinh doanh ngân hàng.

---

## 8. Khuyến nghị triển khai

### 9.1 Cho IB

- Dùng Telesales làm kênh human-touch chủ lực cho các khách có propensity cao.
- Duy trì SMS cho nhóm có EMU dương nhưng không cần human-touch.
- Giới hạn RM cho khách VIP có expected value rất cao để tránh ăn mòn budget.
- Theo dõi actual conversion theo segment để cập nhật lại CR trong EMU.

### 9.2 Cho Non-IB

- Ưu tiên retention trước activation với khách Non-IB at-risk.
- Tập trung RM/Telesales vào `C3_Ultra_Saver`, vì đây là cụm có churn risk cao và giá trị giữ chân lớn.
- Dùng SMS cho `C0_Traditional`, `C5_HV_Saver`, `C6_Stable_Senior` để tối ưu scale.
- Tách riêng runoff và hard churn trong dashboard vận hành để tránh phản ứng quá mức với biến động tiền gửi ngắn hạn.

### 9.3 Cho quản trị mô hình

- Cập nhật conversion rate và cost mỗi tháng/quý dựa trên kết quả thực chiến.
- Theo dõi EMU realized so với EMU expected.
- Chạy lại MILP khi có thay đổi về budget, headcount Telesales/RM hoặc mục tiêu sản phẩm.
- Duy trì stress test bằng cách re-optimize allocation dưới từng scenario, không chỉ re-score allocation baseline.

---

## 9. Output và file liên quan

| File | Vai trò |
|---|---|
| `Vong4_Decision_Engine_Report.md` | Báo cáo chi tiết engine IB |
| `Vong4_Decision_Engine_NonIB_Report.md` | Báo cáo chi tiết engine Non-IB retention |
| `decision_engine.py` | Logic allocation IB |
| `decision_engine_nonib.py` | Logic allocation Non-IB |
| `business_kpi_scenarios.py` | Tính CAC, Incremental ROI, P&L projection và stress re-optimization |
| `decision_config.py` | Cấu hình channel, cost, conversion và utility |
| `gridsearch_budget_human_milp.py` | Grid search phân bổ budget/human cap |
| `calculate_thresholds.py` | Tính threshold EMU dương |
| `cluster_ib.py` | Gán segment IB |
| `IB_clustering.md` | Mô tả segmentation IB |

Output customer-level nằm ở folder gốc project:

| File | Nội dung |
|---|---|
| `final_allocations.csv` | Allocation IB |
| `final_allocations_nonib.csv` | Allocation Non-IB |
| `thresholds.csv` | Threshold IB |
| `thresholds_nonib.csv` | Threshold Non-IB |
| `nonib_retention_cluster_summary.csv` | Tổng hợp retention theo cluster |
| `business_kpis.csv` | CAC, Incremental ROI và EMU/COGS |
| `pnl_projection.csv` | Dự phóng P&L monthly/quarterly |
| `stress_reoptimized_nonib.csv` | Kịch bản stress Non-IB đã re-optimize allocation |

---

## 10. Kết luận

Giải pháp Vòng 4 của GCON xây dựng một decision engine hoàn chỉnh cho ngân hàng số, kết hợp NBFO, persona segmentation, churn prevention và tối ưu hóa nguồn lực. Điểm cốt lõi là chuyển prediction thành action: mỗi khách được đánh giá theo probability/risk, giá trị tài chính, chi phí kênh và ràng buộc vận hành.

Với ngân sách gần 1 tỷ VND, engine đề xuất tiếp cận 100,473 lượt khách qua SMS, Telesales và RM, tạo EMU baseline 28.02 tỷ VND. Kết quả này cho thấy giải pháp có tính ứng dụng thực tế: vừa tăng trưởng cross-sell trên khách IB, vừa bảo vệ giá trị khách hàng Non-IB có rủi ro rời bỏ, đồng thời đảm bảo minh bạch, giải thích được và kiểm soát chi phí.
