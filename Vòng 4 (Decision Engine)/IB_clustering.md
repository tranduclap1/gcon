# IB Customer Segmentation — Rule-based

**Dataset:** `cleaned_data/base_full.parquet` — IB customers only (`is_ib == 1`)
**Tổng IB:** 158,929 KH
**Scope:** gồm cohort 2019 (`IB_REGISTER_DATE.year <= 2019`) và cohort 2020/21 (`IB_REGISTER_DATE.year in [2020, 2021]`)
**Nguồn cohort trong code:** `NBFO_IB/processed_data/gcon_customer_month_clean.parquet` được tạo bởi `NBFO_IB/EDA_Feature_engineering.ipynb`. File model `gcon_model_input.parquet` không giữ `IB_REGISTER_DATE` vì cột này bị drop khỏi feature set.
**Phương pháp:** Rule-based priority cascade (không dùng ML)

---

## Lý do chọn Rule-based

Hai mục đích cần segmentation này:

1. **Map Non-IB → IB** để ước lượng purchase rate sau khi KH Non-IB lên IB (dùng Euclidean distance giữa centroids). Rule-based dùng chung features với Non-IB cluster (`has_loan`, `has_td`, `has_card`, `AVG_LOAN_AMOUNT`, `AVG_TD_BALANCE`, `product_depth`) → có thể tính khoảng cách trực tiếp.

2. **FUM Normal/VIP split** để xác định nhóm nào có False Negative cost thực sự. Ranh giới VIP là business logic (tài sản lớn, rủi ro mất KH) — không phải quyết định thống kê.

GMM loại bỏ vì dùng features khác (`TRANS_AMOUNT_SUM`, `TRANS_NO_SUM`, `AGE`) không overlap với Non-IB cluster, và ranh giới VIP/Normal không tường minh.

---

## Phân loại (Priority Cascade)

KH rơi vào segment đầu tiên thỏa điều kiện theo thứ tự. Hai cohort dùng rule khác nhau:

**Cohort 2019** — dùng `login_count` làm tín hiệu dormant thật:

```python
def assign_seg_2019(r):
    if r['login_count'] == 0:                                           return 'N3_Dormant'
    if r['AVG_LOAN_AMOUNT'] > 500_000_000:                              return 'V1_HV_Borrower'
    if r['AVG_TD_BALANCE'] > 100_000_000 and r['has_loan'] == 0:        return 'V2_Conservative'
    if r['product_depth'] >= 3 and r['AVG_TD_BALANCE'] > 200_000_000:  return 'V3_Multi_Premium'
    if r['has_card'] == 1 and r['has_loan'] == 1:                       return 'N1_Active_Digital'
    return 'N2_Semi_Digital'
```

**Cohort 2020/21** — bỏ `login_count` vì tại snapshot 2019 họ chưa đăng ký IB, nên `login_count = 0` là data artifact:

```python
def assign_seg_2020_21(r):
    if r['AVG_LOAN_AMOUNT'] > 500_000_000:                              return 'V1_HV_Borrower'
    if r['AVG_TD_BALANCE'] > 100_000_000 and r['has_loan'] == 0:        return 'V2_Conservative'
    if r['product_depth'] >= 3 and r['AVG_TD_BALANCE'] > 200_000_000:  return 'V3_Multi_Premium'
    if r['has_card'] == 1 and r['has_loan'] == 1:                       return 'N1_Active_Digital'
    return 'N2_Semi_Digital'
```

**Features sử dụng:** `login_count`, `AVG_LOAN_AMOUNT`, `AVG_TD_BALANCE`, `has_loan`, `product_depth`, `has_card`

---

## VIP vs Normal vs Onboarding — Logic phân chia

Sau khi phân 6 segment, toàn bộ IB được chia thành 3 nhóm xử lý khác nhau:

```
IB (158,929 KH)
├── VIP (3 segment) — FN có chi phí thực
│   ├── V1 HV Borrower     10,309 KH   6.5%
│   ├── V2 Conservative     5,597 KH   3.5%
│   └── V3 Multi Premium      798 KH   0.5%
│
├── Normal (2 segment) — FN ≈ 0
│   ├── N1 Active Digital  11,485 KH   7.2%
│   └── N2 Semi Digital    72,868 KH  45.8%
│
└── Onboarding (1 segment) — KHÔNG đánh NBFO trực tiếp
    └── N3 Dormant         57,872 KH  36.4%
```

### Tại sao VIP khác Normal?

Ranh giới dựa trên chi phí của **False Negative** — tức hậu quả của việc bỏ lỡ một KH:

**VIP — FN có giá trị thực:**
- V1: Loan > 500M → bỏ lỡ = mất cơ hội cross-sell trên KH có dư nợ lớn
- V2: TD > 100M, không vay → bỏ lỡ = mất KH tiết kiệm cao giá trị + rủi ro họ rút tiền nếu tiếp cận sai (*Sleeping Dog*)
- V3: ≥ 3 sản phẩm + TD > 200M → bỏ lỡ = mất KH đa sản phẩm có buy rate cao nhất (11.17%)

**Normal — FN ≈ 0:**
- N1/N2: Tài sản trung bình, bỏ lỡ chỉ mất cơ hội cross-sell nhỏ → không đáng kể trong tổng thể

Sự khác biệt này ảnh hưởng trực tiếp đến threshold quyết định tiếp cận:

| Nhóm | FN cost | FP cost | Threshold θ* | Channel |
|---|---|---|---|---|
| Normal N1 | ≈ 0 | -50K | (50K + 5K) / 5.05M = **1.09%** | SMS |
| Normal N2 | ≈ 0 | -50K | (50K + 50K) / 5.05M = **1.98%** | Call |
| VIP V1/V3 | -30M | -10M | (10M + 2M) / 45M = **26.67%** | RM |
| VIP V2 | -30M | -50M | (50M + 2M) / 85M = **61.18%** | RM |

VIP threshold cao hơn Normal ~20–30 lần vì FP cost lớn hơn — chỉ tiếp cận khi rất chắc chắn KH sẽ mua.

### Tại sao N3 không đánh NBFO?

N3 có `login_count = 0` — đã đăng ký IB nhưng **chưa bao giờ login**. Không thể cross-sell sản phẩm qua kênh digital cho người chưa mở app lần nào.

N3 được xử lý theo 2 bước tách biệt:

```
N3 Dormant (57,872 KH — 43.1% IB)
       ↓
  Bước 1 — Task 2: IB Onboarding
  Kích hoạt IB trước (SMS nhắc login, hướng dẫn dùng app)
       ↓
  Sau khi active → chuyển sang N1 hoặc N2
       ↓
  Bước 2 — Task 1: NBFO cross-sell bình thường
```

57,872 KH chiếm 43% toàn bộ IB 2019 mà chưa được khai thác — đây là pipeline tiềm năng lớn nhất nếu onboarding thành công. Budget Task 2 tính riêng, không gộp vào NBFO.

---

## 6 Segments — Chi tiết

### VIP Group (FN có chi phí thực)

---

#### V3 — Multi Premium
**N=798 | 0.5% IB | NBFO Buy Rate: 11.17%**

**Điều kiện:** `product_depth ≥ 3` VÀ `AVG_TD_BALANCE > 200M`

| | Median | Mean | P25 | P75 | P95 |
|---|---|---|---|---|---|
| Tuổi | 36 | 38.1 | 31 | 44 | — |
| AVG_LOAN_AMOUNT | 18M | — | — | 60M | 362M |
| AVG_TD_BALANCE | 551M | — | — | 1,095M | 3,221M |
| product_depth | 4.0 | 3.85 | — | — | — |
| login_count | 42 | 76.9 | — | — | — |
| trans_count | 5 | 21.8 | — | — | — |
| trans_amount/txn | 4.3M | — | — | — | — |
| digital_ready_score | 5.0 | 4.41 | — | — | — |
| bank_engagement_months | 6 | 6.7 | — | — | — |

**Product ownership:** has_loan=100% | has_td=100% | has_card=92%
**Demographics:** Nữ 59% | Tuổi 36 | Staff NH 2.3%

**Hành vi:**
- Khách hàng đa sản phẩm cao cấp — sở hữu đủ loan + TD + thẻ cùng lúc
- TD median 551M nhưng loan chỉ 18M → vay nhỏ để tối ưu tài chính, không phải nhu cầu vốn
- Digital active nhất toàn IB: login 42 lần, 10 loại hoạt động, digital_ready_score = 5.0
- Giao dịch ít lần (5 lần) nhưng giá trị lớn (4.3M/giao dịch)
- Tỷ lệ nữ cao nhất (59%) — có thể là người quản lý tài sản gia đình

**FUM role:** VIP — FN cost = -30M (bỏ lỡ KH tiềm năng cao nhất, buy rate 11.17%)
**Channel khuyến nghị:** RM

---

#### V1 — HV Borrower
**N=10,309 | 6.5% IB | NBFO Buy Rate: 4.56%**

**Điều kiện:** `AVG_LOAN_AMOUNT > 500M`

| | Median | Mean | P25 | P75 | P95 |
|---|---|---|---|---|---|
| Tuổi | 35 | 36.1 | 30 | 41 | — |
| AVG_LOAN_AMOUNT | 879M | — | — | 1,413M | 2,955M |
| AVG_TD_BALANCE | 0M | — | — | 0M | 1M |
| product_depth | 3.0 | 2.69 | — | — | — |
| login_count | 14 | 32.2 | — | — | — |
| trans_count | 0 | 9.6 | — | — | — |
| digital_ready_score | 3.0 | 3.12 | — | — | — |
| bank_engagement_months | 5 | 5.8 | — | — | — |

**Product ownership:** has_loan=100% | has_td=5.2% | has_card=64.5%
**Demographics:** Nam 65% | Tuổi 35 | Staff NH 0.7%

**Hành vi:**
- Tập trung hoàn toàn vào vay — loan median 879M, gần như không có TD
- Digital thấp: login 14 lần, trans_count median = 0 → xử lý loan offline qua branch
- digital_engaged_months chỉ 3 tháng — dùng IB không thường xuyên
- Nam chiếm ưu thế (65%) — nhóm có tỷ lệ nam cao nhất
- Có thể là chủ doanh nghiệp nhỏ hoặc đầu tư bất động sản

**FUM role:** VIP — FN cost = -30M (CLV lớn từ dư nợ vay)
**Channel khuyến nghị:** RM

---

#### V2 — Conservative Saver (Sleeping Dog)
**N=5,597 | 3.5% IB | NBFO Buy Rate: 3.92%**

**Điều kiện:** `AVG_TD_BALANCE > 100M` VÀ `has_loan = 0`

| | Median | Mean | P25 | P75 | P95 |
|---|---|---|---|---|---|
| Tuổi | 35 | 37.5 | 30 | 43 | — |
| AVG_LOAN_AMOUNT | 0M | — | — | 0M | 0M |
| AVG_TD_BALANCE | 350M | — | — | 801M | 3,093M |
| product_depth | 2.0 | 2.36 | — | — | — |
| login_count | 19 | 42.3 | — | — | — |
| trans_count | 0 | 10.0 | — | — | — |
| digital_ready_score | 3.0 | 2.78 | — | — | — |
| bank_engagement_months | 5 | 5.8 | — | — | — |

**Product ownership:** has_loan=0% | has_td=100% | has_card=43.8%
**Demographics:** Nữ 61% | Tuổi 35 | Staff NH 0.9%

**Hành vi:**
- Tài sản lớn (TD 350M median) nhưng hoàn toàn không vay — rủi ro hất vốn sang NH khác nếu tiếp cận sai
- Ít giao dịch: trans_count = 0 median — tiền "ngủ" trong TD, không ra vào
- Login 19 lần — dùng IB chủ yếu để xem số dư
- Nữ 61% — có thể là người giữ tiền gia đình, ưu tiên an toàn
- digital_engaged_months 4 tháng — dùng IB nhưng không giao dịch nhiều

**FUM role:** VIP — FP cost đặc biệt cao (-50M "Sleeping Dog risk"): tiếp cận sai → KH rút TD
**Channel khuyến nghị:** RM với approach thận trọng

---

### Normal Group (FN ≈ 0)

---

#### N1 — Active Digital
**N=11,485 | 7.2% IB | NBFO Buy Rate: 6.81%**

**Điều kiện:** `has_card = 1` VÀ `has_loan = 1`

| | Median | Mean | P25 | P75 | P95 |
|---|---|---|---|---|---|
| Tuổi | 32 | 33.2 | 28 | 37 | — |
| AVG_LOAN_AMOUNT | 60M | — | — | 295M | 467M |
| AVG_TD_BALANCE | 0M | — | — | 0M | 52M |
| product_depth | 3.0 | 2.98 | — | — | — |
| login_count | 28 | 59.2 | — | — | — |
| trans_count | 3 | 19.0 | — | — | — |
| trans_amount/txn | 1.2M | — | — | — | — |
| digital_ready_score | 4.0 | 3.56 | — | — | — |
| bank_engagement_months | 6 | 6.4 | — | — | — |

**Product ownership:** has_loan=100% | has_td=10.1% | has_card=100%
**Demographics:** Nam 63% | Tuổi 32 | Staff NH 6.4% (cao nhất)

**Hành vi:**
- Digital native: login 28 lần, 10 loại hoạt động, digital_ready_score = 4.0
- Giao dịch thường xuyên (3 lần median, 1.2M/giao dịch) — chuyển khoản, thanh toán hàng ngày
- Loan nhỏ (60M) — tín dụng tiêu dùng cá nhân
- Gắn bó với NH lâu (6 tháng) — KH trung thành
- Staff NH 6.4% cao bất thường → nhiều nhân viên NH nằm trong nhóm này

**FUM role:** Normal — FN ≈ 0, threshold SMS thấp (θ* ≈ 1.1%)
**Channel khuyến nghị:** SMS

---

#### N2 — Semi Digital
**N=72,868 | 45.8% IB | NBFO Buy Rate: 2.13%**

**Điều kiện:** Còn lại (đã login ít nhất 1 lần, không fit V1/V2/V3/N1)

| | Median | Mean | P25 | P75 | P95 |
|---|---|---|---|---|---|
| Tuổi | 27 | 28.8 | 23 | 32 | — |
| AVG_LOAN_AMOUNT | 0M | — | — | 0M | 245M |
| AVG_TD_BALANCE | 0M | — | — | 0M | 4M |
| product_depth | 2.0 | 1.92 | — | — | — |
| login_count | 23 | 47.5 | — | — | — |
| trans_count | 5 | 20.1 | — | — | — |
| trans_amount/txn | 0.6M | — | — | — | — |
| outflow_amount | 3.5M | — | — | — | — |
| digital_ready_score | 2.0 | 2.16 | — | — | — |
| bank_engagement_months | 5 | 5.3 | — | — | — |

**Product ownership:** has_loan=7.3% | has_td=5.7% | has_card=81.6%
**Demographics:** Nam 58% | Tuổi 27 (trẻ nhất IB) | Staff NH 1.8%

**Hành vi:**
- Trẻ nhất toàn IB (27 tuổi) — có thể là sinh viên, người mới đi làm
- Chủ yếu chỉ có thẻ (81.6%), chưa dùng loan hay TD
- Login 23 lần — dùng IB đều nhưng giao dịch nhỏ (0.6M/lần)
- digital_ready_score thấp (2.0) — chưa khai thác hết tiềm năng IB
- product_depth 2.0 — mới ở giai đoạn đầu gắn kết với NH

**FUM role:** Normal — FN ≈ 0, threshold Call (θ* ≈ 2.0%)
**Channel khuyến nghị:** Call (filter bằng propensity score)

---

#### N3 — Dormant IB
**N=57,872 | 36.4% IB | Buy Rate lịch sử: 5.13%***

**Điều kiện:** `login_count = 0` — chưa bao giờ login IB

| | Median | Mean | P25 | P75 | P95 |
|---|---|---|---|---|---|
| Tuổi | 32 | 33.7 | 27 | 39 | — |
| AVG_LOAN_AMOUNT | 3M | — | — | 255M | 844M |
| AVG_TD_BALANCE | 0M | — | — | 0M | 0M |
| product_depth | 2.0 | 1.90 | — | — | — |
| login_count | 0 | 0 | — | — | — |
| trans_count | 0 | 0.1 | — | — | — |
| digital_ready_score | 2.0 | 2.22 | — | — | — |
| bank_engagement_months | 6 | 5.8 | — | — | — |

**Product ownership:** has_loan=55.6% | has_td=4.2% | has_card=69.5%
**Demographics:** Nam 58% | Tuổi 32 | Staff NH 0.1%

**Hành vi:**
- Chưa bao giờ login IB — đăng ký nhưng không kích hoạt
- 55.6% có loan nhưng xử lý hoàn toàn offline — nhiều KH đăng ký IB qua AUTO-JOB (credit card issuance trigger)
- Zero giao dịch digital, zero activity
- Buy rate lịch sử 5.13% là từ sản phẩm offline, không phản ánh khả năng cross-sell qua IB

*Buy rate không dùng cho NBFO targeting vì KH chưa active IB.
**FUM role:** Loại khỏi NBFO cross-sell — thuộc IB Onboarding (Task 2)
**Channel khuyến nghị:** SMS kích hoạt IB trước

---

## Summary Table

| Segment | N | %IB | Tuổi | Nữ% | Loan (med) | TD (med) | Login (med) | Buy Rate | FUM Role |
|---|---|---|---|---|---|---|---|---|---|
| V3 Multi Premium | 798 | 0.5% | 36 | 59% | 18M | 551M | 42 | **11.17%** | VIP |
| V1 HV Borrower | 10,309 | 6.5% | 35 | 35% | 879M | 0M | 14 | 4.56% | VIP |
| V2 Conservative | 5,597 | 3.5% | 35 | 61% | 0M | 350M | 19 | 3.92% | VIP (Sleeping Dog) |
| N1 Active Digital | 11,485 | 7.2% | 32 | 37% | 60M | 0M | 28 | **6.81%** | Normal |
| N2 Semi Digital | 72,868 | 45.8% | 27 | 42% | 0M | 0M | 23 | 2.13% | Normal |
| N3 Dormant | 57,872 | 36.4% | 32 | 42% | 3M | 0M | 0 | 5.13%* | Onboarding (Task 2) |

*N3 buy rate từ lịch sử offline — không dùng cho NBFO targeting.

---

## Vai trò trong pipeline

### Task 1 — NBFO Cross-sell (IB)
- **Target:** V3, V1, V2, N1, N2 (loại N3)
- **Threshold** tính theo FUM từ `plan_code.py`:
  - Normal N1: θ* = (50K + 5K) / 5,050,000 = **1.09%** — SMS
  - Normal N2: θ* = (50K + 50K) / 5,050,000 = **1.98%** — Call
  - VIP V1/V3: θ* = (10M + 2M) / 45,000,000 = **26.67%** — RM
  - VIP V2: θ* = (50M + 2M) / 85,000,000 = **61.18%** — RM (elevated FP cost)

### Task 1 — NBFO Cross-sell (Non-IB)
- **Dùng:** Purchase rate của từng IB segment làm proxy CLV cho Non-IB cluster
- **Phương pháp:** Euclidean distance (normalized) giữa Non-IB cluster centroid và IB segment centroid
- **Features:** `has_loan`, `has_td`, `has_card`, `AVG_LOAN_AMOUNT`, `AVG_TD_BALANCE`, `product_depth`
- **Kết quả:** Xem `verify_mapping.py` và bảng CLV per Non-IB cluster

### Task 2 — IB Onboarding
- **Target:** N3 Dormant (57,872 KH — 43.1% IB 2019 chưa active)
- **Xử lý riêng:** Không tính vào NBFO budget
