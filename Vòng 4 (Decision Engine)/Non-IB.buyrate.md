# Non-IB Buy Rate Estimation

**Mục tiêu:** Ước lượng xác suất mua sản phẩm NBFO của từng nhóm Non-IB sau khi họ chuyển lên IB
**File code:** `Non-IB.buyrate.py`
**Tham chiếu IB segmentation:** `IB_clustering.md`

---

## 1. Vấn đề gốc

Non-IB chưa bao giờ là IB → không có lịch sử mua NBFO → **không thể tính buy rate trực tiếp**.

Buy rate cần thiết để:
- Tính CLV = buy rate × 5,000,000đ (revenue từ 1 sản phẩm NBFO)
- Xác định P_breakeven = Cost(channel) / CLV → ngưỡng hòa vốn cho từng nhóm
- Phân loại VIP/Normal → áp FUM tương ứng

---

## 2. Hướng tiếp cận — IB Segment Proxy

**Logic chuỗi:**

```
Non-IB cluster X
       ↓
Profile tài chính X giống IB segment Y nhất
       ↓
Sau khi lên IB, X sẽ hành xử như Y
       ↓
Buy rate của X ≈ buy rate thực tế của Y
       ↓
CLV = buy rate × 5,000,000đ
```

**Tại sao hợp lý:**
- Các nhóm Non-IB được cluster theo đặc điểm tài chính (sản phẩm sở hữu, số dư, độ sâu sản phẩm) — cùng features với IB segmentation
- KH có profile tài chính tương tự nhau có xu hướng hành xử tương tự sau khi lên cùng một nền tảng
- Đây là cách duy nhất có thể làm với data hiện có — không có lịch sử mua NBFO của Non-IB

**Giới hạn cần thừa nhận:**
- Giả định hành vi sau khi lên IB = hành vi IB segment tương ứng — chưa được verify trực tiếp
- Selection bias: IB segment hiện tại có thể đã trải qua quá trình lọc (chỉ những KH "đáng lên IB" mới lên) → buy rate proxy có thể cao hơn thực tế

---

## 3. Nền tảng — IB Buy Rates (2019 only)

### 3.1 Tại sao chỉ dùng 2019

`base_full.parquet` chứa IB từ cả 3 năm:

| Năm đăng ký IB | Số KH |
|---|---|
| 2019 | 134,162 |
| 2020 | 22,503 |
| 2021 | 2,264 |

KH 2020/2021 tại thời điểm quan sát (2019) vẫn là Non-IB → đưa vào làm loãng buy rate và không đại diện cho hành vi IB ổn định. **Chỉ dùng 134,162 KH đăng ký IB năm 2019.**

### 3.2 IB Segmentation — Rule-based Priority Cascade

```python
def assign_seg(r):
    if r['login_count'] == 0:                                           return 'N3_Dormant'
    if r['AVG_LOAN_AMOUNT'] > 500_000_000:                              return 'V1_HV_Borrower'
    if r['AVG_TD_BALANCE'] > 100_000_000 and r['has_loan'] == 0:        return 'V2_Conservative'
    if r['product_depth'] >= 3 and r['AVG_TD_BALANCE'] > 200_000_000:  return 'V3_Multi_Premium'
    if r['has_card'] == 1 and r['has_loan'] == 1:                       return 'N1_Active_Digital'
    return 'N2_Semi_Digital'
```

Features: `login_count`, `AVG_LOAN_AMOUNT`, `AVG_TD_BALANCE`, `has_loan`, `product_depth`, `has_card`

### 3.3 Buy Rate thực tế theo segment

Label: `TARGET_PRODUCT_OPEN_COUNT_HISTORY > 0` trong `gcon_model_input.parquet`
→ KH đã mua ít nhất 1 sản phẩm NBFO = True, chưa mua = False

| Segment | N | % IB | Buy Rate | Nhóm FUM |
|---|---|---|---|---|
| V3 Multi Premium | 734 | 0.5% | **11.17%** | VIP |
| N1 Active Digital | 9,987 | 7.4% | **6.81%** | Normal |
| V1 HV Borrower | 6,760 | 5.0% | 4.56% | VIP |
| N3 Dormant | 57,872 | 43.1% | 5.13%* | Onboarding |
| V2 Conservative | 3,702 | 2.8% | 3.92% | VIP |
| N2 Semi Digital | 55,107 | 41.1% | 2.13% | Normal |

*N3 buy rate từ lịch sử offline — không dùng làm proxy cho Non-IB.
N3_Dormant bị loại khỏi tập mapping vì chưa active IB.

---

## 4. Phương pháp mapping — Dual Method

Mỗi Non-IB cluster được map về IB segment bằng **2 phương pháp độc lập**, sau đó kết hợp theo logic conservative.

### 4.1 Cách 1 — Rule-based Mapping

Áp cùng bộ quy tắc phân loại IB lên **centroid** (giá trị trung bình) của từng Non-IB cluster. Ngưỡng điều chỉnh từ nhị phân (0/1) sang liên tục (0.3) cho các features có/không:

```python
def rule_map(centroid):
    if centroid['AVG_LOAN_AMOUNT'] > 500M:              → V1
    if centroid['AVG_TD_BALANCE'] > 100M
       and centroid['has_loan'] == 0:                   → V2
    if centroid['product_depth'] >= 3
       and centroid['AVG_TD_BALANCE'] > 200M:           → V3
    if centroid['has_card'] > 0.3
       and centroid['has_loan'] > 0.3:                  → N1
    else:                                               → N2
```

**Điểm mạnh:** Dùng domain logic rõ ràng, dễ giải thích
**Điểm yếu:** Ngưỡng `> 0.3` cho centroid là ad-hoc — không có cơ sở thống kê

### 4.2 Cách 2 — Euclidean Distance Mapping

Đo khoảng cách hình học giữa centroid Non-IB cluster và centroid IB segment trong không gian 6 chiều. Non-IB cluster nào gần IB segment nào nhất → map về đó.

**Bước 1 — Z-score normalization:**

Nếu không normalize, `AVG_LOAN_AMOUNT` (đơn vị VND, ~100M–1B) sẽ áp đảo `has_card` (0–1) khi tính khoảng cách.

```python
all_data = pd.concat([ib_cands, nonib_centroids])
mu = all_data.mean()
sd = all_data.std().replace(0, 1)

ib_norm    = (ib_cands - mu) / sd
nonib_norm = (nonib_centroids - mu) / sd
# → mọi feature đều có đơn vị: số lần lệch chuẩn
```

**Bước 2 — Ma trận khoảng cách (9 × 5):**

```python
dist = cdist(nonib_norm, ib_norm)
# dist[i][j] = sqrt( Σ (nonib_cluster_i[k] - ib_seg_j[k])² )
```

Mỗi Non-IB cluster lấy IB segment có distance nhỏ nhất.

**Điểm mạnh:** Khách quan, không cần đặt ngưỡng tay, phản ánh sự tương đồng tổng thể
**Điểm yếu:** Gần nhất ≠ thực sự giống — nếu cluster không fit tốt vào segment nào thì distance lớn với tất cả

### 4.3 Kết hợp — Conservative Tie-breaking

```
Cách 1 → segment A → rate_A
Cách 2 → segment B → rate_B

Nếu A == B  →  final = rate_A           (đồng thuận → confidence cao)
Nếu A != B  →  final = min(rate_A, rate_B)   (lệch → conservative)
```

**Tại sao conservative:** Hai phương pháp độc lập nhau. Nếu đồng thuận = hai bằng chứng cùng chỉ về một kết quả → tin. Nếu lệch = không biết cái nào đúng → lấy thấp hơn để an toàn khi defend.

---

## 5. Kết quả Mapping

| Cluster | Rule → Seg | Dist → Seg | Dist | Agree? | Rate (R) | Rate (D) | **Final** |
|---|---|---|---|---|---|---|---|
| C0 Traditional | N1 | N2 | 1.28 | NO | 6.81% | 2.13% | **2.13%** |
| C1 HV Traditional | N1 | N1 | 1.15 | YES | 6.81% | 6.81% | **6.81%** |
| C2 Senior HV | N2 | V2 | 3.05 | NO | 2.13% | 3.92% | **2.13%** |
| C3 Ultra Saver | V2 | V2 | 1.61 | YES | 3.92% | 3.92% | **3.92%** |
| C4 Multi Saver | N2 | N2 | 2.26 | YES | 2.13% | 2.13% | **2.13%** |
| C5 HV Saver | V2 | V2 | 2.52 | YES | 3.92% | 3.92% | **3.92%** |
| C6 Stable Senior | V2 | V2 | 1.84 | YES | 3.92% | 3.92% | **3.92%** |
| C7 HV Borrower | V1 | V1 | 2.03 | YES | 4.56% | 4.56% | **4.56%** |
| P0 Dormant | — | — | — | — | — | — | **IGNORE** |

**6/8 cluster active đồng thuận giữa 2 phương pháp** → confidence cao cho kết quả cuối.

**Giải thích 2 trường hợp lệch:**

- **C0 (NO):** Rule map về N1 vì centroid có `has_card=0.47 > 0.3` và `has_loan=0.43 > 0.3` — vừa đủ ngưỡng. Distance map về N2 vì C0 về tổng thể gần N2 hơn (N1 thực sự có 100% loan, 100% card; C0 chỉ ~45%). Distance phản ánh đúng hơn → conservative = 2.13%

- **C2 (NO):** Rule map về N2 vì có loan → không vào V2 (`has_loan == 0` là điều kiện V2). Distance map về V2 vì C2 có TD cao. Distance lớn nhất trong tất cả (3.05) → không fit tốt vào segment nào → conservative = 2.13%, confidence thấp

---

## 6. Phân loại VIP / Normal / Ignore

Non-IB clusters được phân loại dựa trên IB segment mà chúng map về:

```
Non-IB clusters (8 active + 1 ignore)
├── VIP (4 clusters) — map về V1/V2
│   Đặc điểm: TD lớn hoặc loan lớn → CLV cao hơn sau khi lên IB
│   ├── C3 Ultra Saver    → V2   912 KH
│   ├── C5 HV Saver       → V2  13,180 KH
│   ├── C6 Stable Senior  → V2   5,792 KH
│   └── C7 HV Borrower    → V1     653 KH
│
├── Normal (4 clusters) — map về N1/N2
│   Đặc điểm: Sản phẩm cơ bản, balance thấp → CLV trung bình
│   ├── C1 HV Traditional → N1     493 KH
│   ├── C0 Traditional    → N2  80,041 KH
│   ├── C2 Senior HV      → N2     562 KH
│   └── C4 Multi Saver    → N2     800 KH
│
└── Ignore (1 cluster)
    └── P0 Dormant   24,570 KH — zero products, zero balance
        → Không có gì để cross-sell dù lên IB
```

**Lưu ý quan trọng:**
- C1 (Normal) có buy rate **6.81%** — cao nhất toàn Non-IB, cao hơn cả các VIP clusters
- Điều này xảy ra vì C1 map về N1 (Active Digital) — N1 có buy rate cao nhất trong IB vì đây là nhóm chủ động tìm kiếm sản phẩm mới, không phải vì tài sản lớn
- VIP/Normal trong Non-IB phản ánh **loại tài sản**, không phải buy rate cao/thấp

---

## 7. Final Buy Rate & CLV

| Cluster | N | Type | Buy Rate | CLV | Maps to IB |
|---|---|---|---|---|---|
| C1 HV Traditional | 493 | Normal | **6.81%** | 340,443đ | N1 Active Digital |
| C7 HV Borrower | 653 | VIP | 4.56% | 227,811đ | V1 HV Borrower |
| C3 Ultra Saver | 912 | VIP | 3.92% | 195,840đ | V2 Conservative |
| C5 HV Saver | 13,180 | VIP | 3.92% | 195,840đ | V2 Conservative |
| C6 Stable Senior | 5,792 | VIP | 3.92% | 195,840đ | V2 Conservative |
| C0 Traditional | 80,041 | Normal | 2.13% | 106,429đ | N2 Semi Digital |
| C2 Senior HV | 562 | Normal | 2.13% | 106,429đ | N2 Semi Digital |
| C4 Multi Saver | 800 | Normal | 2.13% | 106,429đ | N2 Semi Digital |
| P0 Dormant | 24,570 | IGNORE | — | — | — |

**Weighted average** (loại P0):
```
Tổng KH active = 80,041+493+562+912+800+13,180+5,792+653 = 102,433
Weighted buy rate = Σ(N_i × rate_i) / 102,433 = ~2.5%
```
