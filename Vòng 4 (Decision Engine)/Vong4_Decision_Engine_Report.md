# BAO CAO KET QUA VONG 4.2: DECISION ENGINE - IB
**Doi thi:** GCON (Nguyen Tien Manh, Pham Van Linh, Tran Duc Lap)

---

## 1. Muc tieu va population

Engine IB duoc dung cho bai toan cross-sell NBFO tren tap khach da co IB.

```text
Population: 124,886 khach IB
Budget: 450,000,000 VND
Human-touch cap: 6,000 luot Telesales/RM
Objective: maximize total Expected Marginal Utility (EMU)
```

Non-IB khong duoc dua vao file `final_allocations.csv`.

---

## 2. Segmentation IB

IB segmentation duoc gan theo cohort va product profile. Rieng `N3_Dormant` phai tach theo cohort dang ky IB de tranh gan nham khach chua co IB trong snapshot 2019:

```python
if IB_REGISTER_DATE is missing:
    return DEFAULT_SEGMENT
if IB_REGISTER_DATE.year not in {2020, 2021} and login_count == 0:
    return 'N3_Dormant'
if AVG_LOAN_AMOUNT > 500_000_000:
    return 'V1_HV_Borrower'
if AVG_TD_BALANCE > 100_000_000 and has_loan == 0:
    return 'V2_Conservative'
if product_depth >= 3 and AVG_TD_BALANCE > 200_000_000:
    return 'V3_Multi_Premium'
if has_card == 1 and has_loan == 1:
    return 'N1_Active_Digital'
return 'N2_Semi_Digital'
```

Doc logic dormant:

- Khach co `IB_REGISTER_DATE` trong nam 2019 tro ve truoc: neu `login_count = 0` thi duoc coi la `N3_Dormant`, vi tai thoi diem snapshot 2019 ho da co IB nhung khong phat sinh activity.
- Khach co `IB_REGISTER_DATE` nam 2020 hoac 2021: khong dung `login_count = 0` cua snapshot 2019 de gan dormant, vi luc do khach chua dang ky IB nen zero-login chi la artifact.
- Khach thieu `IB_REGISTER_DATE`: dua ve `DEFAULT_SEGMENT` thay vi gan dormant, de tranh silent error.

`login_count` neu bi missing tung dong se fallback sang `ACTIVITY_NO_SUM`; neu van missing thi fill ve 0. Cach nay giup row co `login_count = NaN` khong bi rot khoi rule dormant mot cach am tham.

---

## 3. Financial Utility Matrix

| Segment | TP | FP | FN |
|---|---:|---:|---:|
| V1_HV_Borrower | 5,000,000 | -50,000 | -30,000,000 |
| V2_Conservative | 5,000,000 | -50,000 | -30,000,000 |
| V3_Multi_Premium | 5,000,000 | -50,000 | -30,000,000 |
| N1_Active_Digital | 5,000,000 | -50,000 | 0 |
| N2_Semi_Digital | 5,000,000 | -50,000 | 0 |
| N3_Dormant | 5,000,000 | -50,000 | 0 |

Cong thuc EMU:

```text
uplift_c(P) = 4 * P * (1 - P) * CR_c
EMU_c(P) = uplift_c(P) * (TP - FN)
           + (1 - P - uplift_c(P)) * FP
           - channel_cost_c
```

---

## 4. MILP Solver

Thay vi sap xep uu tien theo EMU/efficiency, allocation duoc dua ve bai toan Mixed Integer Linear Programming.

Bien quyet dinh:

```text
x[i,c] = 1 neu khach i duoc gan kenh c
x[i,c] = 0 neu khong
```

Objective:

```text
maximize sum_i sum_c EMU[i,c] * x[i,c]
```

Constraints:

```text
sum_c x[i,c] <= 1                         moi khach toi da 1 kenh
sum_i sum_c cost[c] * x[i,c] <= 450M       budget constraint
sum_i (x[i,Telesales] + x[i,RM]) <= 6,000  human-touch cap
x[i,c] = 0 neu channel khong eligible hoac EMU <= 0
```

Solver: `scipy.optimize.milp` voi HiGHS. Ket qua baseline va stress deu tra ve optimal.

---

## 5. Channel va threshold

| Kenh | Cost | CR | Constraint |
|---|---:|---:|---|
| SMS | 5,000 | 2% | Khong gioi han |
| Telesales | 50,000 | 5% | Thuoc human cap |
| RM | 2,000,000 | 15% | Thuoc human cap, VIP only |

| Segment | SMS | Telesales | RM |
|---|---:|---:|---:|
| V1_HV_Borrower | 0.0198 | 0.0144 | 0.1092 |
| V2_Conservative | 0.0198 | 0.0144 | 0.1092 |
| V3_Multi_Premium | 0.0198 | 0.0144 | 0.1092 |
| N1_Active_Digital | 0.1382 | 0.1048 | N/A |
| N2_Semi_Digital | 0.1382 | 0.1048 | N/A |
| N3_Dormant | 0.1382 | 0.1048 | N/A |

Budget va human cap hien tai duoc chon tu coarse-to-refine grid search tren tong budget 1B:

```text
Coarse budget: 400/600, 500/500, 600/400 voi human 5000/5000
Refine quanh coarse best 500/500: 450/550, 500/500, 550/450
Human refine: 4000/6000, 5000/5000, 6000/4000
Best normalized score: IB 450M / Non-IB 550M, human 6000 / 4000
```

---

## 6. Baseline Result

MILP status: optimal.

| Metric | Ket qua |
|---|---:|
| Total customers | 124,886 |
| Profit / EMU ky vong | 5,103,659,144 VND |
| Cost | 448,310,000 VND |
| Budget limit | 450,000,000 VND |
| SMS | 26,942 |
| Telesales | 5,992 |
| RM | 7 |
| None / Auto-brake | 91,945 |
| Human-touch used | 5,999 |

Doc ket qua:

- Budget gan binding: solver dung 448.310M / 450M.
- Human cap gan binding: Telesales + RM = 5,999 / 6,000.
- RM chi con 6 slot vi budget IB giam tu 700M xuong 450M; solver uu tien Telesales/SMS co EMU tren chi phi tot hon.
- EMU van duy tri 5.11 ty VND nho human cap IB tang len 6,000.

---

## 7. Allocation theo segment IB

| Segment | N | SMS | Telesales | RM | None | Cost | EMU |
|---|---:|---:|---:|---:|---:|---:|---:|
| V3_Multi_Premium | 240 | 0 | 36 | 0 | 204 | 1.80M | 23.9M |
| V1_HV_Borrower | 6,260 | 65 | 3,470 | 1 | 2,724 | 175.83M | 1,673.7M |
| V2_Conservative | 3,873 | 213 | 1,599 | 6 | 2,055 | 93.02M | 1,843.5M |
| N1_Active_Digital | 6,531 | 1,051 | 375 | 0 | 5,105 | 24.01M | 130.6M |
| N3_Dormant | 60,223 | 18,117 | 257 | 0 | 41,849 | 103.44M | 963.7M |
| N2_Semi_Digital | 47,759 | 7,496 | 255 | 0 | 40,008 | 50.23M | 468.3M |

Doc ket qua:

- `V1_HV_Borrower` va `V2_Conservative` nhan gan toan bo Telesales/RM vi la nhom high-value co opportunity loss cao.
- `N3_Dormant` va `N2_Semi_Digital` duoc tiep can chu yeu bang SMS de toi uu scale va tranh dot chi phi human-touch.

---

## 8. Stress Test

Kich ban stress:

```text
FP VIP +20%
CR Telesales/RM -15%
```

Stress test duoc chay lai MILP rieng cho kich ban adverse, khong chi re-score allocation baseline.

| Scenario | COGS | EMU | Expected conversions | SMS | Telesales | RM | Incremental ROI |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 448.31M | 5.10B | 624.04 | 26,942 | 5,992 | 7 | 10.38x |
| Adverse CR/FP re-optimized | 449.97M | 4.42B | 601.21 | 26,923 | 5,987 | 8 | 8.83x |

Stress EMU giam nhung van duong lon. Solver giam nhe SMS/Telesales va tang RM tu 7 len 8 vi mot so khach VIP van co EMU du tot trong stress.

---

## 8. Output

File output chinh:

```text
final_allocations.csv
```

Cot chinh:

```text
CUSTOMER_NUMBER
CUSTOMER_TYPE
SEGMENT_CLUSTER
MAPPED_IB_SEGMENT
RECOMMENDED_PRODUCT
PROBABILITY
RECOMMENDED_CHANNEL
```

---

## 9. Ket luan

IB decision engine hien tai khong con la heuristic sorting don thuan. Allocation da duoc solve bang MILP voi objective EMU va cac rang buoc da kenh: budget, human cap, eligibility, va one-channel-per-customer. Sau grid search, IB duoc cap 450M budget va 6,000 human contacts; baseline dat EMU ky vong 5.11 ty VND.
