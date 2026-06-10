# BAO CAO KET QUA VONG 4.2: DECISION ENGINE - NON-IB RETENTION
**Doi thi:** GCON (Nguyen Tien Manh, Pham Van Linh, Tran Duc Lap)

---

## 1. Muc tieu

Non-IB duoc chuyen tu bai toan activation/onboarding sang bai toan retention/churn prevention.

```text
Population: 127,003 khach Non-IB
At-risk base: 77,827 khach
Budget: 550,000,000 VND
Human-touch cap: 4,000 luot Telesales/RM
Objective: maximize expected retention EMU
```

---

## 2. Churn Definition

Churn duoc xac dinh dua tren dien bien tai chinh hang thang trong nam 2019, gom tien gui, du no vay va so luong the. Muc tieu khong phai la gan nhan churn cung cho tat ca khach Non-IB, ma la nhan dien nhom tung co quan he tai chinh voi ngan hang va dang co dau hieu suy giam hoac roi bo.

Truoc het, engine xac dinh nhom at-risk: day la cac khach da tung phat sinh hoat dong tai chinh trong giai doan Jan-Sep 2019. Nhung khach nay co lich su quan he voi ngan hang, nen neu ho bien mat hoac suy giam manh trong Q4 thi can duoc xem la co rui ro retention.

Trong nhom at-risk, churn duoc chia thanh hai muc:

- Hard churn: khach gan nhu khong con hoat dong trong Q4. Day la dau hieu roi bo ro rang nhat, vi khach da tung active truoc do nhung den cuoi nam gan nhu khong con gia tri tai chinh hoac hoat dong the.
- Runoff risk: khach van con hoat dong trong Q4 nhung gia tri tai chinh giam rat manh so voi thang 9. Nhom nay chua duoc xem la churn hoan toan, vi voi khach gui tiet kiem/term deposit, so du giam co the den tu dao han hoac tai co cau ky han.

Vi runoff risk khong chac chan la churn that su, engine chi tinh no voi trong so 30% khi uoc luong rui ro churn hieu dung. Sau do, rui ro churn duoc tong hop theo cluster/persona de tao `P_CHURN_cluster`. Gia tri nay dai dien cho xac suat rui ro cua ca cum, va duoc dung de tinh EMU cho tung kenh tiep can.

P0 Dormant duoc gan `P_CHURN = 0.001` va khong dua vao paid retention target.

---

## 3. Financial Utility Matrix

Theo yeu cau moi, TP retention la gia tri co dinh 50M, khong dung `min(CLV, 50M)`.

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
C6_Stable_Senior
C7_HV_Borrower
```

Cong thuc EMU:

```text
uplift_c(P) = 4 * P * (1 - P) * CR_c
EMU_c(P) = uplift_c(P) * (TP - FN)
           + (1 - P - uplift_c(P)) * FP
           - channel_cost_c
```

---

## 4. MILP Solver

Non-IB co nhieu khach trong cung cluster co cung `P_CHURN`, `TP`, `FP`, `FN`, nen MILP duoc solve o cap cluster-channel bang bien nguyen:

```text
y[g,c] = so khach cluster g duoc gan kenh c
```

Objective:

```text
maximize sum_g sum_c EMU[g,c] * y[g,c]
```

Constraints:

```text
sum_c y[g,c] <= AT_RISK_CUSTOMERS[g]          moi cluster khong vuot so khach at-risk
sum_g sum_c cost[c] * y[g,c] <= 550M          budget constraint
sum_g (y[g,Telesales] + y[g,RM]) <= 4,000     human-touch cap
y[g,c] = 0 neu EMU[g,c] <= 0 hoac channel khong eligible
```

Sau khi MILP tra ve so luong theo cluster-channel, engine bung ra customer-level output. Trong cung mot cluster, EMU cua cac khach la nhu nhau; `ASSET_SCORE` chi dung de chon ID on dinh khi bung allocation, khong phai de giai bai toan toi uu.

Solver: `scipy.optimize.milp` voi HiGHS. Ket qua baseline tra ve optimal.

---

## 5. Channel va constraints

| Kenh | Cost | CR | Vai tro |
|---|---:|---:|---|
| SMS | 5,000 | 2% | Kenh scale/fallback |
| Telesales | 50,000 | 5% | Human retention |
| RM | 2,000,000 | 15% | VIP-only, thuoc human cap |

| Constraint | Gia tri |
|---|---:|
| Budget | 550,000,000 VND |
| Human-touch capacity | 4,000 |
| Population | 127,003 |
| At-risk base | 77,827 |

Budget va human cap hien tai duoc chon tu coarse-to-refine grid search tren tong budget 1B:

```text
Coarse budget: 400/600, 500/500, 600/400 voi human 5000/5000
Refine quanh coarse best 500/500: 450/550, 500/500, 550/450
Human refine: 4000/6000, 5000/5000, 6000/4000
Best normalized score: IB 450M / Non-IB 550M, human 6000 / 4000
```

---

## 6. Churn Rate theo cluster

| Cluster | At Risk | Hard Churn | Runoff Risk | Effective Churn | P Churn |
|---|---:|---:|---:|---:|---:|
| C3_Ultra_Saver | 393 | 229 | 25 | 236.5 | 60.18% |
| C5_HV_Saver | 10,544 | 1,325 | 1,618 | 1,810.4 | 17.17% |
| C4_Multi_Saver | 591 | 21 | 260 | 99.0 | 16.75% |
| C2_Senior_HV | 511 | 44 | 103 | 74.9 | 14.66% |
| C1_HV_Traditional | 470 | 12 | 80 | 36.0 | 7.66% |
| C0_Traditional | 59,864 | 2,198 | 5,775 | 3,930.5 | 6.57% |
| C6_Stable_Senior | 4,858 | 47 | 541 | 209.3 | 4.31% |
| C7_HV_Borrower | 596 | 0 | 70 | 21.0 | 3.52% |
| P0_Dormant | 0 | 0 | 0 | 0.0 | 0.10% |

Doc ket qua:

- C4 giam manh sau khi tach runoff risk, tu raw churn cao ve effective churn 16.75%.
- C3 van cao vi phan lon la hard churn/Q4 inactive, khong phai runoff.
- C0 co population rat lon nen du churn rate thap van duoc MILP chon SMS khi budget con phu hop.
- C7 co churn risk thap va khong duoc allocate vi trong dieu kien ngan sach hien tai, cac cum khac tao EMU tren chi phi tot hon.

---

## 7. Baseline Allocation

MILP status: optimal.

| Metric | Ket qua |
|---|---:|
| Population | 127,003 |
| At-risk base | 77,827 |
| Hard churn proxy | 3,876 |
| Runoff risk proxy | 8,472 |
| Effective churn score | 6,417.6 |
| SMS allocated | 66,680 |
| Telesales | 292 |
| RM | 101 |
| None | 59,930 |
| Cost | 550,000,000 VND |
| Campaign EMU | 19,311,810,610 VND |
| Expected retained customers | 422.81 |
| CLV at risk targeted | 116,585,411,994 VND |
| Human-touch used | 393 / 4,000 |

Budget binding:

```text
66,680 * 5,000 + 292 * 50,000 + 101 * 2,000,000 = 550,000,000 VND
```

Budget binding dung 550M. Human cap chua binding hoan toan vi budget het truoc. RM duoc dung cho nhom co gia tri/rui ro cao, nhung tong so luot RM van rat nho do chi phi moi luot cao so voi SMS/Telesales.

---

## 8. Allocation theo cluster

| Cluster | N | SMS | Telesales | RM | None | Cost | EMU |
|---|---:|---:|---:|---:|---:|---:|---:|
| C3_Ultra_Saver | 912 | 0 | 292 | 101 | 519 | 216.60M | 2,058.4M |
| C5_HV_Saver | 13,180 | 10,544 | 0 | 0 | 2,636 | 52.72M | 5,514.8M |
| C4_Multi_Saver | 800 | 591 | 0 | 0 | 209 | 2.96M | 500.2M |
| C2_Senior_HV | 562 | 511 | 0 | 0 | 51 | 2.56M | 385.0M |
| C1_HV_Traditional | 493 | 470 | 0 | 0 | 23 | 2.35M | 188.8M |
| C0_Traditional | 80,041 | 49,706 | 0 | 0 | 30,335 | 248.53M | 9,638.7M |
| C6_Stable_Senior | 5,792 | 4,858 | 0 | 0 | 934 | 24.29M | 1,025.9M |
| C7_HV_Borrower | 653 | 0 | 0 | 0 | 653 | 0 | 0 |
| P0_Dormant | 24,570 | 0 | 0 | 0 | 24,570 | 0 | 0 |

---

## 9. Stress Test

Kich ban:

```text
FP +20%
CR Telesales/RM -15%
```

| Metric | Baseline | Stress |
|---|---:|---:|
| EMU | 19,311,810,610 | 18,360,141,863 |
| Delta EMU | - | -4.9% |
| Baseline cost | 550,000,000 | 550,000,000 |
| Baseline SMS | 66,680 | 66,680 |
| Baseline Telesales | 292 | 292 |
| Baseline RM | 101 | 101 |

Stress chi tinh lai EMU tren allocation baseline de do sensitivity. Neu can stress allocation toi uu lai, co the chay them MILP rieng cho tung scenario.

---

## 10. Output

File output chinh:

```text
final_allocations_nonib.csv
```

Output phu:

```text
thresholds_nonib.csv
thresholds_nonib.md
nonib_retention_cluster_summary.csv
```

---

## 11. Ket luan

Non-IB engine hien tai da tap trung vao retention, dung churn rate theo cluster, TP co dinh 50M, va allocation bang MILP thay cho heuristic sorting. Sau grid search, Non-IB duoc cap 550M budget va 4,000 human contacts. Baseline dung het budget, tao 422.81 expected retained customers, va dat campaign EMU 19.31 ty VND. RM duoc phan bo vao C3_Ultra_Saver, la cum co EMU RM cao nhat.
