# BAO CAO KET QUA VONG 4.2: DECISION ENGINE - NON-IB RETENTION
**Doi thi:** GCON (Nguyen Tien Manh, Pham Van Linh, Tran Duc Lap)

---

## 1. Muc tieu

Non-IB duoc chuyen tu bai toan activation/onboarding sang bai toan retention/churn prevention.

```text
Population: 127,000 khach Non-IB
At-risk base: 77,824 khach
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
C5_HV_Saver
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

## 4. Optimized Threshold Vector

Do Non-IB khong co NBFO product probability, threshold khong dung xac suat mua san pham ma dung percentile cua expected loss trong tung cluster:

```text
EXPECTED_LOSS_SCORE_i = P_CHURN_cluster * CLV_5YR_i
LOSS_PERCENTILE_i = percentile_rank(EXPECTED_LOSS_SCORE_i within cluster)
```

Grid search duoc chay doc lap truoc MILP theo `cluster x channel`. Moi channel co candidate pool rieng:

```text
Eligible[i,c] = 1 neu CHURN_AT_RISK_i = 1
                 va LOSS_PERCENTILE_i >= threshold[cluster,c]
                 va EMU[cluster,c] > 0
```

De tranh SMS/Telesales/RM co cung cutoff cho moi cluster, engine dung dynamic channel floor theo `P_CHURN_cluster`: cluster rui ro cao duoc ha cutoff de human-touch target rong hon, cluster rui ro thap se co cutoff cao hon. Cong thuc floor:

```text
risk_rank_g = (P_CHURN_g - min(P_CHURN)) / (max(P_CHURN) - min(P_CHURN))
Telesales_floor_g = 85% - risk_rank_g * (85% - 55%)
RM_floor_g = 98% - risk_rank_g * (98% - 90%)
```

Vi cac pool nay long nhau, MILP enforce nested constraint:

```text
y[g,RM] <= top5[g]
y[g,Telesales] + y[g,RM] <= top30[g]
y[g,SMS] + y[g,Telesales] + y[g,RM] <= SMS_eligible[g]
```

| Cluster | SMS cutoff | Telesales cutoff | RM cutoff |
|---|---:|---:|---:|
| C3_Ultra_Saver | 0.33% | 55.00% | 90.00% |
| C5_HV_Saver | 0.01% | 77.77% | 96.07% |
| C4_Multi_Saver | 2.12% | 78.00% | 96.13% |
| C2_Senior_HV | 0.36% | 79.10% | 96.43% |
| C1_HV_Traditional | 2.33% | 82.81% | 97.42% |
| C0_Traditional | 25.36% | 83.39% | N/A |
| C6_Stable_Senior | 0.02% | 84.59% | No ROI |
| C7_HV_Borrower | 0.15% | 85.00% | No ROI |

Full threshold matrix nam trong `optimized_thresholds_nonib.csv`.

---

## 5. MILP Solver

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
y[g,c] khong vuot candidate pool sau threshold cua channel do
```

Sau khi MILP tra ve so luong theo cluster-channel, engine bung ra customer-level output theo thu tu `LOSS_PERCENTILE` va `ASSET_SCORE`. `LOSS_PERCENTILE` dung de uu tien khach co expected loss cao hon trong cung cluster; MILP van la buoc toi uu allocation chinh.

Solver: `scipy.optimize.milp` voi HiGHS. Ket qua baseline tra ve optimal.

---

## 6. Channel va constraints

| Kenh | Cost | CR | Vai tro |
|---|---:|---:|---|
| SMS | 5,000 | 2% | Kenh scale/fallback |
| Telesales | 50,000 | 5% | Human retention |
| RM | 2,000,000 | 15% | VIP-only, thuoc human cap |

| Constraint | Gia tri |
|---|---:|
| Budget | 550,000,000 VND |
| Human-touch capacity | 4,000 |
| Population | 127,000 |
| At-risk base | 77,824 |

Budget va human cap hien tai duoc chon tu coarse-to-refine grid search tren tong budget 1B:

```text
Coarse budget: 400/600, 500/500, 600/400 voi human 5000/5000
Refine quanh coarse best 500/500: 450/550, 500/500, 550/450
Human refine: 4000/6000, 5000/5000, 6000/4000
Best normalized score: IB 450M / Non-IB 550M, human 6000 / 4000
```

---

## 7. Churn Rate theo cluster

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

Doc ket qua:

- C4 giam manh sau khi tach runoff risk, tu raw churn cao ve effective churn 16.75%.
- C3 van cao vi phan lon la hard churn/Q4 inactive, khong phai runoff.
- C0 co population rat lon nen du churn rate thap van duoc MILP chon SMS khi budget con phu hop.
- C7 co churn risk thap va khong duoc allocate vi trong dieu kien ngan sach hien tai, cac cum khac tao EMU tren chi phi tot hon.

---

## 8. Baseline Allocation

MILP status: optimal.

| Metric | Ket qua |
|---|---:|
| Population | 127,000 |
| At-risk base | 77,824 |
| Hard churn proxy | 3,876 |
| Runoff risk proxy | 8,472 |
| Effective churn score | 6,417.6 |
| SMS allocated | 67,920 |
| Telesales | 168 |
| RM | 101 |
| None | 58,811 |
| Cost | 550,000,000 VND |
| Campaign EMU | 22,588,039,112 VND |
| Expected retained customers | 421.49 |
| CLV at risk targeted | 116,608,591,477 VND |
| Human-touch used | 269 / 4,000 |

Budget binding:

```text
67,920 * 5,000 + 168 * 50,000 + 101 * 2,000,000 = 550,000,000 VND
```

Budget binding dung 550M. Human cap chua binding hoan toan vi budget het truoc. Dynamic threshold cho phep cluster rui ro cao nhu C3 dung Telesales/RM rong hon, trong khi cac cluster rui ro thap van bi auto-brake bang cutoff cao.

---

## 9. Allocation theo cluster

| Cluster | N | SMS | Telesales | RM | None | Cost | EMU |
|---|---:|---:|---:|---:|---:|---:|---:|
| C3_Ultra_Saver | 912 | 188 | 168 | 37 | 519 | 83.34M | 1,267.8M |
| C5_HV_Saver | 13,178 | 10,478 | 0 | 64 | 2,636 | 180.39M | 9,364.7M |
| C4_Multi_Saver | 800 | 591 | 0 | 0 | 209 | 2.96M | 500.2M |
| C2_Senior_HV | 562 | 511 | 0 | 0 | 51 | 2.56M | 385.0M |
| C1_HV_Traditional | 493 | 470 | 0 | 0 | 23 | 2.35M | 188.8M |
| C0_Traditional | 80,038 | 50,822 | 0 | 0 | 29,216 | 254.11M | 9,855.7M |
| C6_Stable_Senior | 5,794 | 4,860 | 0 | 0 | 934 | 24.30M | 1,025.8M |
| C7_HV_Borrower | 653 | 0 | 0 | 0 | 653 | 0 | 0 |
| P0_Dormant | 24,570 | 0 | 0 | 0 | 24,570 | 0 | 0 |

---

## 10. Stress Test va Re-optimization

Engine duoc chay lai MILP rieng cho tung kich ban stress, khong chi tinh lai EMU tren allocation baseline. Trong stress scenario, RM minimum duoc noi tu 101 xuong 81 de mo phong viec giam khoang 20 luot cham soc RM khi dieu kien thi truong xau hon.

| Scenario | COGS | EMU | Expected retained | SMS | Telesales | RM | Incremental ROI |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 550.00M | 22.59B | 421.49 | 67,920 | 168 | 101 | 40.07x |
| Adverse CR/FP re-optimized | 550.00M | 19.15B | 390.30 | 75,920 | 168 | 81 | 33.81x |

Bien do dao dong EMU: Non-IB giam tu 22.59B xuong 19.15B, tuong duong khoang -15.2%. Du bi stress, Incremental ROI van dat 33.81x, cho thay chien luoc retention van ben vung. Solver giam RM tu 101 xuong 81 va tang SMS de bao ve scale trong ngan sach.

---

## 11. Output

File output chinh:

```text
final_allocations_nonib.csv
```

Output phu:

```text
thresholds_nonib.csv
thresholds_nonib.md
optimized_thresholds_nonib.csv
nonib_retention_cluster_summary.csv
stress_reoptimized_nonib.csv
```

---

## 12. Ket luan

Non-IB engine hien tai tap trung vao retention, dung churn rate theo cluster, TP co dinh 50M, threshold theo `cluster x channel`, va allocation bang MILP thay cho heuristic sorting. Sau grid search, Non-IB duoc cap 550M budget va 4,000 human contacts. Baseline dung het budget, tao 421.49 expected retained customers, va dat campaign EMU 22.59 ty VND. Dynamic threshold giup C3_Ultra_Saver duoc tiep can human-touch rong hon do churn risk cao, trong khi C5_HV_Saver van duoc xet RM nho la VIP-like cluster va co expected-loss cao.
