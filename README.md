# BA QUẢ TÁO — G'CONTEST 2026

> **Ứng dụng Data Analytics xây dựng giải pháp phân tích hành vi khách hàng, cá nhân hóa dịch vụ và ra quyết định tối ưu trong lĩnh vực ngân hàng số**

**Đội thi:** BA QUẢ TÁO — Nguyễn Tiến Mạnh · Phạm Văn Linh · Trần Đức Lập

---

## Tổng quan dự án

Dự án xây dựng hệ thống **Decision Intelligence** cho ngân hàng số, bao gồm ba nhánh chính:

| Nhánh | Phạm vi | Kỹ thuật cốt lõi |
|---|---|---|
| **NBFO (IB)** | 124,886 khách đã có Internet Banking | XGBoost + Platt Scaling + MILP |
| **Persona (Non-IB)** | ~127,000 khách chưa có IB | GMM Clustering + PCA |
| **Decision Engine** | Toàn bộ tệp khách hàng | Grid Search + MILP (HiGHS) |

Hệ thống không dừng ở bước **dự đoán (prediction)**, mà chuyển kết quả mô hình thành **quyết định hành động**: khách nào được tiếp cận, bằng kênh nào, với ngân sách bao nhiêu, và giá trị kỳ vọng (Expected Marginal Utility — EMU) là bao nhiêu.

---

## Cấu trúc thư mục

```
gcon/
├── data/                              # Dữ liệu gốc (gitignored — 6 bảng CSV)
├── cleaned_data/                      # Dữ liệu sau cleaning (gitignored — parquet)
├── clean.ipynb                        # Notebook làm sạch dữ liệu
├── tổng quan.ipynb                    # EDA tổng quan & xác định vấn đề
├── requirements.txt                   # Python dependencies
│
├── NBFO_IB/                           # Hướng 1: NBFO — Supervised Learning
│   ├── EDA_Feature_engineering.ipynb  # EDA + Feature Engineering (158 features)
│   ├── train.ipynb                    # Model training, evaluation & SHAP
│   ├── processed_data/                # Model input parquet (gitignored)
│   └── saved_models/                  # Trained models & calibrated test scores (gitignored)
│
├── Cluster_nonIB/                     # Hướng 2: Persona — Unsupervised Learning
│   ├── 01_nonib_data_prep.ipynb
│   ├── 02_nonib_eda.ipynb
│   ├── 03_nonib_clustering.ipynb      # GMM Clustering + PCA + Stability check
│   ├── 04_nonib_cluster_profiling.ipynb
│   └── output/                        # Cluster output parquet (gitignored)
│
├── Demo/
│   └── streamlit_app.py               # Interactive demo dashboard
│
├── Vòng 4 (Decision Engine)/          # Hướng 3: Decision Engine — Vòng chung kết
│   ├── GCON_Vong4_BaoCao_TongHop_IB_NonIB.md  # Báo cáo tổng hợp chính (Pitch source)
│   ├── IB_clustering.md               # Tài liệu IB segmentation
│   ├── Vong4_Decision_Engine_Report.md        # Báo cáo chi tiết nhánh IB
│   ├── Vong4_Decision_Engine_NonIB_Report.md  # Báo cáo chi tiết nhánh Non-IB
│   │
│   ├── decision_config.py             # Config trung tâm: channels, FUM, MILP solvers
│   ├── decision_engine.py             # Engine IB: Cross-sell NBFO (MILP customer-level)
│   ├── decision_engine_nonib.py       # Engine Non-IB: Retention/churn (MILP group-level)
│   ├── gridsearch_budget_human_milp.py # Grid search phân bổ budget/human cap
│   ├── optimized_thresholds.py        # Threshold vector tối ưu (Nhiệm vụ 1)
│   ├── calculate_thresholds.py        # Break-even threshold per segment × channel
│   ├── business_kpi_scenarios.py      # CAC, Incremental ROI, P&L, Stress re-optimization
│   ├── generate_heatmap.py            # Heatmap 4×4 sensitivity analysis (IB)
│   ├── cluster_ib.py                  # IB segmentation rule-based
│   │
│   ├── thresholds.md                  # Break-even threshold kết quả (IB)
│   ├── thresholds_nonib.md            # Break-even threshold kết quả (Non-IB)
│   └── persona_onboarding_analysis.md # Phân tích tỷ lệ onboarding theo persona
│
└── README.md
```

---

## Dữ liệu

| Bảng | Mô tả | Rows |
|---|---|---:|
| `Data_Customer.csv` | Nhân khẩu học, ngày tạo TK, đăng ký IB | 285,934 |
| `Data_Deposit.csv` | Tiền gửi TKTT & tiết kiệm theo tháng | 1,258,424 |
| `Data_Lending.csv` | Khoản vay & tín dụng theo tháng | 576,431 |
| `Data_Card.csv` | Thẻ tín dụng & ghi nợ theo tháng | 871,589 |
| `Data_Transaction.csv` | Giao dịch e-banking chi tiết | 1,417,982 |
| `Data_Activity.csv` | Hoạt động đăng nhập & sử dụng app | 1,048,575 |

> Dữ liệu được cung cấp bởi BTC G'CONTEST 2026 và thuộc quyền bảo mật. Không chia sẻ ra ngoài.

---

## Phương pháp

### Hướng 1: NBFO — Đề xuất sản phẩm tài chính (IB)

| Bước | Chi tiết |
|---|---|
| **Target** | Khách hàng IB mua sản phẩm mới trong 2 tháng tới (h=2) |
| **Features** | 158 features: demographic, balance, transaction, activity, RFM, temporal (lag/diff/roll), product co-occurrence affinity |
| **Split** | Time-based: Train M1–M8, Validation M9, Test M10 |
| **Model** | XGBoost — PR-AUC ≈ 0.598 trên tập test |
| **Calibration** | Platt Scaling (Sigmoid) — đưa raw score về empirical probability |
| **Interpretability** | SHAP (global + per-product feature importance) |

### Hướng 2: Persona — Chân dung khách hàng Non-IB

| Bước | Chi tiết |
|---|---|
| **Scope** | ~127,000 khách hàng non-IB |
| **Features** | 18 features: age, tenure, balances, trends, product count, loan, transaction |
| **Clustering** | Gaussian Mixture Model (GMM), k=8, qua PCA 18 components |
| **Evaluation** | Silhouette Score = 0.696, Bootstrap ARI = 0.962 |
| **Output** | 8 personas (7 rõ nét + 1 Dormant), kèm churn risk và chiến lược retention |

### Hướng 3: Decision Engine — Vòng chung kết

#### Kiến trúc tổng thể

```
[ML Probability / Churn Risk]
         ↓
[Financial Utility Matrix (FUM): TP / FP / FN]
         ↓
[Expected Marginal Utility (EMU) per customer × channel]
         ↓
[Grid Search: phân bổ Budget & Human-touch cap]
    IB: 450M / 6,000 lượt  |  Non-IB: 550M / 4,000 lượt
         ↓
[MILP Solver (HiGHS via scipy)]
    IB: customer-level allocation  |  Non-IB: group-level allocation
         ↓
[Output: final_allocations.csv / final_allocations_nonib.csv]
```

#### Engine IB — Cross-sell NBFO

- **Population:** 124,886 khách IB
- **Segmentation:** Rule-based profile (V1_HV_Borrower, V2_Conservative, V3_Multi_Premium, N1_Active_Digital, N2_Semi_Digital, N3_Dormant)
- **MILP:** Tối đa tổng EMU, ràng buộc ngân sách 450M và human-touch ≤ 6,000 (Telesales + RM)
- **Kết quả baseline:** EMU 5.10B VND, COGS 448.3M VND, Incremental ROI 10.38x

#### Engine Non-IB — Retention

- **Population:** 127,000 khách Non-IB, 77,824 at-risk
- **Churn definition:** Hard churn (Q4 gần như inactive) + Runoff risk × 30% trọng số
- **Segmentation:** 8 clusters từ GMM (C0–C7 + P0_Dormant)
- **MILP:** Tối đa tổng EMU giữ chân, ràng buộc ngân sách 550M và human-touch ≤ 4,000
- **Kết quả baseline:** EMU 22.59B VND, COGS 550M VND, Incremental ROI 40.07x

#### Kết quả tổng hợp

| Nhánh | COGS | EMU | Expected Conversions | CAC | Incremental ROI |
|---|---:|---:|---:|---:|---:|
| IB | 448.3M | 5.10B | 624 | 718K | 10.38x |
| Non-IB | 550.0M | 22.59B | 421 | 1.305M | 40.07x |
| **Tổng** | **998.3M** | **27.69B** | **1,046** | **955K** | **26.74x** |

---

## Cài đặt & Chạy

### Yêu cầu

- Python ≥ 3.10
- Các thư viện trong `requirements.txt`

### Cài đặt

```bash
git clone <repo-url>
cd gcon
pip install -r requirements.txt
```

### Thứ tự chạy — Vòng 1–3 (NBFO & Clustering)

> **Lưu ý:** Cần đặt dữ liệu gốc vào thư mục `data/` trước khi chạy.

```
1. clean.ipynb
2. tổng quan.ipynb                          (tùy chọn)
3. NBFO_IB/EDA_Feature_engineering.ipynb
4. NBFO_IB/train.ipynb
5. Cluster_nonIB/01_nonib_data_prep.ipynb
6. Cluster_nonIB/02_nonib_eda.ipynb
7. Cluster_nonIB/03_nonib_clustering.ipynb
8. Cluster_nonIB/04_nonib_cluster_profiling.ipynb
```

### Thứ tự chạy — Vòng 4 (Decision Engine)

> Cần chạy xong bước 1–8 ở trên (để có parquet files trong `NBFO_IB/` và `Cluster_nonIB/output/`).

```bash
# Bước 1: Phân tích ngưỡng Break-even và Threshold tối ưu (Nhiệm vụ 1)
cd "Vòng 4 (Decision Engine)"
python calculate_thresholds.py
python optimized_thresholds.py

# Bước 2: Grid search phân bổ budget / human-touch giữa IB và Non-IB
python gridsearch_budget_human_milp.py

# Bước 3: Chạy engine IB (Cross-sell NBFO)
python decision_engine.py
# Output: ../final_allocations.csv

# Bước 4: Chạy engine Non-IB (Retention)
python decision_engine_nonib.py
# Output: ../final_allocations_nonib.csv

# Bước 5: Tính KPI và Stress test
python business_kpi_scenarios.py
# Output: ../business_kpis.csv, ../pnl_projection.csv, ../stress_reoptimized_nonib.csv

# Bước 6 (tùy chọn): Heatmap sensitivity analysis 4×4 (IB)
python generate_heatmap.py
```

### Chạy Demo

```bash
streamlit run Demo/streamlit_app.py
```

Demo mở tại `http://localhost:8501` với 2 tab:
- **NBFO Recommendation** — Tra cứu propensity score và đề xuất sản phẩm cho từng khách IB
- **Non-IB Clustering** — Xem chân dung persona và gợi ý campaign

---

## Kết quả chính

### NBFO (IB)
- XGBoost đạt **PR-AUC 0.598** trên tập test (imbalanced, tháng 10)
- **Lift @20% = 4x** so với random
- Calibrated propensity (Platt Scaling) sẵn sàng cho downstream EMU calculation
- SHAP: features quan trọng nhất là product affinity, deposit balance, activity frequency

### Clustering (Non-IB)
- **8 personas** từ "P0_Dormant" đến "C3_Ultra_Saver"
- **Silhouette Score 0.696** — phân tách tốt giữa các cụm
- **Bootstrap ARI 0.962** — cluster rất ổn định
- Kruskal-Wallis test: tất cả features khác biệt có ý nghĩa thống kê

### Decision Engine (Vòng 4)
- **Tổng EMU baseline: 27.69 tỷ VND** từ ngân sách 998.3 triệu VND
- **Incremental ROI tổng: 26.74x**
- Stress test (FP VIP +20%, CR Telesales/RM -15%): ROI vẫn đạt **8.83x (IB)** và **33.81x (Non-IB)**
- Engine có cơ chế **auto-brake**: không cấp phát ngân sách cho cặp customer-channel có EMU âm

---

## Thành viên nhóm

| Thành viên | Vai trò |
|---|---|
| Nguyễn Tiến Mạnh | Leader |
| Phạm Văn Linh | Member |
| Trần Đức Lập | Member |
