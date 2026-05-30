# GCON — G'CONTEST 2026

> **Ứng dụng Data Analytics xây dựng giải pháp phân tích hành vi khách hàng và cá nhân hóa dịch vụ trong lĩnh vực ngân hàng số**

---

## Tổng quan dự án

Dự án khai thác dữ liệu khách hàng, lịch sử giao dịch, hành vi sử dụng dịch vụ số và dữ liệu sản phẩm tài chính nhằm:

1. **Next Best Financial Offer (NBFO)** — Dự đoán khả năng khách hàng IB sử dụng thêm sản phẩm tài chính mới, hỗ trợ cross-selling và marketing cá nhân hóa.
2. **Persona-Based Digital Personalization** — Phân cụm khách hàng non-IB theo hành vi, xây dựng chân dung persona phục vụ chiến lược onboarding và gắn kết.

---

## Cấu trúc thư mục

```
gcon/
├── data/                           # Dữ liệu gốc (6 bảng CSV)
├── cleaned_data/                   # Dữ liệu sau cleaning (parquet + csv)
├── clean.ipynb                     # Notebook làm sạch dữ liệu
├── tổng quan.ipynb                 # Phân tích tổng quan & xác định vấn đề
│
├── NBFO_IB/                        # Hướng 1: NBFO — Supervised Learning
│   ├── EDA_Feature_engineering.ipynb   # EDA + Feature Engineering (158 features)
│   ├── train.ipynb                     # Model training, evaluation & SHAP
│   ├── processed_data/                 # Model input parquet (generated)
│   └── saved_models/                   # Trained models (.joblib) & test scores
│
├── Cluster_nonIB/                  # Hướng 2: Persona — Unsupervised Learning
│   ├── 01_nonib_data_prep.ipynb        # Chuẩn bị dữ liệu non-IB
│   ├── 02_nonib_eda.ipynb              # EDA khách hàng non-IB
│   ├── 03_nonib_clustering.ipynb       # GMM Clustering + PCA + Stability check
│   ├── 04_nonib_cluster_profiling.ipynb # Profiling, Persona naming & Recommendations
│   └── output/                         # Cluster output (parquet, csv, figures)
│
├── Demo/
│   └── streamlit_app.py            # Interactive demo dashboard
│
├── requirements.txt                # Python dependencies
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

---

## Phương pháp

### Hướng 1: NBFO — Đề xuất sản phẩm tài chính (Supervised)

| Bước | Chi tiết |
|---|---|
| **Target** | Khách hàng IB mua sản phẩm mới trong 2 tháng tới (h=2) |
| **Features** | 158 features: demographic, balance, transaction, activity, RFM, temporal (lag/diff/roll), product co-occurrence affinity |
| **Split** | Time-based: Train M1–M8, Validation M9, Test M10 |
| **Models** | XGBoost + LightGBM, RandomizedSearchCV (5-fold CV) |
| **Best model** | XGBoost — PR-AUC ≈ 0.598 |
| **Calibration** | Platt Scaling (Sigmoid) cho probability calibration |
| **Interpretability** | SHAP (global + per-product feature importance) |
| **Recommendation** | Top-K hoặc Threshold-based trên calibrated propensity |

### Hướng 2: Persona — Chân dung khách hàng non-IB (Unsupervised)

| Bước | Chi tiết |
|---|---|
| **Scope** | ~127,000 khách hàng non-IB |
| **Features** | 18 features: age, tenure, balances, trends, product count, loan, transaction |
| **Dimensionality reduction** | PCA (18 components) |
| **Clustering** | Gaussian Mixture Model (GMM), k=8 |
| **Evaluation** | Silhouette Score = 0.696, Bootstrap ARI = 0.962 |
| **Validation** | Kruskal-Wallis test — tất cả features khác biệt có ý nghĩa thống kê |
| **Output** | 8 personas + 1 nhóm Dormant, kèm campaign suggestion |

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

### Thứ tự chạy Notebooks

> **Lưu ý:** Cần đặt dữ liệu gốc vào thư mục `data/` trước khi chạy.

```
1. clean.ipynb                              # Cleaning → cleaned_data/
2. tổng quan.ipynb                           # Phân tích tổng quan (tùy chọn)
3. NBFO_IB/EDA_Feature_engineering.ipynb     # Feature engineering → processed_data/
4. NBFO_IB/train.ipynb                       # Training & evaluation → saved_models/
5. Cluster_nonIB/01_nonib_data_prep.ipynb    # Data prep non-IB
6. Cluster_nonIB/02_nonib_eda.ipynb          # EDA non-IB
7. Cluster_nonIB/03_nonib_clustering.ipynb   # Clustering → output/
8. Cluster_nonIB/04_nonib_cluster_profiling.ipynb  # Profiling → output/
```

### Chạy Demo

```bash
cd Demo
streamlit run streamlit_app.py
```

Hoặc chạy trực tiếp:

```bash
python Demo/streamlit_app.py
```

Demo sẽ tự mở trình duyệt tại `http://localhost:8501` với 2 tab:
- **NBFO Recommendation** — Tra cứu propensity score và đề xuất sản phẩm cho từng khách hàng IB
- **non-IB Clustering** — Xem chân dung persona, so sánh với cluster, và gợi ý campaign

> **Lưu ý:** Demo cần các file output từ notebooks (bước 3–8). Nếu chạy lần đầu, hãy chạy notebooks trước.

---

## Kết quả chính

### NBFO
- Mô hình XGBoost đạt **PR-AUC 0.598** trên tập test (month 10), phù hợp với bài toán highly imbalanced
- Propensity score đã được calibrate bằng Platt Scaling, sẵn sàng cho downstream tasks
- SHAP cho thấy các features quan trọng nhất: product affinity, deposit balance, activity frequency

### Clustering
- 8 personas rõ ràng từ "Traditional" (63%) đến "High-Value Heavy Borrower" (0.5%)
- Silhouette Score **0.696** — phân tách tốt giữa các cụm
- Bootstrap ARI **0.962** — cluster rất ổn định qua các lần sampling
- Mỗi persona kèm chiến lược tiếp cận cụ thể (kênh, thông điệp, hành động đầu tiên)

---

## Thành viên nhóm

| Thành viên | Vai trò |
|---|---|
| Nguyễn Tiến Mạnh | Leader | 
| Phạm Văn Linh | Member |
| Trần Đức Lập | Member |

---

## Lưu ý bảo mật

Dữ liệu được cung cấp bởi BTC G'CONTEST 2026 và thuộc quyền bảo mật. Không chia sẻ dữ liệu hoặc kết quả ra bên ngoài.
