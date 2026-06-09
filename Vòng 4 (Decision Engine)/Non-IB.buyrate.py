Th"""
G'CONTEST 2026 — BA QUẢ TÁO
Non-IB Buy Rate Estimation — Hướng 1: IB Segment Proxy
Dual-method: Rule-based + Euclidean Distance (conservative tie-breaking)

Logic:
    Non-IB cluster X → map về IB segment Y → buy rate của Y = proxy buy rate của X
    CLV = buy rate × 5,000,000đ

Chạy: /opt/anaconda3/bin/python "Non-IB.buyrate.py"
"""

import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist

BASE = "/Users/mah/Library/Mobile Documents/com~apple~CloudDocs/G'Con 26"

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────

base     = pd.read_parquet(f'{BASE}/cleaned_data/base_full.parquet')
mi       = pd.read_parquet(f'{BASE}/gcon-main/NBFO_IB/processed_data/gcon_model_input.parquet')
nonib_cl = pd.read_parquet(f'{BASE}/gcon-main/Cluster_nonIB/output/nonib_clustered.parquet')

base['IB_REGISTER_DATE'] = pd.to_datetime(base['IB_REGISTER_DATE'])


# ─────────────────────────────────────────────────────────────────────────────
# 2. IB SEGMENTS — 2019 ONLY
#    Loại trừ KH đăng ký IB năm 2020/2021 vì tại thời điểm quan sát (2019)
#    họ vẫn là Non-IB → không đại diện cho hành vi IB ổn định
# ─────────────────────────────────────────────────────────────────────────────

FEATURES = ['has_loan', 'has_td', 'has_card',
            'AVG_LOAN_AMOUNT', 'AVG_TD_BALANCE', 'product_depth']

ib = base[
    (base['is_ib'] == 1) &
    (base['IB_REGISTER_DATE'].dt.year <= 2019)
].copy()

def assign_seg(r):
    """
    Priority cascade — KH rơi vào segment đầu tiên thỏa điều kiện.
    Thứ tự: N3 → V1 → V2 → V3 → N1 → N2
    """
    if r['login_count'] == 0:                                           return 'N3_Dormant'
    if r['AVG_LOAN_AMOUNT'] > 500_000_000:                              return 'V1_HV_Borrower'
    if r['AVG_TD_BALANCE'] > 100_000_000 and r['has_loan'] == 0:        return 'V2_Conservative'
    if r['product_depth'] >= 3 and r['AVG_TD_BALANCE'] > 200_000_000:  return 'V3_Multi_Premium'
    if r['has_card'] == 1 and r['has_loan'] == 1:                       return 'N1_Active_Digital'
    return 'N2_Semi_Digital'

ib['segment'] = ib.apply(assign_seg, axis=1)

# VIP/Normal classification — dùng cho FUM threshold
VIP_SEGS    = {'V1_HV_Borrower', 'V2_Conservative', 'V3_Multi_Premium'}
NORMAL_SEGS = {'N1_Active_Digital', 'N2_Semi_Digital'}


# ─────────────────────────────────────────────────────────────────────────────
# 3. IB BUY RATES
#    TARGET_PRODUCT_OPEN_COUNT_HISTORY > 0 = đã mua ít nhất 1 sản phẩm NBFO
# ─────────────────────────────────────────────────────────────────────────────

bought = (
    mi.groupby('CUSTOMER_NUMBER')['TARGET_PRODUCT_OPEN_COUNT_HISTORY']
    .max() > 0
).reset_index()
bought.columns = ['CUSTOMER_NUMBER', 'bought']

ib = ib.merge(bought, on='CUSTOMER_NUMBER', how='left')
ib['bought'] = ib['bought'].fillna(False).infer_objects(copy=False)

seg_rates    = ib.groupby('segment')['bought'].mean()
ib_centroids = ib.groupby('segment')[FEATURES].mean()

# Loại N3_Dormant khỏi tập mapping — không dùng làm đích
ib_cands = ib_centroids.drop('N3_Dormant')

print("=== IB Segment Buy Rates (2019 only) ===")
seg_order_ib = ['V3_Multi_Premium', 'V1_HV_Borrower', 'V2_Conservative',
                'N1_Active_Digital', 'N2_Semi_Digital', 'N3_Dormant']
for seg in seg_order_ib:
    n    = (ib['segment'] == seg).sum()
    rate = seg_rates[seg]
    grp  = 'VIP' if seg in VIP_SEGS else ('Normal' if seg in NORMAL_SEGS else 'Onboarding')
    print(f"  {seg:<22}  N={n:>7,}  rate={rate:.2%}  [{grp}]")


# ─────────────────────────────────────────────────────────────────────────────
# 4. NON-IB CLUSTER CENTROIDS
# ─────────────────────────────────────────────────────────────────────────────

CLUSTER_NAMES = {
    -1: 'P0_Dormant',
     0: 'C0_Traditional',
     1: 'C1_HV_Traditional',
     2: 'C2_Senior_HV',
     3: 'C3_Ultra_Saver',
     4: 'C4_Multi_Saver',
     5: 'C5_HV_Saver',
     6: 'C6_Stable_Senior',
     7: 'C7_HV_Borrower',
}

CLUSTER_SIZES = {
    'P0_Dormant': 24_570, 'C0_Traditional': 80_041, 'C1_HV_Traditional': 493,
    'C2_Senior_HV': 562,  'C3_Ultra_Saver': 912,    'C4_Multi_Saver': 800,
    'C5_HV_Saver': 13_180,'C6_Stable_Senior': 5_792, 'C7_HV_Borrower': 653,
}

nonib_base = base[base['is_ib'] == 0][['CUSTOMER_NUMBER'] + FEATURES].copy()
nm = nonib_cl[['CUSTOMER_NUMBER', 'CLUSTER']].merge(nonib_base, on='CUSTOMER_NUMBER', how='left')
nm['cluster_name'] = nm['CLUSTER'].map(CLUSTER_NAMES)
nonib_centroids = nm.groupby('cluster_name')[FEATURES].mean()


# ─────────────────────────────────────────────────────────────────────────────
# 5. CÁCH 1 — RULE-BASED MAPPING
#    Áp bộ quy tắc phân loại IB lên centroid của từng Non-IB cluster
#    Ngưỡng điều chỉnh cho centroid (số thập phân thay vì nhị phân)
# ─────────────────────────────────────────────────────────────────────────────

def rule_map(row):
    loan  = row['AVG_LOAN_AMOUNT']
    td    = row['AVG_TD_BALANCE']
    dep   = row['product_depth']
    card  = row['has_card']
    has_l = row['has_loan']
    if loan > 500_000_000:          return 'V1_HV_Borrower'
    if td > 100_000_000 and has_l == 0: return 'V2_Conservative'
    if dep >= 3 and td > 200_000_000:   return 'V3_Multi_Premium'
    if card > 0.3 and has_l > 0.3:      return 'N1_Active_Digital'
    return 'N2_Semi_Digital'

rule_results = {c: rule_map(nonib_centroids.loc[c]) for c in nonib_centroids.index}


# ─────────────────────────────────────────────────────────────────────────────
# 6. CÁCH 2 — EUCLIDEAN DISTANCE MAPPING (Z-score normalized)
#    Z-score normalization để tránh AVG_LOAN_AMOUNT (đơn vị VND lớn)
#    át hoàn toàn has_card (0-1) trong tính khoảng cách
# ─────────────────────────────────────────────────────────────────────────────

all_data   = pd.concat([ib_cands, nonib_centroids])
mu         = all_data.mean()
sd         = all_data.std().replace(0, 1)
ib_norm    = ((ib_cands - mu) / sd).astype(float)
nonib_norm = ((nonib_centroids - mu) / sd).astype(float)

dist_matrix = pd.DataFrame(
    cdist(nonib_norm.values, ib_norm.values),
    index=nonib_norm.index,
    columns=ib_norm.index,
)

dist_results = dist_matrix.idxmin(axis=1).to_dict()
dist_vals    = dist_matrix.min(axis=1).to_dict()


# ─────────────────────────────────────────────────────────────────────────────
# 7. KẾT HỢP — CONSERVATIVE TIE-BREAKING
#    Đồng thuận → dùng rate đó (confidence cao)
#    Lệch nhau → lấy min (conservative, an toàn khi defend)
# ─────────────────────────────────────────────────────────────────────────────

seg_order_nonib = [
    'C0_Traditional', 'C1_HV_Traditional', 'C2_Senior_HV',
    'C3_Ultra_Saver',  'C4_Multi_Saver',   'C5_HV_Saver',
    'C6_Stable_Senior','C7_HV_Borrower',   'P0_Dormant',
]

print(f"\n{'='*115}")
print("NON-IB BUY RATE ESTIMATION — Dual Method")
print(f"{'='*115}")
print(f"{'Cluster':<22} {'N':>7}  {'Rule→Seg':<22} {'Dist→Seg':<22} {'Dist':>5}  "
      f"{'Agree':>5}  {'R_Rate':>7}  {'D_Rate':>7}  {'Final':>7}  {'Type':<8}  {'Note'}")
print(f"{'-'*115}")

final_rates   = {}
final_types   = {}
final_segs    = {}

for c in seg_order_nonib:
    if c == 'P0_Dormant':
        print(f"{'P0_Dormant':<22} {24570:>7,}  {'—':<22} {'—':<22} {'—':>5}  "
              f"{'—':>5}  {'—':>7}  {'—':>7}  {'—':>7}  {'IGNORE':<8}  zero products/balance")
        final_rates[c] = None
        final_types[c] = 'IGNORE'
        continue

    r_seg  = rule_results[c]
    d_seg  = dist_results[c]
    d_val  = dist_vals[c]
    r_rate = seg_rates[r_seg]
    d_rate = seg_rates[d_seg]
    agree  = r_seg == d_seg

    if agree:
        final = r_rate
        note  = ''
    else:
        final = min(r_rate, d_rate)
        note  = '<-- conservative'

    # VIP/Normal dựa trên IB segment được map tới
    mapped_seg = r_seg if agree else (r_seg if r_rate == final else d_seg)
    seg_type   = 'VIP' if mapped_seg in VIP_SEGS else 'Normal'

    final_rates[c] = final
    final_types[c] = seg_type
    final_segs[c]  = mapped_seg

    print(f"{c:<22} {CLUSTER_SIZES[c]:>7,}  {r_seg:<22} {d_seg:<22} {d_val:>5.2f}  "
          f"{'YES' if agree else 'NO':>5}  {r_rate:>6.2%}  {d_rate:>6.2%}  "
          f"{final:>6.2%}  {seg_type:<8}  {note}")


# ─────────────────────────────────────────────────────────────────────────────
# 8. FINAL CLV PER NON-IB CLUSTER
# ─────────────────────────────────────────────────────────────────────────────

CLV_PER_PRODUCT = 5_000_000

print(f"\n{'='*90}")
print("FINAL BUY RATE & CLV PER NON-IB CLUSTER")
print(f"{'='*90}")
print(f"{'Cluster':<22} {'N':>7}  {'Type':<8}  {'Buy Rate':>9}  {'CLV':>12}  {'Maps to IB'}")
print(f"{'-'*90}")

for c in seg_order_nonib:
    n    = CLUSTER_SIZES[c]
    rate = final_rates[c]
    typ  = final_types[c]
    seg  = final_segs.get(c, '—')

    if rate is None:
        print(f"{c:<22} {n:>7,}  {'IGNORE':<8}  {'N/A':>9}  {'N/A':>12}  —")
    else:
        clv = rate * CLV_PER_PRODUCT
        print(f"{c:<22} {n:>7,}  {typ:<8}  {rate:>8.2%}  {clv:>11,.0f}đ  {seg}")

# Summary by type
print(f"\n{'='*50}")
print("SUMMARY BY TYPE")
print(f"{'='*50}")
vip_clusters    = [c for c, t in final_types.items() if t == 'VIP']
normal_clusters = [c for c, t in final_types.items() if t == 'Normal']
ignore_clusters = [c for c, t in final_types.items() if t == 'IGNORE']

print(f"\nVIP ({len(vip_clusters)} clusters):")
for c in vip_clusters:
    print(f"  {c:<22}  buy rate={final_rates[c]:.2%}  CLV={final_rates[c]*CLV_PER_PRODUCT:,.0f}đ")

print(f"\nNormal ({len(normal_clusters)} clusters):")
for c in normal_clusters:
    print(f"  {c:<22}  buy rate={final_rates[c]:.2%}  CLV={final_rates[c]*CLV_PER_PRODUCT:,.0f}đ")

print(f"\nIgnore ({len(ignore_clusters)} clusters):")
for c in ignore_clusters:
    print(f"  {c}")
