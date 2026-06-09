import os
import warnings

import numpy as np
import pandas as pd

from decision_config import (
    DEFAULT_SEGMENT,
    FUM_MATRIX,
    VIP_SEGMENTS,
    add_ib_segments,
    segment_for_economics,
)


warnings.filterwarnings('ignore')
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
BUDGET_LIMIT = 700_000_000


def frame_to_markdown(df):
    rows = [[str(col) for col in df.columns]]
    rows.extend(df.fillna('').astype(str).values.tolist())
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    header = '| ' + ' | '.join(rows[0][i].ljust(widths[i]) for i in range(len(widths))) + ' |'
    separator = '| ' + ' | '.join('-' * widths[i] for i in range(len(widths))) + ' |'
    body = ['| ' + ' | '.join(row[i].ljust(widths[i]) for i in range(len(widths))) + ' |' for row in rows[1:]]
    return '\n'.join([header, separator] + body)


channels = {
    'SMS': {'cost': 5_000, 'cr': 0.02},
    'Telesales': {'cost': 50_000, 'cr': 0.05},
    'RM': {'cost': 2_000_000, 'cr': 0.15},
}
channel_names = np.array(['SMS', 'Telesales', 'RM'])
channel_costs = np.array([channels[ch]['cost'] for ch in channel_names])


def load_master_data():
    path_ib_prob = os.path.join(
        BASE_DIR,
        "NBFO_IB",
        "saved_models",
        "gcon_test_scores_best_xgboost_calibrated_sigmoid.parquet",
    )
    df_ib = pd.read_parquet(path_ib_prob)
    idx_max_prob = df_ib.groupby('CUSTOMER_NUMBER')['SUBSCRIPTION_PROPENSITY'].idxmax()
    df_ib_prob = df_ib.loc[idx_max_prob, ['CUSTOMER_NUMBER', 'PRODUCT_NAME', 'SUBSCRIPTION_PROPENSITY']].copy()
    df_ib_prob.rename(
        columns={'SUBSCRIPTION_PROPENSITY': 'PROBABILITY', 'PRODUCT_NAME': 'RECOMMENDED_PRODUCT'},
        inplace=True,
    )

    path_ib_features = os.path.join(BASE_DIR, "NBFO_IB", "processed_data", "gcon_model_input.parquet")
    df_ib_features = pd.read_parquet(path_ib_features)
    if 'MONTH' in df_ib_features.columns:
        df_ib_features = df_ib_features.sort_values('MONTH').groupby('CUSTOMER_NUMBER').last().reset_index()
    else:
        df_ib_features = df_ib_features.groupby('CUSTOMER_NUMBER').last().reset_index()
    df_ib_segments = add_ib_segments(df_ib_features)
    df_ib_prob = df_ib_prob.merge(
        df_ib_segments[['CUSTOMER_NUMBER', 'SEGMENT', 'MAPPED_IB_SEGMENT', 'CUSTOMER_TYPE']],
        on='CUSTOMER_NUMBER',
        how='left',
    )
    df_ib_prob['SEGMENT'] = df_ib_prob['SEGMENT'].fillna(DEFAULT_SEGMENT)
    df_ib_prob['MAPPED_IB_SEGMENT'] = df_ib_prob['MAPPED_IB_SEGMENT'].fillna(df_ib_prob['SEGMENT'])
    df_ib_prob['CUSTOMER_TYPE'] = df_ib_prob['CUSTOMER_TYPE'].fillna('IB')
    df_ib_prob['SEGMENT_CLUSTER'] = df_ib_prob['SEGMENT']
    df_ib_prob['BUY_RATE_PROXY'] = np.nan

    common_cols = [
        'CUSTOMER_NUMBER',
        'CUSTOMER_TYPE',
        'SEGMENT',
        'SEGMENT_CLUSTER',
        'MAPPED_IB_SEGMENT',
        'RECOMMENDED_PRODUCT',
        'PROBABILITY',
        'BUY_RATE_PROXY',
    ]
    return df_ib_prob[common_cols].copy()


print("Loading data...")
try:
    df_master = load_master_data()
except Exception as e:
    print("Error loading real data:", e)
    import sys
    sys.exit(1)

print("Master data shape:", df_master.shape)

df_master['ECONOMIC_SEGMENT'] = df_master.apply(segment_for_economics, axis=1)
df_master['IS_VIP'] = df_master['ECONOMIC_SEGMENT'].isin(VIP_SEGMENTS)
df_master['TP'] = df_master['ECONOMIC_SEGMENT'].map(lambda x: FUM_MATRIX.get(x, FUM_MATRIX[DEFAULT_SEGMENT])['TP'])
df_master['FP'] = df_master['ECONOMIC_SEGMENT'].map(lambda x: FUM_MATRIX.get(x, FUM_MATRIX[DEFAULT_SEGMENT])['FP'])
df_master['FN'] = df_master['ECONOMIC_SEGMENT'].map(lambda x: FUM_MATRIX.get(x, FUM_MATRIX[DEFAULT_SEGMENT])['FN'])
df_master['ASSET_SCORE'] = (df_master['CUSTOMER_NUMBER'] % 10000) / 10000.0


def threshold_emu_formula(p, cr, cost, tp_val, fn_val, fp_val):
    uplift = 4 * p * (1 - p) * cr
    fp_probability = 1 - p - uplift
    return uplift * (tp_val - fn_val) + fp_probability * fp_val - cost


def calculate_channel_thresholds():
    thresholds = {}
    ps = np.linspace(0, 1, 5000)
    for segment, economics in FUM_MATRIX.items():
        thresholds[segment] = {}
        for ch_name, ch_data in channels.items():
            if ch_name == 'RM' and segment not in VIP_SEGMENTS:
                thresholds[segment][ch_name] = None
                continue
            emus = threshold_emu_formula(
                ps,
                ch_data['cr'],
                ch_data['cost'],
                economics['TP'],
                economics['FN'],
                economics['FP'],
            )
            valid_ps = ps[emus >= 0]
            thresholds[segment][ch_name] = float(valid_ps[0]) if len(valid_ps) > 0 else None
    return thresholds


def load_thresholds_from_csv():
    threshold_path = os.path.join(BASE_DIR, "thresholds.csv")
    if not os.path.exists(threshold_path):
        return None

    df_thresholds = pd.read_csv(threshold_path)
    segment_col = 'Segment' if 'Segment' in df_thresholds.columns else 'Persona'
    thresholds = {}
    column_map = {'SMS': 'Threshold SMS', 'Telesales': 'Threshold Telesales', 'RM': 'Threshold RM'}
    for _, row in df_thresholds.iterrows():
        segment = row[segment_col]
        thresholds[segment] = {}
        for ch_name, col_name in column_map.items():
            value = row[col_name]
            if pd.isna(value) or str(value) in {'N/A', 'No ROI'}:
                thresholds[segment][ch_name] = None
            else:
                thresholds[segment][ch_name] = float(value)
    return thresholds


threshold_matrix = load_thresholds_from_csv() or calculate_channel_thresholds()


def build_eligibility_matrix(df, thresholds):
    eligible = np.full((len(df), len(channel_names)), False)
    for channel_idx, ch_name in enumerate(channel_names):
        segment_thresholds = df['ECONOMIC_SEGMENT'].map(
            lambda segment: thresholds.get(segment, thresholds[DEFAULT_SEGMENT]).get(ch_name)
        )
        has_threshold = segment_thresholds.notna()
        eligible[:, channel_idx] = has_threshold & (df['PROBABILITY'] >= segment_thresholds.astype(float))
    return eligible


def apply_threshold_filter(value_matrix, eligible_matrix):
    return np.where(eligible_matrix, value_matrix, -999_999_999)


def calculate_emu(df, fp_multiplier=1.0, cr_multipliers=None):
    if cr_multipliers is None:
        cr_multipliers = {'SMS': 1.0, 'Telesales': 1.0, 'RM': 1.0}

    p_base = df['PROBABILITY']
    tp_array = df['TP']
    fn_array = df['FN']
    fp_array = np.where(df['IS_VIP'], df['FP'] * fp_multiplier, df['FP'])

    emus = []
    for ch_name in channel_names:
        uplift = 4 * p_base * (1 - p_base) * channels[ch_name]['cr'] * cr_multipliers.get(ch_name, 1.0)
        fp_probability = 1 - p_base - uplift
        emus.append(uplift * (tp_array - fn_array) + fp_probability * fp_array - channels[ch_name]['cost'])

    tie_breaker = 1e-6 * df['ASSET_SCORE']
    return np.vstack([emu + tie_breaker for emu in emus]).T


eligibility_matrix = build_eligibility_matrix(df_master, threshold_matrix)
print("Calculating Baseline EMU (Uplift Mode)...")
emu_baseline = apply_threshold_filter(calculate_emu(df_master), eligibility_matrix)


def solve_allocation(emu_matrix):
    n = len(df_master)
    allocations = np.full((n, 3), 0)
    best_channels = np.argmax(emu_matrix, axis=1)
    max_emus = np.max(emu_matrix, axis=1)
    valid = max_emus > 0

    tele_rm_indices = np.where(valid & ((best_channels == 1) | (best_channels == 2)))[0]
    tele_rm_indices = tele_rm_indices[np.argsort(-max_emus[tele_rm_indices])]
    selected = tele_rm_indices[:10000]
    rejected = tele_rm_indices[10000:]

    for idx in selected:
        allocations[idx, best_channels[idx]] = 1

    sms_emu = emu_matrix[:, 0]
    fallback_sms = rejected[sms_emu[rejected] > 0]
    allocations[fallback_sms, 0] = 1

    native_sms = np.where(valid & (best_channels == 0))[0]
    allocations[native_sms, 0] = 1

    total_cost = np.sum(allocations * channel_costs)
    if total_cost > BUDGET_LIMIT:
        assigned = np.where(allocations.sum(axis=1) > 0)[0]
        assigned_costs = allocations[assigned] @ channel_costs
        assigned_emus = emu_matrix[assigned, np.argmax(allocations[assigned], axis=1)]
        efficiency = assigned_emus / assigned_costs
        sorted_keep_idx = assigned[np.argsort(-efficiency)]

        current_cost = 0
        keep_list = []
        for idx in sorted_keep_idx:
            c = allocations[idx] @ channel_costs
            if current_cost + c <= BUDGET_LIMIT:
                current_cost += c
                keep_list.append(idx)

        final_alloc = np.zeros((n, 3))
        final_alloc[keep_list] = allocations[keep_list]
        allocations = final_alloc

    total_profit = np.sum(allocations * emu_matrix)
    counts = np.sum(allocations, axis=0)
    cost = np.sum(allocations * channel_costs)
    return total_profit, counts, cost, allocations


profit_base, counts_base, cost_base, alloc_base = solve_allocation(emu_baseline)

assigned_indices = np.where(alloc_base == 1)
df_master['RECOMMENDED_CHANNEL'] = 'None'
df_master.loc[assigned_indices[0], 'RECOMMENDED_CHANNEL'] = channel_names[assigned_indices[1]]

output_cols = [
    'CUSTOMER_NUMBER',
    'CUSTOMER_TYPE',
    'SEGMENT_CLUSTER',
    'MAPPED_IB_SEGMENT',
    'RECOMMENDED_PRODUCT',
    'PROBABILITY',
    'RECOMMENDED_CHANNEL',
]
out_alloc = os.path.join(BASE_DIR, "final_allocations.csv")
df_master[output_cols].to_csv(out_alloc, index=False)

print("Baseline Results:")
print(f"Profit: {profit_base:,.0f}")
print(f"Cost: {cost_base:,.0f}")
print(f"Allocations: SMS={counts_base[0]}, Tele={counts_base[1]}, RM={counts_base[2]}")

print("\nSample Allocation Output:")
print(frame_to_markdown(df_master[output_cols].head(5)))

print("\nCalculating Stress-Test (FP VIP cost +20%, CR -15% for Telesales and RM)...")
stress_cr_multipliers = {'SMS': 1.0, 'Telesales': 0.85, 'RM': 0.85}
emu_stress = apply_threshold_filter(
    calculate_emu(df_master, fp_multiplier=1.2, cr_multipliers=stress_cr_multipliers),
    eligibility_matrix,
)
profit_stress, counts_stress, cost_stress, alloc_stress = solve_allocation(emu_stress)

print("Stress Results:")
print(f"Profit: {profit_stress:,.0f}")
print(f"Cost: {cost_stress:,.0f}")
print(f"Allocations: SMS={counts_stress[0]}, Tele={counts_stress[1]}, RM={counts_stress[2]}")
