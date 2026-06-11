import os
import warnings

import numpy as np
import pandas as pd

from decision_config import (
    DEFAULT_SEGMENT,
    FUM_MATRIX,
    VIP_SEGMENTS,
    add_ib_segments,
    attach_ib_register_date,
    calculate_asset_score,
    segment_for_economics,
    solve_channel_milp,
)


warnings.filterwarnings('ignore')
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
BUDGET_LIMIT = 450_000_000
HUMAN_CAP = 6_000

channels = {
    'SMS': {'cost': 5_000, 'cr': 0.02},
    'Telesales': {'cost': 50_000, 'cr': 0.05},
    'RM': {'cost': 2_000_000, 'cr': 0.15},
}
channel_names = np.array(['SMS', 'Telesales', 'RM'])
channel_costs = np.array([channels[ch]['cost'] for ch in channel_names])


def frame_to_markdown(df):
    rows = [[str(col) for col in df.columns]]
    rows.extend(df.fillna('').astype(str).values.tolist())
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    header = '| ' + ' | '.join(rows[0][i].ljust(widths[i]) for i in range(len(widths))) + ' |'
    separator = '| ' + ' | '.join('-' * widths[i] for i in range(len(widths))) + ' |'
    body = ['| ' + ' | '.join(row[i].ljust(widths[i]) for i in range(len(widths))) + ' |' for row in rows[1:]]
    return '\n'.join([header, separator] + body)


def load_master_data():
    path_ib_prob = os.path.join(
        BASE_DIR,
        "NBFO_IB",
        "saved_models",
        "gcon_test_scores_best_xgboost_calibrated_sigmoid.parquet",
    )
    df_prob = pd.read_parquet(path_ib_prob)
    df_prob = df_prob.rename(
        columns={
            'SUBSCRIPTION_PROPENSITY': 'PROBABILITY',
            'PRODUCT_NAME': 'RECOMMENDED_PRODUCT',
        }
    )

    path_ib_features = os.path.join(BASE_DIR, "NBFO_IB", "processed_data", "gcon_model_input.parquet")
    df_features = pd.read_parquet(path_ib_features)
    if 'MONTH' in df_features.columns:
        df_features = df_features.sort_values('MONTH').groupby('CUSTOMER_NUMBER').last().reset_index()
    else:
        df_features = df_features.groupby('CUSTOMER_NUMBER').last().reset_index()
    df_features = attach_ib_register_date(df_features, BASE_DIR)
    df_segments = add_ib_segments(df_features)

    asset_cols = ['AVG_TD_BALANCE', 'AVG_CA_BALANCE', 'AVG_LOAN_AMOUNT']
    segment_cols = ['CUSTOMER_NUMBER', 'SEGMENT', 'MAPPED_IB_SEGMENT', 'CUSTOMER_TYPE']
    segment_cols += [col for col in asset_cols if col in df_segments.columns]
    df = df_prob.merge(df_segments[segment_cols], on='CUSTOMER_NUMBER', how='left')
    df['SEGMENT'] = df['SEGMENT'].fillna(DEFAULT_SEGMENT)
    df['MAPPED_IB_SEGMENT'] = df['MAPPED_IB_SEGMENT'].fillna(df['SEGMENT'])
    df['CUSTOMER_TYPE'] = df['CUSTOMER_TYPE'].fillna('IB')
    df['SEGMENT_CLUSTER'] = df['SEGMENT']
    df['ECONOMIC_SEGMENT'] = df.apply(segment_for_economics, axis=1)
    df['IS_VIP'] = df['ECONOMIC_SEGMENT'].isin(VIP_SEGMENTS)
    df['TP'] = df['ECONOMIC_SEGMENT'].map(lambda x: FUM_MATRIX.get(x, FUM_MATRIX[DEFAULT_SEGMENT])['TP'])
    df['FP'] = df['ECONOMIC_SEGMENT'].map(lambda x: FUM_MATRIX.get(x, FUM_MATRIX[DEFAULT_SEGMENT])['FP'])
    df['FN'] = df['ECONOMIC_SEGMENT'].map(lambda x: FUM_MATRIX.get(x, FUM_MATRIX[DEFAULT_SEGMENT])['FN'])
    df['ASSET_SCORE'] = calculate_asset_score(df)
    return df


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
        emu = uplift * (tp_array - fn_array) + fp_probability * fp_array - channels[ch_name]['cost']
        if ch_name == 'RM':
            emu = np.where(df['IS_VIP'], emu, -999_999_999)
        emus.append(emu + 1e-6 * df['ASSET_SCORE'])
    return np.vstack(emus).T


def load_optimized_thresholds():
    path = os.path.join(BASE_DIR, 'optimized_thresholds_ib.csv')
    if not os.path.exists(path):
        return None
    thresholds = pd.read_csv(path)
    thresholds = thresholds[thresholds['Threshold'].notna()].copy()
    thresholds = thresholds[
        thresholds['Optimization Status'].isin(['CAC feasible', 'Max EMU, CAC cap unmet'])
    ]
    return {
        (row['Segment'], row['Product_or_Risk'], row['Channel']): float(row['Threshold'])
        for _, row in thresholds.iterrows()
    }


def build_eligibility_matrix(df, thresholds):
    eligible = np.full((len(df), len(channel_names)), False)
    if thresholds is None:
        return np.ones_like(eligible, dtype=bool)

    for channel_idx, ch_name in enumerate(channel_names):
        keys = list(zip(df['MAPPED_IB_SEGMENT'], df['RECOMMENDED_PRODUCT'], [ch_name] * len(df)))
        threshold_values = pd.Series([thresholds.get(key, np.nan) for key in keys], index=df.index)
        eligible[:, channel_idx] = threshold_values.notna() & (df['PROBABILITY'] >= threshold_values.astype(float))
    return eligible


def compress_to_customer_channel(df, emu_matrix):
    customers = (
        df.sort_values('CUSTOMER_NUMBER')
        .drop_duplicates('CUSTOMER_NUMBER')
        [[
            'CUSTOMER_NUMBER',
            'CUSTOMER_TYPE',
            'SEGMENT_CLUSTER',
            'MAPPED_IB_SEGMENT',
            'ECONOMIC_SEGMENT',
            'IS_VIP',
        ]]
        .reset_index(drop=True)
    )
    customer_pos = {customer: idx for idx, customer in enumerate(customers['CUSTOMER_NUMBER'])}
    compressed_emu = np.full((len(customers), len(channel_names)), -999_999_999.0)
    selected_product = np.full((len(customers), len(channel_names)), None, dtype=object)
    selected_probability = np.full((len(customers), len(channel_names)), np.nan)

    for channel_idx, ch_name in enumerate(channel_names):
        work = df[['CUSTOMER_NUMBER', 'RECOMMENDED_PRODUCT', 'PROBABILITY']].copy()
        work['EMU'] = emu_matrix[:, channel_idx]
        work = work[np.isfinite(work['EMU']) & (work['EMU'] > 0)]
        if work.empty:
            continue
        idx = work.groupby('CUSTOMER_NUMBER')['EMU'].idxmax()
        best = work.loc[idx]
        for _, row in best.iterrows():
            pos = customer_pos[row['CUSTOMER_NUMBER']]
            compressed_emu[pos, channel_idx] = row['EMU']
            selected_product[pos, channel_idx] = row['RECOMMENDED_PRODUCT']
            selected_probability[pos, channel_idx] = row['PROBABILITY']

    return customers, compressed_emu, selected_product, selected_probability


def solve_allocation(emu_matrix):
    result = solve_channel_milp(
        emu_matrix,
        channel_costs,
        BUDGET_LIMIT,
        np.array([0, 1, 1]),
        HUMAN_CAP,
    )
    allocations = result['allocations']
    print(f"MILP allocation status: {result['status']} - {result['message']}")

    total_profit = np.sum(allocations * emu_matrix)
    counts = np.sum(allocations, axis=0)
    cost = np.sum(allocations * channel_costs)
    return total_profit, counts, cost, allocations


print("Loading IB product-level data...")
df_master = load_master_data()
print("Product-level data shape:", df_master.shape)

threshold_matrix = load_optimized_thresholds()
if threshold_matrix is None:
    print("WARNING: optimized_thresholds_ib.csv not found; no optimized threshold filter applied.")
else:
    print(f"Loaded optimized IB thresholds: {len(threshold_matrix):,} persona-product-channel rows")

print("Calculating Baseline EMU and applying optimized thresholds...")
eligibility_matrix = build_eligibility_matrix(df_master, threshold_matrix)
emu_product = np.where(eligibility_matrix, calculate_emu(df_master), -999_999_999)
df_customer, emu_baseline, product_by_channel, prob_by_channel = compress_to_customer_channel(df_master, emu_product)

profit_base, counts_base, cost_base, alloc_base = solve_allocation(emu_baseline)

assigned_indices = np.where(alloc_base == 1)
df_customer['RECOMMENDED_CHANNEL'] = 'None'
df_customer['RECOMMENDED_PRODUCT'] = None
df_customer['PROBABILITY'] = np.nan
df_customer['CAMPAIGN_COST'] = 0
df_customer['CAMPAIGN_EMU'] = 0.0
df_customer.loc[assigned_indices[0], 'RECOMMENDED_CHANNEL'] = channel_names[assigned_indices[1]]
for row_pos, channel_pos in zip(*assigned_indices):
    df_customer.loc[row_pos, 'RECOMMENDED_PRODUCT'] = product_by_channel[row_pos, channel_pos]
    df_customer.loc[row_pos, 'PROBABILITY'] = prob_by_channel[row_pos, channel_pos]
    df_customer.loc[row_pos, 'CAMPAIGN_COST'] = channel_costs[channel_pos]
    df_customer.loc[row_pos, 'CAMPAIGN_EMU'] = emu_baseline[row_pos, channel_pos]

output_cols = [
    'CUSTOMER_NUMBER',
    'CUSTOMER_TYPE',
    'SEGMENT_CLUSTER',
    'MAPPED_IB_SEGMENT',
    'RECOMMENDED_PRODUCT',
    'PROBABILITY',
    'RECOMMENDED_CHANNEL',
    'CAMPAIGN_COST',
    'CAMPAIGN_EMU',
]
out_alloc = os.path.join(BASE_DIR, "final_allocations.csv")
df_customer[output_cols].to_csv(out_alloc, index=False)

print("Baseline Results:")
print(f"Customers: {len(df_customer):,}")
print(f"Profit: {profit_base:,.0f}")
print(f"Cost: {cost_base:,.0f}")
print(f"Allocations: SMS={counts_base[0]}, Tele={counts_base[1]}, RM={counts_base[2]}")

print("\nSample Allocation Output:")
print(frame_to_markdown(df_customer[output_cols].head(5)))

print("\nCalculating Stress-Test (FP VIP cost +20%, CR -15% for Telesales and RM)...")
stress_cr_multipliers = {'SMS': 1.0, 'Telesales': 0.85, 'RM': 0.85}
profit_stress = 0.0
paid_customer = df_customer[df_customer['RECOMMENDED_CHANNEL'] != 'None'].copy()
for _, row in paid_customer.iterrows():
    ch_name = row['RECOMMENDED_CHANNEL']
    economics = FUM_MATRIX.get(row['ECONOMIC_SEGMENT'], FUM_MATRIX[DEFAULT_SEGMENT])
    p = row['PROBABILITY']
    cr = channels[ch_name]['cr'] * stress_cr_multipliers.get(ch_name, 1.0)
    fp = economics['FP'] * (1.2 if row['IS_VIP'] else 1.0)
    uplift = 4 * p * (1 - p) * cr
    fp_probability = 1 - p - uplift
    profit_stress += uplift * (economics['TP'] - economics['FN']) + fp_probability * fp - channels[ch_name]['cost']
counts_stress = np.sum(alloc_base, axis=0)
cost_stress = np.sum(alloc_base * channel_costs)

print("Stress Results:")
print(f"Profit: {profit_stress:,.0f}")
print(f"Cost: {cost_stress:,.0f}")
print(f"Allocations: SMS={counts_stress[0]}, Tele={counts_stress[1]}, RM={counts_stress[2]}")
