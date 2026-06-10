import os

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


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
BUDGET_LIMIT = 700_000_000
HUMAN_CAP = 5_000


def frame_to_markdown(df):
    rows = [[''] + [str(col) for col in df.columns]]
    rows.extend([[str(idx)] + [str(value) for value in row] for idx, row in df.iterrows()])
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    header = '| ' + ' | '.join(rows[0][i].ljust(widths[i]) for i in range(len(widths))) + ' |'
    separator = '| ' + ' | '.join('-' * widths[i] for i in range(len(widths))) + ' |'
    body = ['| ' + ' | '.join(row[i].ljust(widths[i]) for i in range(len(widths))) + ' |' for row in rows[1:]]
    return '\n'.join([header, separator] + body)


channels_base = {
    'SMS': {'cost': 5_000, 'cr': 0.02},
    'Telesales': {'cost': 50_000, 'cr': 0.05},
    'RM': {'cost': 2_000_000, 'cr': 0.15},
}
channel_names = np.array(['SMS', 'Telesales', 'RM'])
channel_costs = np.array([channels_base[ch]['cost'] for ch in channel_names])


def load_master_data():
    path_ib_prob = os.path.join(BASE_DIR, "NBFO_IB", "saved_models", "gcon_test_scores_best_xgboost_calibrated_sigmoid.parquet")
    df_ib = pd.read_parquet(path_ib_prob)
    df_ib_prob = df_ib.groupby('CUSTOMER_NUMBER')['SUBSCRIPTION_PROPENSITY'].max().reset_index()
    df_ib_prob.rename(columns={'SUBSCRIPTION_PROPENSITY': 'PROBABILITY'}, inplace=True)

    path_ib_features = os.path.join(BASE_DIR, "NBFO_IB", "processed_data", "gcon_model_input.parquet")
    df_ib_features = pd.read_parquet(path_ib_features)
    df_ib_features = df_ib_features.sort_values('MONTH').groupby('CUSTOMER_NUMBER').last().reset_index()
    df_ib_features = attach_ib_register_date(df_ib_features, BASE_DIR)
    df_ib_segments = add_ib_segments(df_ib_features)
    asset_cols = ['AVG_TD_BALANCE', 'AVG_CA_BALANCE', 'AVG_LOAN_AMOUNT']
    segment_cols = ['CUSTOMER_NUMBER', 'SEGMENT', 'MAPPED_IB_SEGMENT', 'CUSTOMER_TYPE']
    segment_cols += [col for col in asset_cols if col in df_ib_segments.columns]
    df_ib_prob = df_ib_prob.merge(
        df_ib_segments[segment_cols],
        on='CUSTOMER_NUMBER',
        how='left',
    )
    df_ib_prob['SEGMENT'] = df_ib_prob['SEGMENT'].fillna(DEFAULT_SEGMENT)
    df_ib_prob['MAPPED_IB_SEGMENT'] = df_ib_prob['MAPPED_IB_SEGMENT'].fillna(df_ib_prob['SEGMENT'])
    df_ib_prob['CUSTOMER_TYPE'] = 'IB'
    df_ib_prob['SEGMENT_CLUSTER'] = df_ib_prob['SEGMENT']

    cols = ['CUSTOMER_NUMBER', 'CUSTOMER_TYPE', 'SEGMENT', 'SEGMENT_CLUSTER', 'MAPPED_IB_SEGMENT', 'PROBABILITY']
    extra_cols = [col for col in asset_cols if col in df_ib_prob.columns]
    df_master = df_ib_prob[cols + extra_cols].copy()
    df_master['ECONOMIC_SEGMENT'] = df_master.apply(segment_for_economics, axis=1)
    df_master['IS_VIP'] = df_master['ECONOMIC_SEGMENT'].isin(VIP_SEGMENTS)
    df_master['TP'] = df_master['ECONOMIC_SEGMENT'].map(lambda x: FUM_MATRIX.get(x, FUM_MATRIX[DEFAULT_SEGMENT])['TP'])
    df_master['FP'] = df_master['ECONOMIC_SEGMENT'].map(lambda x: FUM_MATRIX.get(x, FUM_MATRIX[DEFAULT_SEGMENT])['FP'])
    df_master['FN'] = df_master['ECONOMIC_SEGMENT'].map(lambda x: FUM_MATRIX.get(x, FUM_MATRIX[DEFAULT_SEGMENT])['FN'])
    df_master['ASSET_SCORE'] = calculate_asset_score(df_master)
    return df_master


def threshold_emu_formula(p, cr, cost, tp_val, fn_val, fp_val):
    uplift = 4 * p * (1 - p) * cr
    fp_probability = 1 - p - uplift
    return uplift * (tp_val - fn_val) + fp_probability * fp_val - cost


def calculate_channel_thresholds(channels, fp_multiplier=1.0):
    thresholds = {}
    ps = np.linspace(0, 1, 5000)
    for segment, economics in FUM_MATRIX.items():
        thresholds[segment] = {}
        for ch_name, ch_data in channels.items():
            if ch_name == 'RM' and segment not in VIP_SEGMENTS:
                thresholds[segment][ch_name] = None
                continue
            fp_val = economics['FP'] * fp_multiplier if segment in VIP_SEGMENTS else economics['FP']
            emus = threshold_emu_formula(ps, ch_data['cr'], ch_data['cost'], economics['TP'], economics['FN'], fp_val)
            valid_ps = ps[emus >= 0]
            thresholds[segment][ch_name] = float(valid_ps[0]) if len(valid_ps) > 0 else None
    return thresholds


def build_eligibility_matrix(df, thresholds):
    eligible = np.full((len(df), len(channel_names)), False)
    for channel_idx, ch_name in enumerate(channel_names):
        segment_thresholds = df['ECONOMIC_SEGMENT'].map(
            lambda segment: thresholds.get(segment, thresholds[DEFAULT_SEGMENT]).get(ch_name)
        )
        has_threshold = segment_thresholds.notna()
        eligible[:, channel_idx] = has_threshold & (df['PROBABILITY'] >= segment_thresholds.astype(float))
    return eligible


print("Loading data for Heatmap generation...")
df_master = load_master_data()


def run_scenario(human_cr_drop, vip_fp_inc):
    channels = {
        'SMS': {'cost': 5_000, 'cr': 0.02},
        'Telesales': {'cost': 50_000, 'cr': 0.05 * (1 - human_cr_drop)},
        'RM': {'cost': 2_000_000, 'cr': 0.15 * (1 - human_cr_drop)},
    }

    p_base = df_master['PROBABILITY']
    tp_array = df_master['TP']
    fn_array = df_master['FN']
    fp_array = np.where(df_master['IS_VIP'], df_master['FP'] * (1 + vip_fp_inc), df_master['FP'])

    emus = []
    for ch_name in channel_names:
        uplift = 4 * p_base * (1 - p_base) * channels[ch_name]['cr']
        fp_probability = 1 - p_base - uplift
        emus.append(uplift * (tp_array - fn_array) + fp_probability * fp_array - channels[ch_name]['cost'])

    tie_breaker = 1e-6 * df_master['ASSET_SCORE']
    emu_matrix = np.vstack([emu + tie_breaker for emu in emus]).T
    scenario_eligibility_matrix = build_eligibility_matrix(
        df_master,
        calculate_channel_thresholds(channels, fp_multiplier=1 + vip_fp_inc),
    )
    emu_matrix = np.where(scenario_eligibility_matrix, emu_matrix, -999_999_999)

    allocations = solve_channel_milp(
        emu_matrix,
        channel_costs,
        BUDGET_LIMIT,
        np.array([0, 1, 1]),
        HUMAN_CAP,
    )['allocations']

    profit = np.sum(allocations * emu_matrix)
    rm_count = np.sum(allocations[:, 2])
    return profit, rm_count


human_cr_drops = [0.05, 0.10, 0.15, 0.20]
fp_incs = [0.10, 0.20, 0.30, 0.40]
results_profit = np.zeros((4, 4))
results_rm = np.zeros((4, 4))

print("Running 4x4 Heatmap Scenarios...")
for i, fp in enumerate(fp_incs):
    for j, cr in enumerate(human_cr_drops):
        prof, rm = run_scenario(cr, fp)
        results_profit[i, j] = prof
        results_rm[i, j] = rm

df_profit = pd.DataFrame(
    results_profit,
    index=["+10% FP", "+20% FP", "+30% FP", "+40% FP"],
    columns=[f"-{int(cr * 100)}% CR K2/K3" for cr in human_cr_drops],
)
df_rm = pd.DataFrame(
    results_rm,
    index=["+10% FP", "+20% FP", "+30% FP", "+40% FP"],
    columns=[f"-{int(cr * 100)}% CR K2/K3" for cr in human_cr_drops],
)

out_path = os.path.join(BASE_DIR, "heatmap_results.txt")
with open(out_path, "w", encoding="utf-8") as f:
    f.write("PROFIT MATRIX (VND):\n")
    f.write(frame_to_markdown(df_profit))
    f.write("\n\nRM SLOTS MATRIX:\n")
    f.write(frame_to_markdown(df_rm))

print("Done! Saved to heatmap_results.txt")
