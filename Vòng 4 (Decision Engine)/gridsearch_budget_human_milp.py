import os
import sys
import warnings

import numpy as np
import pandas as pd

from decision_config import (
    DEFAULT_SEGMENT,
    FUM_MATRIX,
    NONIB_CLUSTER_NAMES,
    VIP_SEGMENTS,
    add_ib_segments,
    attach_ib_register_date,
    calculate_asset_score,
    segment_for_economics,
    solve_channel_milp,
    solve_grouped_channel_milp,
)


warnings.filterwarnings('ignore')

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OUT_DIR = BASE_DIR
TOTAL_BUDGET = 1_000_000_000

CHANNELS = {
    'SMS': {'cost': 5_000, 'cr': 0.02},
    'Telesales': {'cost': 50_000, 'cr': 0.05},
    'RM': {'cost': 2_000_000, 'cr': 0.15},
}
CHANNEL_NAMES = np.array(['SMS', 'Telesales', 'RM'])
CHANNEL_COSTS = np.array([CHANNELS[ch]['cost'] for ch in CHANNEL_NAMES])
HUMAN_MASK = np.array([0, 1, 1])

TP_RETENTION = 50_000_000
FP_CONTACT = -50_000
RUNOFF_WEIGHT = 0.30
VIP_CLUSTERS = {1, 2, 3, 4, 6, 7}
FN_BY_CLUSTER = {cluster: (-30_000_000 if cluster in VIP_CLUSTERS else 0) for cluster in NONIB_CLUSTER_NAMES}


def numeric_col(df, name, default=0):
    if name in df.columns:
        return pd.to_numeric(df[name], errors='coerce').fillna(default).astype(float)
    return pd.Series(default, index=df.index, dtype='float64')


def load_ib_master():
    prob_path = os.path.join(BASE_DIR, 'NBFO_IB', 'saved_models', 'gcon_test_scores_best_xgboost_calibrated_sigmoid.parquet')
    df_prob = pd.read_parquet(prob_path)
    df_prob = df_prob.groupby('CUSTOMER_NUMBER')['SUBSCRIPTION_PROPENSITY'].max().reset_index()
    df_prob.rename(columns={'SUBSCRIPTION_PROPENSITY': 'PROBABILITY'}, inplace=True)

    feature_path = os.path.join(BASE_DIR, 'NBFO_IB', 'processed_data', 'gcon_model_input.parquet')
    df_features = pd.read_parquet(feature_path)
    df_features = df_features.sort_values('MONTH').groupby('CUSTOMER_NUMBER').last().reset_index()
    df_features = attach_ib_register_date(df_features, BASE_DIR)
    df_segments = add_ib_segments(df_features)

    asset_cols = ['AVG_TD_BALANCE', 'AVG_CA_BALANCE', 'AVG_LOAN_AMOUNT']
    segment_cols = ['CUSTOMER_NUMBER', 'SEGMENT', 'MAPPED_IB_SEGMENT', 'CUSTOMER_TYPE']
    segment_cols += [col for col in asset_cols if col in df_segments.columns]
    df = df_prob.merge(df_segments[segment_cols], on='CUSTOMER_NUMBER', how='left')
    df['SEGMENT'] = df['SEGMENT'].fillna(DEFAULT_SEGMENT)
    df['MAPPED_IB_SEGMENT'] = df['MAPPED_IB_SEGMENT'].fillna(df['SEGMENT'])
    df['CUSTOMER_TYPE'] = 'IB'
    df['SEGMENT_CLUSTER'] = df['SEGMENT']
    df['ECONOMIC_SEGMENT'] = df.apply(segment_for_economics, axis=1)
    df['IS_VIP'] = df['ECONOMIC_SEGMENT'].isin(VIP_SEGMENTS)
    df['TP'] = df['ECONOMIC_SEGMENT'].map(lambda x: FUM_MATRIX.get(x, FUM_MATRIX[DEFAULT_SEGMENT])['TP'])
    df['FP'] = df['ECONOMIC_SEGMENT'].map(lambda x: FUM_MATRIX.get(x, FUM_MATRIX[DEFAULT_SEGMENT])['FP'])
    df['FN'] = df['ECONOMIC_SEGMENT'].map(lambda x: FUM_MATRIX.get(x, FUM_MATRIX[DEFAULT_SEGMENT])['FN'])
    df['ASSET_SCORE'] = calculate_asset_score(df)
    return df


def threshold_emu_formula(p, cr, cost, tp_val, fn_val, fp_val):
    uplift = 4 * p * (1 - p) * cr
    fp_probability = 1 - p - uplift
    return uplift * (tp_val - fn_val) + fp_probability * fp_val - cost


def calculate_ib_thresholds():
    thresholds = {}
    ps = np.linspace(0, 1, 5000)
    for segment, economics in FUM_MATRIX.items():
        thresholds[segment] = {}
        for ch_name, ch_data in CHANNELS.items():
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
            thresholds[segment][ch_name] = float(valid_ps[0]) if len(valid_ps) else None
    return thresholds


def build_ib_eligibility(df, thresholds):
    eligible = np.full((len(df), len(CHANNEL_NAMES)), False)
    for channel_idx, ch_name in enumerate(CHANNEL_NAMES):
        segment_thresholds = df['ECONOMIC_SEGMENT'].map(
            lambda segment: thresholds.get(segment, thresholds[DEFAULT_SEGMENT]).get(ch_name)
        )
        has_threshold = segment_thresholds.notna()
        eligible[:, channel_idx] = has_threshold & (df['PROBABILITY'] >= segment_thresholds.astype(float))
    return eligible


def calculate_ib_emu(df):
    p_base = df['PROBABILITY']
    emus = []
    for ch_name in CHANNEL_NAMES:
        ch = CHANNELS[ch_name]
        uplift = 4 * p_base * (1 - p_base) * ch['cr']
        fp_probability = 1 - p_base - uplift
        emu = uplift * (df['TP'] - df['FN']) + fp_probability * df['FP'] - ch['cost']
        emus.append(emu + 1e-6 * df['ASSET_SCORE'])
    emu_matrix = np.vstack(emus).T
    return np.where(build_ib_eligibility(df, calculate_ib_thresholds()), emu_matrix, -999_999_999)


def load_nonib_personas():
    path = os.path.join(BASE_DIR, 'Cluster_nonIB', 'output', 'nonib_final_personas.parquet')
    df = pd.read_parquet(path)
    df['CLUSTER_NAME'] = df['CLUSTER'].map(NONIB_CLUSTER_NAMES)
    df['SEGMENT_CLUSTER'] = df['CLUSTER_NAME'].fillna(df.get('PERSONA_NAME', 'UNKNOWN'))
    df['CUSTOMER_TYPE'] = 'Non-IB'
    return df


def aggregate_monthly_value(source_path, value_columns, customer_ids):
    df = pd.read_parquet(source_path)
    df = df[df['CUSTOMER_NUMBER'].isin(customer_ids)].copy()
    if df.empty:
        return pd.DataFrame(columns=['CUSTOMER_NUMBER', 'MONTH', 'VALUE'])

    df['MONTH'] = pd.to_datetime(df['MONTH'], errors='coerce').dt.to_period('M').astype(str)
    value = pd.Series(0.0, index=df.index)
    for col in value_columns:
        value += numeric_col(df, col, 0)
    df['VALUE'] = value
    return df.groupby(['CUSTOMER_NUMBER', 'MONTH'], as_index=False)['VALUE'].sum()


def build_churn_panel(customer_ids):
    months = pd.period_range('2019-01', '2019-12', freq='M').astype(str)
    panel = pd.MultiIndex.from_product(
        [pd.Index(customer_ids, name='CUSTOMER_NUMBER'), months],
        names=['CUSTOMER_NUMBER', 'MONTH'],
    ).to_frame(index=False)

    customer_set = set(customer_ids)
    deposit = aggregate_monthly_value(
        os.path.join(BASE_DIR, 'cleaned_data', 'deposit_clean.parquet'),
        ['AVG_CA_BALANCE', 'AVG_TD_BALANCE'],
        customer_set,
    )
    lending = aggregate_monthly_value(
        os.path.join(BASE_DIR, 'cleaned_data', 'lending_clean.parquet'),
        ['AVG_LOAN_AMOUNT'],
        customer_set,
    )
    card = aggregate_monthly_value(
        os.path.join(BASE_DIR, 'cleaned_data', 'card_clean.parquet'),
        ['COUNT_CREDITCARD', 'COUNT_DEBITCARD'],
        customer_set,
    )

    panel = panel.merge(deposit.rename(columns={'VALUE': 'DEPOSIT_VALUE'}), on=['CUSTOMER_NUMBER', 'MONTH'], how='left')
    panel = panel.merge(lending.rename(columns={'VALUE': 'LOAN_VALUE'}), on=['CUSTOMER_NUMBER', 'MONTH'], how='left')
    panel = panel.merge(card.rename(columns={'VALUE': 'CARD_COUNT'}), on=['CUSTOMER_NUMBER', 'MONTH'], how='left')
    panel[['DEPOSIT_VALUE', 'LOAN_VALUE', 'CARD_COUNT']] = panel[
        ['DEPOSIT_VALUE', 'LOAN_VALUE', 'CARD_COUNT']
    ].fillna(0)
    panel['FINANCIAL_VALUE'] = panel['DEPOSIT_VALUE'] + panel['LOAN_VALUE']
    panel['ACTIVE'] = ((panel['FINANCIAL_VALUE'] > 0) | (panel['CARD_COUNT'] > 0)).astype(int)

    active = panel.pivot(index='CUSTOMER_NUMBER', columns='MONTH', values='ACTIVE').fillna(0)
    value = panel.pivot(index='CUSTOMER_NUMBER', columns='MONTH', values='FINANCIAL_VALUE').fillna(0)
    active_jan_sep = active[[f'2019-{month:02d}' for month in range(1, 10)]].max(axis=1)
    active_q4 = active[['2019-10', '2019-11', '2019-12']].max(axis=1)
    sep_value = value['2019-09']
    dec_value = value['2019-12']

    churn_at_risk = active_jan_sep == 1
    q4_inactive = active_q4 == 0
    hard_churn = churn_at_risk & q4_inactive & ((sep_value <= 0) | (dec_value <= 0.05 * sep_value))
    runoff_risk = churn_at_risk & ~q4_inactive & (sep_value > 0) & (dec_value <= 0.2 * sep_value)
    effective_churn_score = hard_churn.astype(float) + RUNOFF_WEIGHT * runoff_risk.astype(float)

    return pd.DataFrame({
        'CUSTOMER_NUMBER': active.index,
        'CHURN_AT_RISK': churn_at_risk.astype(int).values,
        'HARD_CHURN': hard_churn.astype(int).values,
        'RUNOFF_RISK': runoff_risk.astype(int).values,
        'EFFECTIVE_CHURN_SCORE': effective_churn_score.values,
    })


def compute_cluster_churn(df, churn_panel):
    tagged = df[['CUSTOMER_NUMBER', 'CLUSTER']].merge(churn_panel, on='CUSTOMER_NUMBER', how='left')
    tagged[['CHURN_AT_RISK', 'HARD_CHURN', 'RUNOFF_RISK', 'EFFECTIVE_CHURN_SCORE']] = tagged[
        ['CHURN_AT_RISK', 'HARD_CHURN', 'RUNOFF_RISK', 'EFFECTIVE_CHURN_SCORE']
    ].fillna(0)
    at_risk = tagged[tagged['CHURN_AT_RISK'] == 1]
    cluster_rate = (
        at_risk.groupby('CLUSTER')
        .agg(
            CUSTOMER_COUNT=('CUSTOMER_NUMBER', 'count'),
            EFFECTIVE_CHURN=('EFFECTIVE_CHURN_SCORE', 'sum'),
        )
        .reset_index()
    )
    all_clusters = pd.DataFrame({'CLUSTER': sorted(df['CLUSTER'].dropna().unique())})
    cluster_rate = all_clusters.merge(cluster_rate, on='CLUSTER', how='left')
    fallback_rate = float(at_risk['EFFECTIVE_CHURN_SCORE'].mean()) if len(at_risk) else 0.0
    cluster_rate['CUSTOMER_COUNT'] = cluster_rate['CUSTOMER_COUNT'].fillna(0).astype(int)
    cluster_rate['EFFECTIVE_CHURN'] = cluster_rate['EFFECTIVE_CHURN'].fillna(0.0)
    cluster_rate['P_CHURN'] = (
        cluster_rate['EFFECTIVE_CHURN'] / cluster_rate['CUSTOMER_COUNT'].clip(lower=1)
    ).fillna(fallback_rate).clip(lower=0.001, upper=0.70)
    cluster_rate.loc[cluster_rate['CLUSTER'] == -1, 'P_CHURN'] = 0.001
    return cluster_rate


def retention_emu(p, tp, fn, channel):
    ch = CHANNELS[channel]
    uplift = 4 * p * (1 - p) * ch['cr']
    fp_probability = 1 - p - uplift
    return uplift * (tp - fn) + fp_probability * FP_CONTACT - ch['cost']


def build_nonib_groups():
    df = load_nonib_personas()
    churn_panel = build_churn_panel(df['CUSTOMER_NUMBER'].drop_duplicates())
    cluster_churn = compute_cluster_churn(df, churn_panel)
    df = df.merge(churn_panel, on='CUSTOMER_NUMBER', how='left')
    df = df.merge(cluster_churn[['CLUSTER', 'P_CHURN']], on='CLUSTER', how='left')
    df[['CHURN_AT_RISK', 'HARD_CHURN', 'RUNOFF_RISK']] = df[
        ['CHURN_AT_RISK', 'HARD_CHURN', 'RUNOFF_RISK']
    ].fillna(0).astype(int)
    df['P_CHURN'] = df['P_CHURN'].fillna(cluster_churn['P_CHURN'].mean()).clip(lower=0.001, upper=0.70)
    eligible = df[(df['CLUSTER'] != -1) & (df['CHURN_AT_RISK'] == 1)].copy()

    groups = (
        eligible.groupby('CLUSTER', as_index=False)
        .agg(CUSTOMER_COUNT=('CUSTOMER_NUMBER', 'count'), P_CHURN=('P_CHURN', 'first'))
        .sort_values('CLUSTER')
        .reset_index(drop=True)
    )
    groups['TP'] = TP_RETENTION
    groups['FN'] = groups['CLUSTER'].map(FN_BY_CLUSTER).fillna(0)
    for channel in CHANNEL_NAMES:
        groups[f'EMU_{channel.upper()}'] = retention_emu(groups['P_CHURN'], groups['TP'], groups['FN'], channel)
    groups.loc[~groups['CLUSTER'].isin(VIP_CLUSTERS), 'EMU_RM'] = -999_999_999
    return groups


def solve_ib(emu_matrix, budget, human_cap):
    result = solve_channel_milp(
        emu_matrix,
        CHANNEL_COSTS,
        budget,
        HUMAN_MASK,
        human_cap,
        time_limit=180,
        mip_rel_gap=0.001,
    )
    alloc = result['allocations']
    return {
        'EMU': float(np.sum(alloc * emu_matrix)),
        'COST': float(np.sum(alloc * CHANNEL_COSTS)),
        'SMS': int(alloc[:, 0].sum()),
        'TELESALES': int(alloc[:, 1].sum()),
        'RM': int(alloc[:, 2].sum()),
        'STATUS': result['status'],
    }


def solve_nonib(groups, budget, human_cap):
    emu_matrix = groups[['EMU_SMS', 'EMU_TELESALES', 'EMU_RM']].to_numpy()
    result = solve_grouped_channel_milp(
        emu_matrix,
        groups['CUSTOMER_COUNT'].to_numpy(),
        CHANNEL_COSTS,
        budget,
        HUMAN_MASK,
        human_cap,
        time_limit=60,
        mip_rel_gap=0.001,
    )
    counts = result['counts']
    expected_retained = 0.0
    for channel_pos, channel in enumerate(CHANNEL_NAMES):
        uplift = 4 * groups['P_CHURN'] * (1 - groups['P_CHURN']) * CHANNELS[channel]['cr']
        expected_retained += float(np.sum(counts[:, channel_pos] * uplift))
    return {
        'EMU': float(np.sum(counts * emu_matrix)),
        'COST': float(np.sum(counts * CHANNEL_COSTS)),
        'SMS': int(counts[:, 0].sum()),
        'TELESALES': int(counts[:, 1].sum()),
        'RM': int(counts[:, 2].sum()),
        'EXPECTED_RETAINED': expected_retained,
        'STATUS': result['status'],
    }


def normalize(series):
    min_value = series.min()
    max_value = series.max()
    if max_value == min_value:
        return pd.Series(1.0, index=series.index)
    return (series - min_value) / (max_value - min_value)


def run_scenario(name, ib_emu, nonib_groups, ib_budget, ib_human):
    nonib_budget = TOTAL_BUDGET - ib_budget
    nonib_human = 10_000 - ib_human
    print(
        f"Running {name}: budget IB/NonIB={ib_budget/1e6:.0f}/{nonib_budget/1e6:.0f}M, "
        f"human IB/NonIB={ib_human}/{nonib_human}"
    )
    ib = solve_ib(ib_emu, ib_budget, ib_human)
    nonib = solve_nonib(nonib_groups, nonib_budget, nonib_human)
    return {
        'STAGE': name,
        'IB_BUDGET': ib_budget,
        'NONIB_BUDGET': nonib_budget,
        'IB_HUMAN_CAP': ib_human,
        'NONIB_HUMAN_CAP': nonib_human,
        'IB_EMU': ib['EMU'],
        'NONIB_EMU': nonib['EMU'],
        'TOTAL_RAW_EMU': ib['EMU'] + nonib['EMU'],
        'IB_COST': ib['COST'],
        'NONIB_COST': nonib['COST'],
        'IB_SMS': ib['SMS'],
        'IB_TELESALES': ib['TELESALES'],
        'IB_RM': ib['RM'],
        'NONIB_SMS': nonib['SMS'],
        'NONIB_TELESALES': nonib['TELESALES'],
        'NONIB_RM': nonib['RM'],
        'NONIB_EXPECTED_RETAINED': nonib['EXPECTED_RETAINED'],
        'IB_STATUS': ib['STATUS'],
        'NONIB_STATUS': nonib['STATUS'],
    }


def write_markdown(results, path):
    top_cols = [
        'STAGE', 'IB_BUDGET_M', 'NONIB_BUDGET_M', 'IB_HUMAN_CAP', 'NONIB_HUMAN_CAP',
        'IB_EMU_B', 'NONIB_EMU_B', 'NORMALIZED_SCORE', 'IB_SMS', 'IB_TELESALES', 'IB_RM',
        'NONIB_SMS', 'NONIB_TELESALES', 'NONIB_RM', 'NONIB_EXPECTED_RETAINED',
    ]
    table = results.copy()
    table['IB_BUDGET_M'] = (table['IB_BUDGET'] / 1_000_000).round(0).astype(int)
    table['NONIB_BUDGET_M'] = (table['NONIB_BUDGET'] / 1_000_000).round(0).astype(int)
    table['IB_EMU_B'] = (table['IB_EMU'] / 1_000_000_000).round(3)
    table['NONIB_EMU_B'] = (table['NONIB_EMU'] / 1_000_000_000).round(3)
    table['NORMALIZED_SCORE'] = table['NORMALIZED_SCORE'].round(4)
    table['NONIB_EXPECTED_RETAINED'] = table['NONIB_EXPECTED_RETAINED'].round(2)
    top = table.sort_values('NORMALIZED_SCORE', ascending=False)[top_cols]
    rows = [[str(col) for col in top.columns]]
    rows.extend(top.astype(str).values.tolist())
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    header = '| ' + ' | '.join(rows[0][i].ljust(widths[i]) for i in range(len(widths))) + ' |'
    separator = '| ' + ' | '.join('-' * widths[i] for i in range(len(widths))) + ' |'
    body = ['| ' + ' | '.join(row[i].ljust(widths[i]) for i in range(len(widths))) + ' |' for row in rows[1:]]
    with open(path, 'w', encoding='utf-8') as f:
        f.write('# MILP Budget/Human Coarse-to-Refine Grid\n\n')
        f.write('Score = 0.5 * normalized IB EMU + 0.5 * normalized Non-IB EMU.\n\n')
        f.write('\n'.join([header, separator] + body))
        f.write('\n')


def main():
    out_csv = os.path.join(OUT_DIR, 'milp_budget_human_grid_results.csv')
    out_md = os.path.join(OUT_DIR, 'milp_budget_human_grid_results.md')
    if '--render-only' in sys.argv:
        results = pd.read_csv(out_csv)
        write_markdown(results, out_md)
        print(f'Wrote {out_md}')
        return
    if '--append-boundary' in sys.argv:
        results = pd.read_csv(out_csv)
        exists = (
            (results['IB_BUDGET'] == 400_000_000)
            & (results['IB_HUMAN_CAP'] == 6_000)
        ).any()
        if not exists:
            print('Loading matrices for one boundary scenario...')
            ib_emu = calculate_ib_emu(load_ib_master())
            nonib_groups = build_nonib_groups()
            extra = run_scenario('boundary', ib_emu, nonib_groups, 400_000_000, 6_000)
            results = pd.concat([results, pd.DataFrame([extra])], ignore_index=True)
        results['IB_NORM'] = normalize(results['IB_EMU'])
        results['NONIB_NORM'] = normalize(results['NONIB_EMU'])
        results['NORMALIZED_SCORE'] = 0.5 * results['IB_NORM'] + 0.5 * results['NONIB_NORM']
        results = results.sort_values('NORMALIZED_SCORE', ascending=False).reset_index(drop=True)
        results.to_csv(out_csv, index=False)
        write_markdown(results, out_md)
        print(f'Wrote {out_csv}')
        print(f'Wrote {out_md}')
        return

    print('Loading IB data and EMU matrix...')
    ib_master = load_ib_master()
    ib_emu = calculate_ib_emu(ib_master)

    print('Loading Non-IB churn groups...')
    nonib_groups = build_nonib_groups()

    records = []
    for ib_budget in [400_000_000, 500_000_000, 600_000_000]:
        records.append(run_scenario('coarse', ib_emu, nonib_groups, ib_budget, 5_000))

    coarse = pd.DataFrame(records)
    coarse['IB_NORM'] = normalize(coarse['IB_EMU'])
    coarse['NONIB_NORM'] = normalize(coarse['NONIB_EMU'])
    coarse['NORMALIZED_SCORE'] = 0.5 * coarse['IB_NORM'] + 0.5 * coarse['NONIB_NORM']
    best_budget = int(coarse.sort_values('NORMALIZED_SCORE', ascending=False).iloc[0]['IB_BUDGET'])
    print(f"Best coarse IB budget: {best_budget/1e6:.0f}M")

    refine_budgets = sorted(set([max(300_000_000, best_budget - 50_000_000), best_budget, min(700_000_000, best_budget + 50_000_000)]))
    for ib_budget in refine_budgets:
        for ib_human in [4_000, 5_000, 6_000]:
            if ib_budget == best_budget and ib_human == 5_000:
                continue
            records.append(run_scenario('refine', ib_emu, nonib_groups, ib_budget, ib_human))

    results = pd.DataFrame(records)
    results['IB_NORM'] = normalize(results['IB_EMU'])
    results['NONIB_NORM'] = normalize(results['NONIB_EMU'])
    results['NORMALIZED_SCORE'] = 0.5 * results['IB_NORM'] + 0.5 * results['NONIB_NORM']
    results = results.sort_values('NORMALIZED_SCORE', ascending=False).reset_index(drop=True)

    results.to_csv(out_csv, index=False)
    write_markdown(results, out_md)

    best = results.iloc[0]
    print('Best normalized scenario:')
    print(
        f"IB/NonIB budget={best['IB_BUDGET']/1e6:.0f}/{best['NONIB_BUDGET']/1e6:.0f}M, "
        f"human={int(best['IB_HUMAN_CAP'])}/{int(best['NONIB_HUMAN_CAP'])}, "
        f"score={best['NORMALIZED_SCORE']:.4f}, "
        f"IB_EMU={best['IB_EMU']:,.0f}, NonIB_EMU={best['NONIB_EMU']:,.0f}"
    )
    print(f'Wrote {out_csv}')
    print(f'Wrote {out_md}')


if __name__ == '__main__':
    main()
