import os
import warnings

import numpy as np
import pandas as pd

from decision_config import NONIB_CLUSTER_NAMES, solve_grouped_channel_milp


warnings.filterwarnings('ignore')

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
BUDGET_LIMIT = 550_000_000
HUMAN_CAP = 4_000
TP_RETENTION = 50_000_000
FP_CONTACT = -50_000
RUNOFF_WEIGHT = 0.30

channels = {
    'SMS': {'cost': 5_000, 'cr': 0.02},
    'Telesales': {'cost': 50_000, 'cr': 0.05},
    'RM': {'cost': 2_000_000, 'cr': 0.15},
}
channel_names = np.array(['SMS', 'Telesales', 'RM'])
channel_costs = np.array([channels[ch]['cost'] for ch in channel_names])

VIP_CLUSTERS = {1, 2, 3, 4, 5, 6, 7}
FN_BY_CLUSTER = {cluster: (-30_000_000 if cluster in VIP_CLUSTERS else 0) for cluster in NONIB_CLUSTER_NAMES}


def frame_to_markdown(df):
    rows = [[str(col) for col in df.columns]]
    rows.extend(df.fillna('').astype(str).values.tolist())
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    header = '| ' + ' | '.join(rows[0][i].ljust(widths[i]) for i in range(len(widths))) + ' |'
    separator = '| ' + ' | '.join('-' * widths[i] for i in range(len(widths))) + ' |'
    body = ['| ' + ' | '.join(row[i].ljust(widths[i]) for i in range(len(widths))) + ' |' for row in rows[1:]]
    return '\n'.join([header, separator] + body)


def numeric_col(df, name, default=0):
    if name in df.columns:
        return pd.to_numeric(df[name], errors='coerce').fillna(default).astype(float)
    return pd.Series(default, index=df.index, dtype='float64')


def load_nonib_personas():
    path_nonib = os.path.join(BASE_DIR, "Cluster_nonIB", "output", "nonib_final_personas.parquet")
    df = pd.read_parquet(path_nonib)
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

    deposit = aggregate_monthly_value(
        os.path.join(BASE_DIR, 'cleaned_data', 'deposit_clean.parquet'),
        ['AVG_CA_BALANCE', 'AVG_TD_BALANCE'],
        set(customer_ids),
    )
    lending = aggregate_monthly_value(
        os.path.join(BASE_DIR, 'cleaned_data', 'lending_clean.parquet'),
        ['AVG_LOAN_AMOUNT'],
        set(customer_ids),
    )
    card = aggregate_monthly_value(
        os.path.join(BASE_DIR, 'cleaned_data', 'card_clean.parquet'),
        ['COUNT_CREDITCARD', 'COUNT_DEBITCARD'],
        set(customer_ids),
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
        'SEP_FINANCIAL_VALUE': sep_value.values,
        'DEC_FINANCIAL_VALUE': dec_value.values,
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
            AT_RISK=('CUSTOMER_NUMBER', 'count'),
            HARD_CHURNERS=('HARD_CHURN', 'sum'),
            RUNOFF_CUSTOMERS=('RUNOFF_RISK', 'sum'),
            EFFECTIVE_CHURN=('EFFECTIVE_CHURN_SCORE', 'sum'),
        )
        .reset_index()
    )
    cluster_rate['P_CHURN'] = cluster_rate['EFFECTIVE_CHURN'] / cluster_rate['AT_RISK'].clip(lower=1)

    all_clusters = pd.DataFrame({'CLUSTER': sorted(df['CLUSTER'].dropna().unique())})
    cluster_rate = all_clusters.merge(cluster_rate, on='CLUSTER', how='left')
    fallback_rate = float(at_risk['EFFECTIVE_CHURN_SCORE'].mean()) if len(at_risk) else 0.0
    cluster_rate['AT_RISK'] = cluster_rate['AT_RISK'].fillna(0).astype(int)
    cluster_rate['HARD_CHURNERS'] = cluster_rate['HARD_CHURNERS'].fillna(0).astype(int)
    cluster_rate['RUNOFF_CUSTOMERS'] = cluster_rate['RUNOFF_CUSTOMERS'].fillna(0).astype(int)
    cluster_rate['EFFECTIVE_CHURN'] = cluster_rate['EFFECTIVE_CHURN'].fillna(0.0)
    cluster_rate['P_CHURN'] = cluster_rate['P_CHURN'].fillna(fallback_rate).clip(lower=0.001, upper=0.70)
    cluster_rate.loc[cluster_rate['CLUSTER'] == -1, 'P_CHURN'] = 0.001
    cluster_rate['CLUSTER_NAME'] = cluster_rate['CLUSTER'].map(NONIB_CLUSTER_NAMES)
    return cluster_rate


def calculate_clv_5yr(df):
    annual = (
        numeric_col(df, 'TD_BALANCE_MEAN', 0) * 0.015
        + numeric_col(df, 'LOAN_AMOUNT_MEAN', 0) * 0.020
        + numeric_col(df, 'HAS_CREDIT_CARD', 0) * 100_000
    )
    return annual * 5


def calculate_asset_score(df):
    value = (
        numeric_col(df, 'TOTAL_FINANCIAL_VALUE', 0)
        + numeric_col(df, 'NET_WORTH_PROXY', 0).clip(lower=0)
        + numeric_col(df, 'TOTAL_DEPOSIT', 0)
        + numeric_col(df, 'TD_BALANCE_LAST', 0)
        + numeric_col(df, 'CA_BALANCE_LAST', 0)
        + numeric_col(df, 'LOAN_AMOUNT_LAST', 0)
    )
    if (value > 0).any():
        return value.rank(method='average', pct=True).fillna(0)
    return (numeric_col(df, 'CUSTOMER_NUMBER', 0) % 10000) / 10000.0


def emu(p, tp, fn, channel, fp=FP_CONTACT, cr_multiplier=1.0):
    cr = channels[channel]['cr'] * cr_multiplier
    cost = channels[channel]['cost']
    uplift = 4 * p * (1 - p) * cr
    fp_probability = 1 - p - uplift
    return uplift * (tp - fn) + fp_probability * fp - cost


def threshold_for_channel(fn, channel):
    ps = np.linspace(0.001, 0.999, 9990)
    emus = emu(ps, TP_RETENTION, fn, channel)
    valid = ps[emus >= 0]
    return float(valid[0]) if len(valid) else None


def solve_retention_allocation(df, threshold_by_segment_channel):
    result = df.copy()
    result['RECOMMENDED_CHANNEL'] = 'None'
    result['CAMPAIGN_COST'] = 0
    result['CAMPAIGN_EMU'] = 0.0

    group_table = (
        result.groupby(['CLUSTER', 'SEGMENT_CLUSTER'], as_index=False)
        .agg(
            CUSTOMER_COUNT=('CUSTOMER_NUMBER', 'count'),
            EMU_SMS=('EMU_SMS', 'first'),
            EMU_TELESALES=('EMU_TELESALES', 'first'),
            EMU_RM=('EMU_RM', 'first'),
        )
        .sort_values('CLUSTER')
        .reset_index(drop=True)
    )
    eligible_counts = np.zeros((len(group_table), len(channel_names)), dtype=int)
    for group_pos, row in group_table.iterrows():
        cluster_pool = result[result['CLUSTER'] == row['CLUSTER']]
        segment = row['SEGMENT_CLUSTER']
        for channel_pos, channel_name in enumerate(channel_names):
            cutoff = threshold_by_segment_channel.get((segment, channel_name))
            if cutoff is None or pd.isna(cutoff):
                eligible_counts[group_pos, channel_pos] = 0
                continue
            emu_col = f'EMU_{channel_name.upper()}' if channel_name != 'Telesales' else 'EMU_TELESALES'
            eligible_counts[group_pos, channel_pos] = int(
                (
                    (cluster_pool['LOSS_PERCENTILE'] >= cutoff)
                    & (cluster_pool[emu_col] > 0)
                ).sum()
            )

    emu_matrix = group_table[['EMU_SMS', 'EMU_TELESALES', 'EMU_RM']].to_numpy()
    milp_result = solve_grouped_channel_milp(
        emu_matrix,
        group_table['CUSTOMER_COUNT'].to_numpy(),
        channel_costs,
        BUDGET_LIMIT,
        np.array([0, 1, 1]),
        HUMAN_CAP,
        max_group_channel_counts=eligible_counts,
        nested_group_channel_counts=eligible_counts,
    )
    print(f"Non-IB MILP allocation status: {milp_result['status']} - {milp_result['message']}")

    allocation_counts = milp_result['counts']
    for group_pos, row in group_table.iterrows():
        cluster_pool = result[result['CLUSTER'] == row['CLUSTER']]
        used_idx = set()
        for channel_pos, channel_name in sorted(enumerate(channel_names), reverse=True):
            take = int(allocation_counts[group_pos, channel_pos])
            if take <= 0:
                continue
            cutoff = threshold_by_segment_channel.get((row['SEGMENT_CLUSTER'], channel_name), 1.0)
            emu_col = f'EMU_{channel_name.upper()}' if channel_name != 'Telesales' else 'EMU_TELESALES'
            channel_pool = cluster_pool[
                (cluster_pool['LOSS_PERCENTILE'] >= cutoff)
                & (cluster_pool[emu_col] > 0)
                & (~cluster_pool.index.isin(used_idx))
            ].sort_values(
                ['LOSS_PERCENTILE', 'ASSET_SCORE', 'CUSTOMER_NUMBER'],
                ascending=[False, False, True],
            )
            selected_idx = channel_pool.head(take).index
            used_idx.update(selected_idx)
            result.loc[selected_idx, 'RECOMMENDED_CHANNEL'] = channel_name
            result.loc[selected_idx, 'CAMPAIGN_COST'] = channel_costs[channel_pos]
            result.loc[selected_idx, 'CAMPAIGN_EMU'] = group_table.loc[
                group_pos,
                f'EMU_{channel_name.upper()}' if channel_name != 'Telesales' else 'EMU_TELESALES',
            ]

    return result


def load_nonib_percentile_thresholds():
    path = os.path.join(BASE_DIR, 'optimized_thresholds_nonib.csv')
    if not os.path.exists(path):
        return {}
    thresholds = pd.read_csv(path)
    thresholds = thresholds[
        (thresholds['Threshold'].notna())
        & thresholds['Optimization Status'].isin(['CAC feasible', 'Max EMU, CAC cap unmet'])
    ].copy()
    return {
        (row['Segment'], row['Channel']): float(row['Threshold'])
        for _, row in thresholds.iterrows()
    }


print("Loading Non-IB retention data...")
df_master = load_nonib_personas()
churn_panel = build_churn_panel(df_master['CUSTOMER_NUMBER'].drop_duplicates())
cluster_churn = compute_cluster_churn(df_master, churn_panel)

df_master = df_master.merge(churn_panel, on='CUSTOMER_NUMBER', how='left')
df_master = df_master.merge(
    cluster_churn[['CLUSTER', 'P_CHURN', 'AT_RISK', 'HARD_CHURNERS', 'RUNOFF_CUSTOMERS', 'EFFECTIVE_CHURN']],
    on='CLUSTER',
    how='left',
)
df_master[['CHURN_AT_RISK', 'HARD_CHURN', 'RUNOFF_RISK']] = df_master[
    ['CHURN_AT_RISK', 'HARD_CHURN', 'RUNOFF_RISK']
].fillna(0).astype(int)
df_master['EFFECTIVE_CHURN_SCORE'] = df_master['EFFECTIVE_CHURN_SCORE'].fillna(0.0)
df_master['P_CHURN'] = df_master['P_CHURN'].fillna(cluster_churn['P_CHURN'].mean()).clip(lower=0.001, upper=0.70)
df_master['CLV_5YR'] = calculate_clv_5yr(df_master)
df_master['TP'] = TP_RETENTION
df_master['FP'] = FP_CONTACT
df_master['FN'] = df_master['CLUSTER'].map(FN_BY_CLUSTER).fillna(0)
df_master['ASSET_SCORE'] = calculate_asset_score(df_master)
df_master['EXPECTED_LOSS_SCORE'] = df_master['P_CHURN'] * df_master['CLV_5YR']
df_master['LOSS_PERCENTILE'] = (
    df_master.groupby('SEGMENT_CLUSTER')['EXPECTED_LOSS_SCORE']
    .rank(method='average', pct=True)
    .fillna(0.0)
)
df_master['RECOMMENDED_PRODUCT'] = 'Retention'
df_master['EMU_SMS'] = emu(df_master['P_CHURN'], df_master['TP'], df_master['FN'], 'SMS')
df_master['EMU_TELESALES'] = emu(df_master['P_CHURN'], df_master['TP'], df_master['FN'], 'Telesales')
df_master['EMU_RM'] = emu(df_master['P_CHURN'], df_master['TP'], df_master['FN'], 'RM')
df_master.loc[~df_master['CLUSTER'].isin(VIP_CLUSTERS), 'EMU_RM'] = -999_999_999

# P0 and customers outside the historical at-risk base are not paid retention targets.
threshold_by_segment = load_nonib_percentile_thresholds()
eligible_mask = (
    (df_master['CLUSTER'] != -1)
    & (df_master['CHURN_AT_RISK'] == 1)
)
if threshold_by_segment:
    print(f"Applied optimized Non-IB percentile thresholds for {len(threshold_by_segment)} cluster-channel pairs")
allocated = solve_retention_allocation(df_master[eligible_mask], threshold_by_segment)
df_master['RECOMMENDED_CHANNEL'] = 'None'
df_master['CAMPAIGN_COST'] = 0
df_master['CAMPAIGN_EMU'] = 0.0
df_master.loc[allocated.index, ['RECOMMENDED_CHANNEL', 'CAMPAIGN_COST', 'CAMPAIGN_EMU']] = allocated[
    ['RECOMMENDED_CHANNEL', 'CAMPAIGN_COST', 'CAMPAIGN_EMU']
]

counts = df_master['RECOMMENDED_CHANNEL'].value_counts()
total_cost = float(df_master['CAMPAIGN_COST'].sum())
total_emu = float(df_master['CAMPAIGN_EMU'].sum())
sms_sub = df_master[df_master['RECOMMENDED_CHANNEL'] == 'SMS']
assigned_sub = df_master[df_master['RECOMMENDED_CHANNEL'] != 'None'].copy()
expected_retained = 0.0
for ch_name, ch_data in channels.items():
    sub = assigned_sub[assigned_sub['RECOMMENDED_CHANNEL'] == ch_name]
    expected_retained += float((4 * sub['P_CHURN'] * (1 - sub['P_CHURN']) * ch_data['cr']).sum())
clv_at_risk_targeted = float((assigned_sub['CLV_5YR'] * assigned_sub['P_CHURN']).sum())

print("Non-IB Retention Results:")
print(f"Customers: {len(df_master):,}")
print(f"At-risk base: {int(df_master['CHURN_AT_RISK'].sum()):,}")
print(f"Hard churn proxy: {int(df_master['HARD_CHURN'].sum()):,}")
print(f"Runoff risk proxy: {int(df_master['RUNOFF_RISK'].sum()):,}")
print(f"Effective churn score: {df_master['EFFECTIVE_CHURN_SCORE'].sum():,.1f}")
print(f"Cost: {total_cost:,.0f}")
print(f"Campaign EMU: {total_emu:,.0f}")
print(f"Expected retained customers: {expected_retained:,.2f}")
print(f"CLV at risk targeted: {clv_at_risk_targeted:,.0f}")
print(
    f"Allocations: SMS={int(counts.get('SMS', 0))}, "
    f"Tele={int(counts.get('Telesales', 0))}, "
    f"RM={int(counts.get('RM', 0))}, "
    f"None={int(counts.get('None', 0))}"
)

threshold_rows = []
for _, row in cluster_churn.sort_values('P_CHURN', ascending=False).iterrows():
    cluster = int(row['CLUSTER'])
    fn = FN_BY_CLUSTER.get(cluster, 0)
    threshold_rows.append({
        'Cluster': cluster,
        'Cluster Name': row['CLUSTER_NAME'],
        'At Risk': int(row['AT_RISK']),
        'Hard Churners': int(row['HARD_CHURNERS']),
        'Runoff Risk': int(row['RUNOFF_CUSTOMERS']),
        'Effective Churn': float(row['EFFECTIVE_CHURN']),
        'P Churn': row['P_CHURN'],
        'Threshold SMS': threshold_for_channel(fn, 'SMS'),
        'Threshold Telesales': threshold_for_channel(fn, 'Telesales'),
        'Threshold RM': threshold_for_channel(fn, 'RM') if cluster in VIP_CLUSTERS else None,
        'FN': fn,
        'TP': TP_RETENTION,
    })
threshold_df = pd.DataFrame(threshold_rows)
threshold_path = os.path.join(BASE_DIR, "thresholds_nonib.csv")
threshold_df.to_csv(threshold_path, index=False)

threshold_display = threshold_df.copy()
for col in ['P Churn', 'Threshold SMS', 'Threshold Telesales', 'Threshold RM']:
    threshold_display[col] = threshold_display[col].map(
        lambda value: f"{value:.4f}" if pd.notna(value) else 'N/A'
    )
with open(os.path.join(BASE_DIR, "thresholds_nonib.md"), "w", encoding="utf-8") as f:
    f.write(frame_to_markdown(threshold_display))

cluster_summary = (
    df_master.groupby(['CLUSTER', 'SEGMENT_CLUSTER'])
    .agg(
        N=('CUSTOMER_NUMBER', 'count'),
        AT_RISK=('CHURN_AT_RISK', 'sum'),
        HARD_CHURNERS=('HARD_CHURN', 'sum'),
        RUNOFF_CUSTOMERS=('RUNOFF_RISK', 'sum'),
        EFFECTIVE_CHURN=('EFFECTIVE_CHURN_SCORE', 'sum'),
        P_CHURN=('P_CHURN', 'first'),
        SMS=('RECOMMENDED_CHANNEL', lambda x: (x == 'SMS').sum()),
        TELESALES=('RECOMMENDED_CHANNEL', lambda x: (x == 'Telesales').sum()),
        RM=('RECOMMENDED_CHANNEL', lambda x: (x == 'RM').sum()),
        NONE=('RECOMMENDED_CHANNEL', lambda x: (x == 'None').sum()),
        COST=('CAMPAIGN_COST', 'sum'),
        EMU=('CAMPAIGN_EMU', 'sum'),
        MEDIAN_CLV_5YR=('CLV_5YR', 'median'),
    )
    .reset_index()
    .sort_values('P_CHURN', ascending=False)
)
cluster_summary.to_csv(os.path.join(BASE_DIR, "nonib_retention_cluster_summary.csv"), index=False)

print("\nCluster churn/allocation summary:")
display_cols = [
    'CLUSTER', 'SEGMENT_CLUSTER', 'N', 'AT_RISK', 'HARD_CHURNERS', 'RUNOFF_CUSTOMERS',
    'EFFECTIVE_CHURN', 'P_CHURN',
    'SMS', 'TELESALES', 'RM', 'NONE', 'COST', 'EMU',
]
print(frame_to_markdown(cluster_summary[display_cols].head(12)))

output_cols = [
    'CUSTOMER_NUMBER',
    'CUSTOMER_TYPE',
    'SEGMENT_CLUSTER',
    'RECOMMENDED_PRODUCT',
    'P_CHURN',
    'CHURN_AT_RISK',
    'HARD_CHURN',
    'RUNOFF_RISK',
    'EFFECTIVE_CHURN_SCORE',
    'CLV_5YR',
    'EXPECTED_LOSS_SCORE',
    'LOSS_PERCENTILE',
    'TP',
    'FP',
    'FN',
    'EMU_SMS',
    'EMU_TELESALES',
    'EMU_RM',
    'RECOMMENDED_CHANNEL',
    'CAMPAIGN_COST',
    'CAMPAIGN_EMU',
]
out_alloc = os.path.join(BASE_DIR, "final_allocations_nonib.csv")
df_master[output_cols].to_csv(out_alloc, index=False)

print("\nSample Non-IB Retention Allocation Output:")
print(frame_to_markdown(df_master[output_cols].head(8)))

print("\nStress-Test (FP +20%, Telesales/RM CR -15%):")
baseline_emu = total_emu
stress_emu = 0.0
for ch_name in channels:
    sub = assigned_sub[assigned_sub['RECOMMENDED_CHANNEL'] == ch_name]
    multiplier = 0.85 if ch_name in {'Telesales', 'RM'} else 1.0
    stress_emu += float(emu(sub['P_CHURN'], sub['TP'], sub['FN'], ch_name, fp=FP_CONTACT * 1.2, cr_multiplier=multiplier).sum())
delta = (stress_emu - baseline_emu) / baseline_emu if baseline_emu else 0
print(f"Baseline EMU: {baseline_emu:,.0f}")
print(f"Stress EMU: {stress_emu:,.0f}")
print(f"Delta: {delta:+.1%}")
