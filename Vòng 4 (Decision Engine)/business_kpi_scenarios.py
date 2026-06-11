import os

import numpy as np
import pandas as pd

from decision_config import solve_grouped_channel_milp


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

CHANNELS = {
    'SMS': {'cost': 5_000, 'cr': 0.02},
    'Telesales': {'cost': 50_000, 'cr': 0.05},
    'RM': {'cost': 2_000_000, 'cr': 0.15},
}
CHANNEL_NAMES = np.array(['SMS', 'Telesales', 'RM'])
CHANNEL_COSTS = np.array([CHANNELS[ch]['cost'] for ch in CHANNEL_NAMES])
HUMAN_MASK = np.array([0, 1, 1])

IB_BUDGET = 450_000_000
IB_HUMAN_CAP = 6_000
NONIB_BUDGET = 550_000_000
NONIB_HUMAN_CAP = 4_000
NONIB_MIN_RM = 101
TP_RETENTION = 50_000_000
FP_CONTACT = -50_000
VIP_CLUSTERS = {1, 2, 3, 4, 5, 6, 7}


def fmt_money(value):
    return round(float(value), 2)


def channel_uplift(p, channel, cr_multiplier=1.0):
    return 4 * p * (1 - p) * CHANNELS[channel]['cr'] * cr_multiplier


def nonib_emu(p, fn, channel, fp=FP_CONTACT, cr_multiplier=1.0):
    uplift = channel_uplift(p, channel, cr_multiplier)
    fp_probability = 1 - p - uplift
    return uplift * (TP_RETENTION - fn) + fp_probability * fp - CHANNELS[channel]['cost']


def business_kpis():
    ib = pd.read_csv(os.path.join(BASE_DIR, 'final_allocations.csv'))
    nonib = pd.read_csv(os.path.join(BASE_DIR, 'final_allocations_nonib.csv'))
    ib['RECOMMENDED_CHANNEL'] = ib['RECOMMENDED_CHANNEL'].fillna('None')
    nonib['RECOMMENDED_CHANNEL'] = nonib['RECOMMENDED_CHANNEL'].fillna('None')

    ib_paid = ib[ib['RECOMMENDED_CHANNEL'] != 'None'].copy()
    ib_paid['CAMPAIGN_COST'] = ib_paid['RECOMMENDED_CHANNEL'].map(lambda ch: CHANNELS[ch]['cost'])
    ib_paid['EXPECTED_CONVERSIONS'] = ib_paid.apply(
        lambda row: channel_uplift(row['PROBABILITY'], row['RECOMMENDED_CHANNEL']),
        axis=1,
    )
    ib_cost = float(ib_paid['CAMPAIGN_COST'].sum())
    if 'CAMPAIGN_EMU' in ib_paid.columns:
        ib_emu = float(ib_paid['CAMPAIGN_EMU'].sum())
    else:
        ib_emu = 5_109_748_849.0
    ib_conversions = float(ib_paid['EXPECTED_CONVERSIONS'].sum())

    nonib_paid = nonib[nonib['RECOMMENDED_CHANNEL'] != 'None'].copy()
    nonib_paid['EXPECTED_CONVERSIONS'] = nonib_paid.apply(
        lambda row: channel_uplift(row['P_CHURN'], row['RECOMMENDED_CHANNEL']),
        axis=1,
    )
    nonib_cost = float(nonib_paid['CAMPAIGN_COST'].sum())
    nonib_emu = float(nonib_paid['CAMPAIGN_EMU'].sum())
    nonib_conversions = float(nonib_paid['EXPECTED_CONVERSIONS'].sum())

    rows = [
        {
            'Branch': 'IB',
            'COGS': ib_cost,
            'EMU': ib_emu,
            'Expected Conversions': ib_conversions,
        },
        {
            'Branch': 'Non-IB',
            'COGS': nonib_cost,
            'EMU': nonib_emu,
            'Expected Conversions': nonib_conversions,
        },
    ]
    total = {
        'Branch': 'Total',
        'COGS': ib_cost + nonib_cost,
        'EMU': ib_emu + nonib_emu,
        'Expected Conversions': ib_conversions + nonib_conversions,
    }
    rows.append(total)
    kpi = pd.DataFrame(rows)
    kpi['CAC'] = kpi['COGS'] / kpi['Expected Conversions']
    kpi['Incremental ROI'] = (kpi['EMU'] - kpi['COGS']) / kpi['COGS']
    kpi['EMU/COGS'] = kpi['EMU'] / kpi['COGS']
    kpi.to_csv(os.path.join(BASE_DIR, 'business_kpis.csv'), index=False)
    return kpi


def pnl_projection(kpi):
    total_cogs = float(kpi.loc[kpi['Branch'] == 'Total', 'COGS'].iloc[0])
    total_emu = float(kpi.loc[kpi['Branch'] == 'Total', 'EMU'].iloc[0])

    cogs_profile = np.array([0.12, 0.15, 0.18, 0.15, 0.12, 0.08, 0.06, 0.04, 0.03, 0.03, 0.02, 0.02])
    emu_profile = np.array([0.04, 0.06, 0.08, 0.09, 0.10, 0.10, 0.10, 0.09, 0.09, 0.09, 0.08, 0.08])
    months = pd.period_range('2020-01', periods=12, freq='M').astype(str)
    monthly = pd.DataFrame({
        'Period': months,
        'COGS': total_cogs * cogs_profile,
        'Incremental EMU': total_emu * emu_profile,
    })
    monthly['Net Incremental Profit'] = monthly['Incremental EMU'] - monthly['COGS']
    monthly['Cumulative Profit'] = monthly['Net Incremental Profit'].cumsum()
    monthly['View'] = 'Monthly'

    quarterly = monthly.copy()
    quarterly['Period'] = pd.PeriodIndex(quarterly['Period'], freq='M').asfreq('Q').astype(str)
    quarterly = quarterly.groupby('Period', as_index=False)[
        ['COGS', 'Incremental EMU', 'Net Incremental Profit']
    ].sum()
    quarterly['Cumulative Profit'] = quarterly['Net Incremental Profit'].cumsum()
    quarterly['View'] = 'Quarterly'

    out = pd.concat([monthly, quarterly], ignore_index=True)
    out.to_csv(os.path.join(BASE_DIR, 'pnl_projection.csv'), index=False)
    return out


def solve_nonib_scenario(name, budget, cr_multipliers=None, fp_multiplier=1.0):
    if cr_multipliers is None:
        cr_multipliers = {}
    summary = pd.read_csv(os.path.join(BASE_DIR, 'nonib_retention_cluster_summary.csv'))
    summary = summary[summary['AT_RISK'] > 0].copy().sort_values('CLUSTER').reset_index(drop=True)
    summary['FN'] = summary['CLUSTER'].map(lambda c: -30_000_000 if c in VIP_CLUSTERS else 0)

    emu_cols = []
    for channel in CHANNEL_NAMES:
        values = nonib_emu(
            summary['P_CHURN'],
            summary['FN'],
            channel,
            fp=FP_CONTACT * fp_multiplier,
            cr_multiplier=cr_multipliers.get(channel, 1.0),
        )
        if channel == 'RM':
            values = values.where(summary['CLUSTER'].isin(VIP_CLUSTERS), -999_999_999)
        emu_cols.append(values)
    emu_matrix = np.vstack(emu_cols).T

    min_rm = NONIB_MIN_RM if budget >= CHANNELS['RM']['cost'] * NONIB_MIN_RM else 0
    result = solve_grouped_channel_milp(
        emu_matrix,
        summary['AT_RISK'].to_numpy(),
        CHANNEL_COSTS,
        budget,
        HUMAN_MASK,
        NONIB_HUMAN_CAP,
        min_channel_counts=np.array([0, 0, min_rm]),
    )
    counts = result['counts']
    expected_retained = 0.0
    for channel_idx, channel in enumerate(CHANNEL_NAMES):
        uplift = channel_uplift(
            summary['P_CHURN'],
            channel,
            cr_multipliers.get(channel, 1.0),
        )
        expected_retained += float(np.sum(counts[:, channel_idx] * uplift))

    total_cost = float(np.sum(counts * CHANNEL_COSTS))
    total_emu = float(np.sum(counts * emu_matrix))
    return {
        'Branch': 'Non-IB',
        'Scenario': name,
        'Status': result['status'],
        'Budget': budget,
        'COGS': total_cost,
        'EMU': total_emu,
        'Expected Conversions': np.nan,
        'Expected Retained': expected_retained,
        'SMS': int(counts[:, 0].sum()),
        'Telesales': int(counts[:, 1].sum()),
        'RM': int(counts[:, 2].sum()),
        'Incremental ROI': (total_emu - total_cost) / total_cost if total_cost else 0,
        'EMU/COGS': total_emu / total_cost if total_cost else 0,
    }


def stress_reoptimized():
    rows = [
        solve_nonib_scenario('Baseline', NONIB_BUDGET),
        solve_nonib_scenario(
            'Adverse CR/FP re-optimized',
            NONIB_BUDGET,
            cr_multipliers={'Telesales': 0.85, 'RM': 0.85},
            fp_multiplier=1.2,
        ),
        solve_nonib_scenario('Budget cut -20%', int(NONIB_BUDGET * 0.8)),
        solve_nonib_scenario('SMS CR -25%', NONIB_BUDGET, cr_multipliers={'SMS': 0.75}),
    ]
    df = pd.DataFrame(rows)
    cols = [
        'Branch', 'Scenario', 'Status', 'Budget', 'COGS', 'EMU',
        'Expected Conversions', 'Expected Retained', 'SMS', 'Telesales', 'RM',
        'Incremental ROI', 'EMU/COGS',
    ]
    df = df.reindex(columns=cols)
    df.to_csv(os.path.join(BASE_DIR, 'stress_reoptimized_nonib.csv'), index=False)
    return df


def main():
    kpi = business_kpis()
    pnl = pnl_projection(kpi)
    stress = stress_reoptimized()

    print('Business KPI')
    print(kpi.to_string(index=False))
    print('\nQuarterly P&L')
    print(pnl[pnl['View'] == 'Quarterly'].to_string(index=False))
    print('\nRe-optimized Non-IB stress')
    print(stress.to_string(index=False))


if __name__ == '__main__':
    main()
