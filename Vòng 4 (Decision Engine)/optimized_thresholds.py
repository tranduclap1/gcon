import os

import numpy as np
import pandas as pd

from decision_config import (
    DEFAULT_SEGMENT,
    FUM_MATRIX,
    VIP_SEGMENTS,
    add_ib_segments,
    attach_ib_register_date,
)


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

CHANNELS = {
    'SMS': {'cost': 5_000, 'cr': 0.02},
    'Telesales': {'cost': 50_000, 'cr': 0.05},
    'RM': {'cost': 2_000_000, 'cr': 0.15},
}
CHANNEL_NAMES = ['SMS', 'Telesales', 'RM']

IB_CAC_CAP = 750_000
NONIB_CAC_CAP = 1_500_000
TP_RETENTION = 50_000_000
FP_CONTACT = -50_000


def uplift(p, channel):
    cr = CHANNELS[channel]['cr']
    return 4 * p * (1 - p) * cr


def emu_cross_sell(p, tp, fp, fn, channel):
    u = uplift(p, channel)
    fp_probability = 1 - p - u
    return u * (tp - fn) + fp_probability * fp - CHANNELS[channel]['cost']


def emu_retention(p, fn, channel):
    u = uplift(p, channel)
    fp_probability = 1 - p - u
    return u * (TP_RETENTION - fn) + fp_probability * FP_CONTACT - CHANNELS[channel]['cost']


def first_break_even(values, scores):
    valid = scores[values >= 0]
    return float(valid.min()) if len(valid) else np.nan


def optimize_threshold(df, score_col, emu_col, channel, cac_cap):
    if df.empty:
        return None

    scores = df[score_col].to_numpy(dtype=float)
    emus = df[emu_col].to_numpy(dtype=float)
    expected = df['EXPECTED_CONVERSIONS'].to_numpy(dtype=float)
    cost_per_contact = CHANNELS[channel]['cost']
    break_even = first_break_even(emus, scores)
    if np.isnan(break_even):
        return {
            'Threshold': np.nan,
            'Break-even Threshold': np.nan,
            'Selected': 0,
            'COGS': 0.0,
            'Expected Conversions': 0.0,
            'Expected Retained': 0.0,
            'EMU': 0.0,
            'CAC': np.nan,
            'Incremental ROI': np.nan,
            'CAC Cap': cac_cap,
            'Optimization Status': 'No ROI',
        }

    candidate_thresholds = np.unique(
        np.concatenate([
            [break_even],
            np.quantile(scores, np.linspace(0, 1, 101)),
        ])
    )
    candidate_thresholds = candidate_thresholds[candidate_thresholds >= break_even]

    best = None
    fallback = None
    for threshold in candidate_thresholds:
        mask = (scores >= threshold) & (emus > 0)
        n = int(mask.sum())
        if n == 0:
            continue
        cogs = float(n * cost_per_contact)
        exp_conv = float(expected[mask].sum())
        emu = float(emus[mask].sum())
        cac = cogs / exp_conv if exp_conv > 0 else np.inf
        incremental_roi = (emu - cogs) / cogs if cogs > 0 else -np.inf
        record = {
            'Threshold': float(threshold),
            'Break-even Threshold': float(break_even) if not np.isnan(break_even) else np.nan,
            'Selected': n,
            'COGS': cogs,
            'Expected Conversions': exp_conv,
            'Expected Retained': exp_conv,
            'EMU': emu,
            'CAC': cac,
            'Incremental ROI': incremental_roi,
        }
        if fallback is None or (emu, incremental_roi) > (fallback['EMU'], fallback['Incremental ROI']):
            fallback = record
        if cac <= cac_cap:
            if best is None or (emu, incremental_roi) > (best['EMU'], best['Incremental ROI']):
                best = record

    chosen = best or fallback
    if chosen is not None:
        chosen['CAC Cap'] = cac_cap
        chosen['Optimization Status'] = 'CAC feasible' if best is not None else 'Max EMU, CAC cap unmet'
    return chosen


def load_ib_candidates():
    scores_path = os.path.join(
        BASE_DIR,
        'NBFO_IB',
        'saved_models',
        'gcon_test_scores_best_xgboost_calibrated_sigmoid.parquet',
    )
    score = pd.read_parquet(scores_path)
    score = score.rename(columns={'SUBSCRIPTION_PROPENSITY': 'PROBABILITY'})

    features_path = os.path.join(BASE_DIR, 'NBFO_IB', 'processed_data', 'gcon_model_input.parquet')
    features = pd.read_parquet(features_path)
    if 'MONTH' in features.columns:
        features = features.sort_values('MONTH').groupby('CUSTOMER_NUMBER').last().reset_index()
    else:
        features = features.groupby('CUSTOMER_NUMBER').last().reset_index()
    features = attach_ib_register_date(features, BASE_DIR)
    segments = add_ib_segments(features)

    out = score.merge(
        segments[['CUSTOMER_NUMBER', 'SEGMENT', 'MAPPED_IB_SEGMENT']],
        on='CUSTOMER_NUMBER',
        how='left',
    )
    out['SEGMENT'] = out['SEGMENT'].fillna(DEFAULT_SEGMENT)
    out['MAPPED_IB_SEGMENT'] = out['MAPPED_IB_SEGMENT'].fillna(out['SEGMENT'])
    out['TP'] = out['MAPPED_IB_SEGMENT'].map(lambda x: FUM_MATRIX.get(x, FUM_MATRIX[DEFAULT_SEGMENT])['TP'])
    out['FP'] = out['MAPPED_IB_SEGMENT'].map(lambda x: FUM_MATRIX.get(x, FUM_MATRIX[DEFAULT_SEGMENT])['FP'])
    out['FN'] = out['MAPPED_IB_SEGMENT'].map(lambda x: FUM_MATRIX.get(x, FUM_MATRIX[DEFAULT_SEGMENT])['FN'])
    return out


def optimize_ib_thresholds():
    df = load_ib_candidates()
    rows = []
    for segment in sorted(df['MAPPED_IB_SEGMENT'].dropna().unique()):
        seg_df = df[df['MAPPED_IB_SEGMENT'] == segment]
        for product in sorted(seg_df['PRODUCT_NAME'].dropna().unique()):
            product_df = seg_df[seg_df['PRODUCT_NAME'] == product].copy()
            for channel in CHANNEL_NAMES:
                if channel == 'RM' and segment not in VIP_SEGMENTS:
                    rows.append({
                        'Branch': 'IB',
                        'Segment': segment,
                        'Product_or_Risk': product,
                        'Channel': channel,
                        'Threshold': np.nan,
                        'Reason': 'RM not eligible for non-VIP segment',
                    })
                    continue
                product_df['EXPECTED_CONVERSIONS'] = uplift(product_df['PROBABILITY'], channel)
                product_df['EMU'] = emu_cross_sell(
                    product_df['PROBABILITY'],
                    product_df['TP'],
                    product_df['FP'],
                    product_df['FN'],
                    channel,
                )
                chosen = optimize_threshold(product_df, 'PROBABILITY', 'EMU', channel, IB_CAC_CAP)
                if chosen is None:
                    continue
                chosen.update({
                    'Branch': 'IB',
                    'Segment': segment,
                    'Product_or_Risk': product,
                    'Channel': channel,
                    'Reason': 'Optimized persona-product-channel threshold',
                })
                rows.append(chosen)
    result = pd.DataFrame(rows)
    result.to_csv(os.path.join(BASE_DIR, 'optimized_thresholds_ib.csv'), index=False)
    return result


def load_nonib_candidates():
    alloc = pd.read_csv(os.path.join(BASE_DIR, 'final_allocations_nonib.csv'))
    alloc['RECOMMENDED_CHANNEL'] = alloc['RECOMMENDED_CHANNEL'].fillna('None')
    alloc['EXPECTED_LOSS_SCORE'] = alloc['P_CHURN'] * alloc['CLV_5YR']
    alloc['LOSS_PERCENTILE'] = (
        alloc.groupby('SEGMENT_CLUSTER')['EXPECTED_LOSS_SCORE']
        .rank(method='average', pct=True)
        .fillna(0.0)
    )
    alloc['FN'] = pd.to_numeric(alloc['FN'], errors='coerce').fillna(0)
    return alloc


def optimize_nonib_thresholds():
    df = load_nonib_candidates()
    rows = []
    for segment in sorted(df['SEGMENT_CLUSTER'].dropna().unique()):
        seg_df = df[(df['SEGMENT_CLUSTER'] == segment) & (df['CHURN_AT_RISK'] == 1)].copy()
        if seg_df.empty:
            continue
        for channel in CHANNEL_NAMES:
            if channel == 'RM' and (seg_df['FN'].max() >= 0):
                rows.append({
                    'Branch': 'Non-IB',
                    'Segment': segment,
                    'Product_or_Risk': 'Expected loss',
                    'Channel': channel,
                    'Threshold': np.nan,
                    'Reason': 'RM not eligible for non-VIP cluster',
                })
                continue
            work = seg_df.copy()
            work['EXPECTED_CONVERSIONS'] = uplift(work['P_CHURN'], channel)
            work['EMU'] = emu_retention(work['P_CHURN'], work['FN'], channel)
            chosen = optimize_threshold(
                work,
                'LOSS_PERCENTILE',
                'EMU',
                channel,
                NONIB_CAC_CAP,
            )
            if chosen is None:
                continue
            threshold = chosen.get('Threshold')
            chosen['Threshold Unit'] = 'Expected-loss percentile'
            chosen['Target Top Percent'] = (1 - threshold) if pd.notna(threshold) else np.nan
            chosen.update({
                'Branch': 'Non-IB',
                'Segment': segment,
                'Product_or_Risk': 'Expected-loss percentile',
                'Channel': channel,
                'Reason': 'Optimized cluster-channel threshold on percentile rank of P_CHURN x CLV_5YR',
            })
            rows.append(chosen)
    result = pd.DataFrame(rows)
    result.to_csv(os.path.join(BASE_DIR, 'optimized_thresholds_nonib.csv'), index=False)
    return result


def main():
    ib = optimize_ib_thresholds()
    nonib = optimize_nonib_thresholds()
    combined = pd.concat([ib, nonib], ignore_index=True, sort=False)
    combined.to_csv(os.path.join(BASE_DIR, 'optimized_thresholds.csv'), index=False)

    print('IB optimized thresholds sample:')
    print(ib.head(12).to_string(index=False))
    print('\nNon-IB optimized thresholds:')
    print(nonib.to_string(index=False))


if __name__ == '__main__':
    main()
