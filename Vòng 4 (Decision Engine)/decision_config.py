import numpy as np
import pandas as pd


IB_SEGMENT_LABELS = {
    'V1_HV_Borrower': 'V1 HV Borrower',
    'V2_Conservative': 'V2 Conservative',
    'V3_Multi_Premium': 'V3 Multi Premium',
    'N1_Active_Digital': 'N1 Active Digital',
    'N2_Semi_Digital': 'N2 Semi Digital',
    'N3_Dormant': 'N3 Dormant',
}

IB_BUY_RATE = {
    'V1_HV_Borrower': 0.0456,
    'V2_Conservative': 0.0392,
    'V3_Multi_Premium': 0.1117,
    'N1_Active_Digital': 0.0681,
    'N2_Semi_Digital': 0.0213,
    'N3_Dormant': 0.0513,
}

NONIB_CLUSTER_NAMES = {
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

NONIB_CLUSTER_MAPPING = {
    'C0_Traditional': 'N2_Semi_Digital',
    'C1_HV_Traditional': 'N1_Active_Digital',
    'C2_Senior_HV': 'N2_Semi_Digital',
    'C3_Ultra_Saver': 'V2_Conservative',
    'C4_Multi_Saver': 'N2_Semi_Digital',
    'C5_HV_Saver': 'V2_Conservative',
    'C6_Stable_Senior': 'V2_Conservative',
    'C7_HV_Borrower': 'V1_HV_Borrower',
    'P0_Dormant': None,
}

NONIB_BUY_RATE_PROXY = {
    cluster: (IB_BUY_RATE[mapped] if mapped else 0.0)
    for cluster, mapped in NONIB_CLUSTER_MAPPING.items()
}

FUM_MATRIX = {
    'V1_HV_Borrower': {'TP': 5_000_000, 'FP': -50_000, 'FN': -30_000_000},
    'V2_Conservative': {'TP': 5_000_000, 'FP': -50_000, 'FN': -30_000_000},
    'V3_Multi_Premium': {'TP': 5_000_000, 'FP': -50_000, 'FN': -30_000_000},
    'N1_Active_Digital': {'TP': 5_000_000, 'FP': -50_000, 'FN': 0},
    'N2_Semi_Digital': {'TP': 5_000_000, 'FP': -50_000, 'FN': 0},
    'N3_Dormant': {'TP': 5_000_000, 'FP': -50_000, 'FN': 0},
}

VIP_SEGMENTS = {'V1_HV_Borrower', 'V2_Conservative', 'V3_Multi_Premium'}
DEFAULT_SEGMENT = 'N2_Semi_Digital'


def _numeric_col(df, name, default=0):
    if name in df.columns:
        return pd.to_numeric(df[name], errors='coerce').fillna(default)
    return pd.Series(default, index=df.index)


def prepare_ib_features(df):
    out = df.copy()
    out['login_count'] = _numeric_col(out, 'login_count', np.nan)
    if out['login_count'].isna().all():
        out['login_count'] = _numeric_col(out, 'ACTIVITY_NO_SUM', 0)

    out['has_loan'] = (
        (_numeric_col(out, 'OWN_LENDING', 0) > 0)
        | (_numeric_col(out, 'COUNT_OF_LOAN', 0) > 0)
        | (_numeric_col(out, 'AVG_LOAN_AMOUNT', 0) > 0)
    ).astype(int)
    out['has_td'] = (
        (_numeric_col(out, 'OWN_TERM_DEPOSIT', 0) > 0)
        | (_numeric_col(out, 'COUNT_TD_ACCT', 0) > 0)
        | (_numeric_col(out, 'AVG_TD_BALANCE', 0) > 0)
    ).astype(int)
    out['has_card'] = (
        (_numeric_col(out, 'OWN_CREDIT_CARD', 0) > 0)
        | (_numeric_col(out, 'OWN_DEBIT_CARD', 0) > 0)
        | (_numeric_col(out, 'COUNT_CREDITCARD', 0) > 0)
        | (_numeric_col(out, 'COUNT_DEBITCARD', 0) > 0)
    ).astype(int)

    if 'product_depth' not in out.columns:
        product_cols = [
            ('OWN_CURRENT_ACCOUNT', 'COUNT_CA_ACCT', 'AVG_CA_BALANCE'),
            ('OWN_TERM_DEPOSIT', 'COUNT_TD_ACCT', 'AVG_TD_BALANCE'),
            ('OWN_CREDIT_CARD', 'COUNT_CREDITCARD', None),
            ('OWN_DEBIT_CARD', 'COUNT_DEBITCARD', None),
            ('OWN_LENDING', 'COUNT_OF_LOAN', 'AVG_LOAN_AMOUNT'),
        ]
        depth = pd.Series(0, index=out.index)
        for own_col, count_col, balance_col in product_cols:
            has_product = _numeric_col(out, own_col, 0) > 0
            has_product |= _numeric_col(out, count_col, 0) > 0
            if balance_col:
                has_product |= _numeric_col(out, balance_col, 0) > 0
            depth += has_product.astype(int)
        out['product_depth'] = depth

    out['AVG_LOAN_AMOUNT'] = _numeric_col(out, 'AVG_LOAN_AMOUNT', 0)
    out['AVG_TD_BALANCE'] = _numeric_col(out, 'AVG_TD_BALANCE', 0)
    return out


def assign_ib_segment(row):
    if row['login_count'] == 0:
        return 'N3_Dormant'
    if row['AVG_LOAN_AMOUNT'] > 500_000_000:
        return 'V1_HV_Borrower'
    if row['AVG_TD_BALANCE'] > 100_000_000 and row['has_loan'] == 0:
        return 'V2_Conservative'
    if row['product_depth'] >= 3 and row['AVG_TD_BALANCE'] > 200_000_000:
        return 'V3_Multi_Premium'
    if row['has_card'] == 1 and row['has_loan'] == 1:
        return 'N1_Active_Digital'
    return 'N2_Semi_Digital'


def add_ib_segments(df):
    out = prepare_ib_features(df)
    out['SEGMENT'] = out.apply(assign_ib_segment, axis=1)
    out['MAPPED_IB_SEGMENT'] = out['SEGMENT']
    out['CUSTOMER_TYPE'] = 'IB'
    return out


def add_nonib_proxy(df):
    out = df.copy()
    if 'CLUSTER_NAME' not in out.columns:
        out['CLUSTER_NAME'] = out['CLUSTER'].map(NONIB_CLUSTER_NAMES)
    out['SEGMENT_CLUSTER'] = out['CLUSTER_NAME'].fillna(out.get('PERSONA_NAME', 'UNKNOWN'))
    out['MAPPED_IB_SEGMENT'] = out['SEGMENT_CLUSTER'].map(NONIB_CLUSTER_MAPPING)
    out['PROBABILITY'] = out['SEGMENT_CLUSTER'].map(NONIB_BUY_RATE_PROXY).fillna(0.0)
    out['CUSTOMER_TYPE'] = 'Non-IB'
    out['BUY_RATE_PROXY'] = out['PROBABILITY']
    return out


def segment_for_economics(row):
    if row.get('CUSTOMER_TYPE') == 'Non-IB':
        return row.get('MAPPED_IB_SEGMENT') or DEFAULT_SEGMENT
    return row.get('SEGMENT') or row.get('MAPPED_IB_SEGMENT') or DEFAULT_SEGMENT
