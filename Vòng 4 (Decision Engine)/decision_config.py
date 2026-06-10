import os

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix, vstack


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
IB_MASTER_FILENAME = 'gcon_customer_month_clean.parquet'


def solve_channel_milp(
    emu_matrix,
    channel_costs,
    budget_limit,
    human_channel_mask,
    human_cap,
    *,
    min_emu=0.0,
    time_limit=180,
    mip_rel_gap=0.001,
):
    emu_matrix = np.asarray(emu_matrix, dtype=float)
    channel_costs = np.asarray(channel_costs, dtype=float)
    human_channel_mask = np.asarray(human_channel_mask, dtype=float)
    n_customers, n_channels = emu_matrix.shape

    selectable = np.isfinite(emu_matrix) & (emu_matrix > min_emu)
    candidate_mask = selectable.any(axis=1)
    allocations = np.zeros((n_customers, n_channels), dtype=int)

    if not candidate_mask.any():
        return {
            'allocations': allocations,
            'status': 'no_positive_emu',
            'message': 'No customer-channel pair has positive EMU.',
            'objective': 0.0,
        }

    candidate_idx = np.where(candidate_mask)[0]
    candidate_emu = emu_matrix[candidate_mask]
    candidate_selectable = selectable[candidate_mask]
    n_candidates = len(candidate_idx)
    n_vars = n_candidates * n_channels

    objective = -candidate_emu.ravel()
    upper_bounds = candidate_selectable.astype(float).ravel()
    bounds = Bounds(np.zeros(n_vars), upper_bounds)

    costs = np.tile(channel_costs, n_candidates)
    humans = np.tile(human_channel_mask, n_candidates)
    rows = []
    rows.append(csr_matrix(costs.reshape(1, -1)))
    rows.append(csr_matrix(humans.reshape(1, -1)))

    row_index = np.repeat(np.arange(n_candidates), n_channels)
    col_index = np.arange(n_vars)
    customer_matrix = csr_matrix(
        (np.ones(n_vars), (row_index, col_index)),
        shape=(n_candidates, n_vars),
    )
    rows.append(customer_matrix)
    constraint_matrix = vstack(rows, format='csr')

    lower = np.concatenate(([-np.inf, -np.inf], np.zeros(n_candidates)))
    upper = np.concatenate(([budget_limit, human_cap], np.ones(n_candidates)))
    constraints = LinearConstraint(constraint_matrix, lower, upper)

    result = milp(
        c=objective,
        integrality=np.ones(n_vars),
        bounds=bounds,
        constraints=constraints,
        options={
            'time_limit': time_limit,
            'mip_rel_gap': mip_rel_gap,
            'disp': False,
        },
    )

    if result.x is None:
        return {
            'allocations': allocations,
            'status': f'status_{result.status}',
            'message': result.message,
            'objective': 0.0,
        }

    selected = (result.x.reshape(n_candidates, n_channels) > 0.5).astype(int)
    allocations[candidate_idx] = selected
    return {
        'allocations': allocations,
        'status': f'status_{result.status}',
        'message': result.message,
        'objective': float(np.sum(allocations * emu_matrix)),
    }


def solve_grouped_channel_milp(
    group_emu_matrix,
    group_sizes,
    channel_costs,
    budget_limit,
    human_channel_mask,
    human_cap,
    *,
    min_channel_counts=None,
    min_emu=0.0,
    time_limit=180,
    mip_rel_gap=0.001,
):
    group_emu_matrix = np.asarray(group_emu_matrix, dtype=float)
    group_sizes = np.asarray(group_sizes, dtype=float)
    channel_costs = np.asarray(channel_costs, dtype=float)
    human_channel_mask = np.asarray(human_channel_mask, dtype=float)
    n_groups, n_channels = group_emu_matrix.shape
    n_vars = n_groups * n_channels

    selectable = np.isfinite(group_emu_matrix) & (group_emu_matrix > min_emu)
    objective = -group_emu_matrix.ravel()
    upper_bounds = (selectable * group_sizes[:, None]).ravel()
    bounds = Bounds(np.zeros(n_vars), upper_bounds)

    costs = np.tile(channel_costs, n_groups)
    humans = np.tile(human_channel_mask, n_groups)
    rows = [
        csr_matrix(costs.reshape(1, -1)),
        csr_matrix(humans.reshape(1, -1)),
    ]
    lower = [-np.inf, -np.inf]
    upper = [budget_limit, human_cap]

    if min_channel_counts is not None:
        min_channel_counts = np.asarray(min_channel_counts, dtype=float)
        for channel_idx, min_count in enumerate(min_channel_counts):
            if min_count <= 0:
                continue
            channel_row = np.zeros(n_vars)
            channel_row[channel_idx::n_channels] = 1
            rows.append(csr_matrix(channel_row.reshape(1, -1)))
            lower.append(min_count)
            upper.append(np.inf)

    row_index = np.repeat(np.arange(n_groups), n_channels)
    col_index = np.arange(n_vars)
    group_matrix = csr_matrix(
        (np.ones(n_vars), (row_index, col_index)),
        shape=(n_groups, n_vars),
    )
    rows.append(group_matrix)
    constraint_matrix = vstack(rows, format='csr')

    lower = np.concatenate((np.asarray(lower, dtype=float), np.zeros(n_groups)))
    upper = np.concatenate((np.asarray(upper, dtype=float), group_sizes))
    constraints = LinearConstraint(constraint_matrix, lower, upper)

    result = milp(
        c=objective,
        integrality=np.ones(n_vars),
        bounds=bounds,
        constraints=constraints,
        options={
            'time_limit': time_limit,
            'mip_rel_gap': mip_rel_gap,
            'disp': False,
        },
    )

    if result.x is None:
        return {
            'counts': np.zeros((n_groups, n_channels), dtype=int),
            'status': f'status_{result.status}',
            'message': result.message,
            'objective': 0.0,
        }

    counts = np.rint(result.x.reshape(n_groups, n_channels)).astype(int)
    return {
        'counts': counts,
        'status': f'status_{result.status}',
        'message': result.message,
        'objective': float(np.sum(counts * group_emu_matrix)),
    }


def _numeric_col(df, name, default=0):
    if name in df.columns:
        return pd.to_numeric(df[name], errors='coerce').fillna(default)
    return pd.Series(default, index=df.index)


def calculate_asset_score(df):
    asset_value = (
        _numeric_col(df, 'AVG_TD_BALANCE', 0)
        + _numeric_col(df, 'AVG_CA_BALANCE', 0)
        + _numeric_col(df, 'AVG_LOAN_AMOUNT', 0)
        + _numeric_col(df, 'TOTAL_FINANCIAL_VALUE', 0)
        + _numeric_col(df, 'NET_WORTH_PROXY', 0).clip(lower=0)
        + _numeric_col(df, 'TOTAL_DEPOSIT', 0)
        + _numeric_col(df, 'TD_BALANCE_LAST', 0)
        + _numeric_col(df, 'CA_BALANCE_LAST', 0)
        + _numeric_col(df, 'LOAN_AMOUNT_LAST', 0)
    )
    if (asset_value > 0).any():
        return asset_value.rank(method='average', pct=True).fillna(0)
    return (_numeric_col(df, 'CUSTOMER_NUMBER', 0) % 10000) / 10000.0


def attach_ib_register_date(df, base_dir, required=True):
    out = df.copy()
    has_register_date = (
        'IB_REGISTER_DATE' in out.columns
        and pd.to_datetime(out['IB_REGISTER_DATE'], errors='coerce').notna().any()
    )
    if has_register_date:
        return out

    register_sources = [
        os.path.join(base_dir, "NBFO_IB", "processed_data", IB_MASTER_FILENAME),
        os.path.join(base_dir, "cleaned_data", "customer_clean.parquet"),
        os.path.join(base_dir, "cleaned_data", "customer_clean.csv"),
    ]
    register_path = next((path for path in register_sources if os.path.exists(path)), None)
    if register_path is None:
        if required:
            raise FileNotFoundError(
                "Missing customer register-date source. Run NBFO_IB/EDA_Feature_engineering.ipynb "
                f"to create NBFO_IB/processed_data/{IB_MASTER_FILENAME}, or run clean.ipynb "
                "to create cleaned_data/customer_clean.parquet."
            )
        return out

    if register_path.endswith('.csv'):
        register_dates = pd.read_csv(register_path, usecols=['CUSTOMER_NUMBER', 'IB_REGISTER_DATE'])
    else:
        register_dates = pd.read_parquet(register_path, columns=['CUSTOMER_NUMBER', 'IB_REGISTER_DATE'])
    register_dates['IB_REGISTER_DATE'] = pd.to_datetime(
        register_dates['IB_REGISTER_DATE'],
        errors='coerce',
    )
    register_dates = (
        register_dates.dropna(subset=['IB_REGISTER_DATE'])
        .sort_values('IB_REGISTER_DATE')
        .drop_duplicates('CUSTOMER_NUMBER', keep='first')
    )

    if 'IB_REGISTER_DATE' in out.columns:
        out = out.drop(columns=['IB_REGISTER_DATE'])
    return out.merge(register_dates, on='CUSTOMER_NUMBER', how='left')


def prepare_ib_features(df):
    out = df.copy()
    out['IB_REGISTER_DATE'] = pd.to_datetime(out.get('IB_REGISTER_DATE'), errors='coerce')
    out['login_count'] = _numeric_col(out, 'login_count', np.nan)
    missing_login = out['login_count'].isna()
    if missing_login.any():
        activity_fallback = _numeric_col(out, 'ACTIVITY_NO_SUM', 0)
        out.loc[missing_login, 'login_count'] = activity_fallback.loc[missing_login]
    out['login_count'] = out['login_count'].fillna(0)

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


def _assign_ib_segment_by_profile(row):
    if row['AVG_LOAN_AMOUNT'] > 500_000_000:
        return 'V1_HV_Borrower'
    if row['AVG_TD_BALANCE'] > 100_000_000 and row['has_loan'] == 0:
        return 'V2_Conservative'
    if row['product_depth'] >= 3 and row['AVG_TD_BALANCE'] > 200_000_000:
        return 'V3_Multi_Premium'
    if row['has_card'] == 1 and row['has_loan'] == 1:
        return 'N1_Active_Digital'
    return 'N2_Semi_Digital'


def assign_ib_segment(row):
    register_date = row.get('IB_REGISTER_DATE')
    register_year = register_date.year if pd.notna(register_date) else None

    if register_year is None:
        return DEFAULT_SEGMENT

    # For customers already on IB by 2019, zero login is a real dormant signal.
    # For 2020/21 registrants, zero login in the 2019 snapshot is an artifact.
    if register_year not in {2020, 2021} and row['login_count'] == 0:
        return 'N3_Dormant'
    return _assign_ib_segment_by_profile(row)


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
