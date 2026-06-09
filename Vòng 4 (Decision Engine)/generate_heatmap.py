import pandas as pd
import numpy as np

import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def frame_to_markdown(df):
    rows = [[''] + [str(col) for col in df.columns]]
    rows.extend([[str(idx)] + [str(value) for value in row] for idx, row in df.iterrows()])
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    header = '| ' + ' | '.join(rows[0][i].ljust(widths[i]) for i in range(len(widths))) + ' |'
    separator = '| ' + ' | '.join('-' * widths[i] for i in range(len(widths))) + ' |'
    body = ['| ' + ' | '.join(row[i].ljust(widths[i]) for i in range(len(widths))) + ' |' for row in rows[1:]]
    return '\n'.join([header, separator] + body)

print("Loading data for Heatmap generation...")
# We will simulate the dataframe logic here to quickly run 16 scenarios
# 1. Load probabilities (IB)
path_ib_prob = os.path.join(BASE_DIR, "NBFO_IB", "saved_models", "gcon_test_scores_best_xgboost_calibrated_sigmoid.parquet")
df_ib = pd.read_parquet(path_ib_prob)
df_ib_prob = df_ib.groupby('CUSTOMER_NUMBER')['SUBSCRIPTION_PROPENSITY'].max().reset_index()
df_ib_prob.rename(columns={'SUBSCRIPTION_PROPENSITY': 'PROBABILITY'}, inplace=True)

path_ib_personas = os.path.join(BASE_DIR, "NBFO_IB", "processed_data", "ib_final_personas.parquet")
if os.path.exists(path_ib_personas):
    df_ib_personas = pd.read_parquet(path_ib_personas)
    df_ib_prob = df_ib_prob.merge(df_ib_personas[['CUSTOMER_NUMBER', 'PERSONA']], on='CUSTOMER_NUMBER', how='left')
else:
    print("IB persona file not found; defaulting IB personas to Standard.")
    df_ib_prob['PERSONA'] = 'Standard'
df_ib_prob['PERSONA'] = df_ib_prob['PERSONA'].fillna('Standard')

# 2. Load Non-IB
path_nonib = os.path.join(BASE_DIR, "Cluster_nonIB", "output", "nonib_final_personas.parquet")
df_nonib = pd.read_parquet(path_nonib)
df_nonib = df_nonib[['CUSTOMER_NUMBER', 'PERSONA_NAME']].copy()
df_nonib.rename(columns={'PERSONA_NAME': 'PERSONA'}, inplace=True)

base_cr_nonib = {
    'Senior High-Value Heavy Borrower': 0.9183,
    'High-Value Heavy Borrower': 0.9144,
    'High-Value Traditional': 0.7803,
    'Dormant / Ngủ đông': 0.4018,
    'Senior High-Value Saver': 0.1802,
    'High-Value Saver': 0.1562,
    'Traditional': 0.1068
}
df_nonib['PROBABILITY'] = df_nonib['PERSONA'].map(base_cr_nonib).fillna(0.10)
np.random.seed(42)
df_nonib['PROBABILITY'] *= np.random.uniform(0.9, 1.1, len(df_nonib))

df_master = pd.concat([df_ib_prob, df_nonib], ignore_index=True)

# Unique Financial Utility Matrix (FUM) for each Persona
fum_matrix = {
    'Wealthy Passive': {'TP': 5_000_000, 'FP': -50_000, 'FN': -30_000_000},
    'Digital VIP':     {'TP': 5_000_000, 'FP': -80_000, 'FN': -20_000_000},
    'Mass Active':     {'TP': 5_000_000, 'FP': -20_000, 'FN': 0},
    'Young Digital':   {'TP': 5_000_000, 'FP': -100_000, 'FN': 0},
    'Standard':        {'TP': 5_000_000, 'FP': -40_000, 'FN': 0},

    'Senior High-Value Saver':          {'TP': 2_060_000, 'FP': -10_000, 'FN': -5_000_000},
    'Traditional':                      {'TP': 2_060_000, 'FP': -5_000,  'FN': 0},
    'Dormant / Ngủ đông':               {'TP': 2_060_000, 'FP': -2_000,  'FN': 0},
    'High-Value Saver':                 {'TP': 2_060_000, 'FP': -15_000, 'FN': -1_000_000},
    'High-Value Heavy Borrower':        {'TP': 2_060_000, 'FP': -15_000, 'FN': -2_000_000},
    'Senior High-Value Heavy Borrower': {'TP': 2_060_000, 'FP': -10_000, 'FN': -4_000_000},
    'High-Value Traditional':           {'TP': 2_060_000, 'FP': -8_000,  'FN': -500_000},
}

vip_personas = ['Wealthy Passive', 'Digital VIP', 'Senior High-Value Saver', 'Senior High-Value Heavy Borrower']
df_master['IS_VIP'] = df_master['PERSONA'].isin(vip_personas)
df_master['TP'] = df_master['PERSONA'].map(lambda x: fum_matrix.get(x, fum_matrix['Standard'])['TP'])
df_master['FP'] = df_master['PERSONA'].map(lambda x: fum_matrix.get(x, fum_matrix['Standard'])['FP'])
df_master['FN'] = df_master['PERSONA'].map(lambda x: fum_matrix.get(x, fum_matrix['Standard'])['FN'])
df_master['ASSET_SCORE'] = (df_master['CUSTOMER_NUMBER'] % 10000) / 10000.0

channel_names = np.array(['SMS', 'Telesales', 'RM'])
channel_costs = np.array([5_000, 50_000, 2_000_000])

def threshold_emu_formula(p, cr, cost, tp_val, fn_val, fp_val):
    uplift = 4 * p * (1 - p) * cr
    return uplift * (tp_val - fn_val) + fp_val - cost

def calculate_channel_thresholds(channels):
    thresholds = {}
    ps = np.linspace(0, 1, 5000)
    for persona, economics in fum_matrix.items():
        thresholds[persona] = {}
        is_vip = persona in vip_personas
        for ch_name, ch_data in channels.items():
            if ch_name == 'RM' and not is_vip:
                thresholds[persona][ch_name] = None
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
            thresholds[persona][ch_name] = float(valid_ps[0]) if len(valid_ps) > 0 else None
    return thresholds

def build_eligibility_matrix(df, thresholds):
    eligible = np.full((len(df), len(channel_names)), False)
    for channel_idx, ch_name in enumerate(channel_names):
        persona_thresholds = df['PERSONA'].map(
            lambda persona: thresholds.get(persona, thresholds['Standard']).get(ch_name)
        )
        has_threshold = persona_thresholds.notna()
        eligible[:, channel_idx] = has_threshold & (df['PROBABILITY'] >= persona_thresholds.astype(float))
    return eligible

baseline_channels = {
    'SMS': {'cost': 5_000, 'cr': 0.02},
    'Telesales': {'cost': 50_000, 'cr': 0.05},
    'RM': {'cost': 2_000_000, 'cr': 0.15}
}
baseline_thresholds = calculate_channel_thresholds(baseline_channels)
baseline_eligibility_matrix = build_eligibility_matrix(df_master, baseline_thresholds)

def run_scenario(rm_cr_drop, vip_fp_inc):
    channels = {
        'SMS': {'cost': 5_000, 'cr': 0.02},
        'Telesales': {'cost': 50_000, 'cr': 0.05},
        'RM': {'cost': 2_000_000, 'cr': 0.15 * (1 - rm_cr_drop)} # Apply CR drop
    }
    
    P_base = df_master['PROBABILITY']
    uplift_sms = 4 * P_base * (1 - P_base) * channels['SMS']['cr']
    uplift_tele = 4 * P_base * (1 - P_base) * channels['Telesales']['cr']
    uplift_rm = 4 * P_base * (1 - P_base) * channels['RM']['cr']

    tp_array = df_master['TP']
    fn_array = df_master['FN']
    fp_array = np.where(df_master['IS_VIP'], df_master['FP'] * (1 + vip_fp_inc), df_master['FP'])

    emu_sms = uplift_sms * (tp_array - fn_array) + fp_array - channels['SMS']['cost']
    emu_tele = uplift_tele * (tp_array - fn_array) + fp_array - channels['Telesales']['cost']
    emu_rm = uplift_rm * (tp_array - fn_array) + fp_array - channels['RM']['cost']
    
    tie_breaker = 1e-6 * df_master['ASSET_SCORE']
    emu_matrix = np.vstack([emu_sms + tie_breaker, emu_tele + tie_breaker, emu_rm + tie_breaker]).T
    emu_matrix = np.where(baseline_eligibility_matrix, emu_matrix, -999_999_999)
    
    # Fast Greedy Solve
    N = len(df_master)
    allocations = np.full((N, 3), 0)
    best_channels = np.argmax(emu_matrix, axis=1)
    max_emus = np.max(emu_matrix, axis=1)
    
    valid = max_emus > 0
    
    # No lower-bound constraint for RM. RM is selected only when it wins on EMU.
    rm_forced_count = 0
    forced_rm_idx = np.array([], dtype=int)
        
    tele_rm_indices = np.where(valid & ((best_channels == 1) | (best_channels == 2)))[0]
    tele_rm_remaining = np.setdiff1d(tele_rm_indices, forced_rm_idx, assume_unique=True)
    tele_rm_remaining = tele_rm_remaining[np.argsort(-max_emus[tele_rm_remaining])]
    
    remaining_capacity = 10000 - rm_forced_count
    selected = tele_rm_remaining[:remaining_capacity]
    rejected = tele_rm_remaining[remaining_capacity:]
    
    for idx in selected:
        allocations[idx, best_channels[idx]] = 1
        
    sms_emu = emu_matrix[:, 0]
    fallback_sms = rejected[sms_emu[rejected] > 0]
    allocations[fallback_sms, 0] = 1
    
    native_sms = np.where(valid & (best_channels == 0))[0]
    allocations[native_sms, 0] = 1
    
    # Budget Check
    total_cost = np.sum(allocations * channel_costs)
    if total_cost > 1_000_000_000:
        assigned = np.where(allocations.sum(axis=1) > 0)[0]
        can_drop = np.setdiff1d(assigned, forced_rm_idx, assume_unique=True)
        assigned_costs = allocations[can_drop] @ channel_costs
        assigned_emus = emu_matrix[can_drop, np.argmax(allocations[can_drop], axis=1)]
        efficiency = assigned_emus / assigned_costs
        
        sorted_drop_idx = can_drop[np.argsort(-efficiency)]
        current_cost = rm_forced_count * 2000000
        
        keep_list = list(forced_rm_idx)
        for idx in sorted_drop_idx:
            c = allocations[idx] @ channel_costs
            if current_cost + c <= 1_000_000_000:
                current_cost += c
                keep_list.append(idx)
            else:
                break
        final_alloc = np.zeros((N, 3))
        final_alloc[keep_list] = allocations[keep_list]
        allocations = final_alloc
        
    profit = np.sum(allocations * emu_matrix)
    rm_count = np.sum(allocations[:, 2])
    return profit, rm_count

rm_drops = [0.05, 0.10, 0.15, 0.20]
fp_incs = [0.10, 0.20, 0.30, 0.40]

results_profit = np.zeros((4, 4))
results_rm = np.zeros((4, 4))

print("Running 4x4 Heatmap Scenarios...")
for i, fp in enumerate(fp_incs):
    for j, cr in enumerate(rm_drops):
        prof, rm = run_scenario(cr, fp)
        results_profit[i, j] = prof
        results_rm[i, j] = rm

df_profit = pd.DataFrame(results_profit, index=[f"+{int(fp*100)}% FP", f"+{int(fp*100)}% FP", f"+{int(fp*100)}% FP", f"+{int(fp*100)}% FP"], columns=[f"-{int(cr*100)}% CR" for cr in rm_drops])
df_profit.index = [f"+10% FP", "+20% FP", "+30% FP", "+40% FP"]
df_rm = pd.DataFrame(results_rm, index=[f"+10% FP", "+20% FP", "+30% FP", "+40% FP"], columns=[f"-{int(cr*100)}% CR" for cr in rm_drops])

out_path = os.path.join(BASE_DIR, "heatmap_results.txt")
with open(out_path, "w", encoding="utf-8") as f:
    f.write("PROFIT MATRIX (VND):\n")
    f.write(frame_to_markdown(df_profit))
    f.write("\n\nRM SLOTS MATRIX:\n")
    f.write(frame_to_markdown(df_rm))

print("Done! Saved to heatmap_results.txt")
