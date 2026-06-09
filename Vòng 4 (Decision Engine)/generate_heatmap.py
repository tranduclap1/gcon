import pandas as pd
import numpy as np

import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

print("Loading data for Heatmap generation...")
# We will simulate the dataframe logic here to quickly run 16 scenarios
# 1. Load probabilities (IB)
path_ib_prob = os.path.join(BASE_DIR, "NBFO_IB", "saved_models", "gcon_test_scores_best_xgboost_calibrated_sigmoid.parquet")
df_ib = pd.read_parquet(path_ib_prob)
df_ib_prob = df_ib.groupby('CUSTOMER_NUMBER')['SUBSCRIPTION_PROPENSITY'].max().reset_index()
df_ib_prob.rename(columns={'SUBSCRIPTION_PROPENSITY': 'PROBABILITY'}, inplace=True)

path_ib_personas = os.path.join(BASE_DIR, "NBFO_IB", "processed_data", "ib_final_personas.parquet")
df_ib_personas = pd.read_parquet(path_ib_personas)
df_ib_prob = df_ib_prob.merge(df_ib_personas[['CUSTOMER_NUMBER', 'PERSONA']], on='CUSTOMER_NUMBER', how='left')
df_ib_prob['PERSONA'].fillna('Standard', inplace=True)

# 2. Load Non-IB
path_nonib = os.path.join(BASE_DIR, "Cluster_nonIB", "output", "nonib_final_personas.parquet")
df_nonib = pd.read_parquet(path_nonib)
df_nonib = df_nonib[['CUSTOMER_NUMBER', 'PERSONA_NAME']].copy()
df_nonib.rename(columns={'PERSONA_NAME': 'PERSONA'}, inplace=True)

base_cr_nonib = {
    'Senior High-Value Saver': 0.05,
    'Traditional': 0.01,
    'Dormant / Ngủ đông': 0.005,
    'High-Value Saver': 0.04,
    'High-Value Heavy Borrower': 0.06,
    'High-Value Traditional': 0.03,
    'Senior High-Value Heavy Borrower': 0.07
}
df_nonib['PROBABILITY'] = df_nonib['PERSONA'].map(base_cr_nonib).fillna(0.01)
np.random.seed(42)
df_nonib['PROBABILITY'] *= np.random.uniform(0.8, 1.2, len(df_nonib))

df_master = pd.concat([df_ib_prob, df_nonib], ignore_index=True)

# Unit Economics
TP = 5_000_000
FP = -50_000
FN_VIP = -30_000_000

vip_personas = ['Wealthy Passive', 'Digital VIP', 'Senior High-Value Saver', 'Senior High-Value Heavy Borrower']
df_master['IS_VIP'] = df_master['PERSONA'].isin(vip_personas)
df_master['ASSET_SCORE'] = (df_master['CUSTOMER_NUMBER'] % 10000) / 10000.0

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
    
    fn_array = np.where(df_master['IS_VIP'], FN_VIP, 0)
    
    # Apply VIP FP Increase
    fp_array = np.where(df_master['IS_VIP'], FP * (1 + vip_fp_inc), FP)
    
    emu_sms = uplift_sms * (TP - fn_array) + (1 - P_base - uplift_sms) * fp_array - channels['SMS']['cost']
    emu_tele = uplift_tele * (TP - fn_array) + (1 - P_base - uplift_tele) * fp_array - channels['Telesales']['cost']
    emu_rm = uplift_rm * (TP - fn_array) + (1 - P_base - uplift_rm) * fp_array - channels['RM']['cost']
    
    emu_rm = np.where(df_master['IS_VIP'], emu_rm, -999_999_999)
    
    tie_breaker = 1e-6 * df_master['ASSET_SCORE']
    emu_matrix = np.vstack([emu_sms + tie_breaker, emu_tele + tie_breaker, emu_rm + tie_breaker]).T
    
    # Fast Greedy Solve
    N = len(df_master)
    allocations = np.full((N, 3), 0)
    best_channels = np.argmax(emu_matrix, axis=1)
    max_emus = np.max(emu_matrix, axis=1)
    
    valid = max_emus > 0
    
    # Sunk Cost RM >= 100
    rm_valid_indices = np.where(valid & (best_channels == 2))[0]
    rm_valid_indices = rm_valid_indices[np.argsort(-max_emus[rm_valid_indices])]
    rm_forced_count = min(100, len(rm_valid_indices))
    forced_rm_idx = rm_valid_indices[:rm_forced_count]
    
    for idx in forced_rm_idx:
        allocations[idx, 2] = 1
        
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
    total_cost = np.sum(allocations * np.array([5000, 50000, 2000000]))
    if total_cost > 1_000_000_000:
        assigned = np.where(allocations.sum(axis=1) > 0)[0]
        can_drop = np.setdiff1d(assigned, forced_rm_idx, assume_unique=True)
        assigned_costs = allocations[can_drop] @ np.array([5000, 50000, 2000000])
        assigned_emus = emu_matrix[can_drop, np.argmax(allocations[can_drop], axis=1)]
        efficiency = assigned_emus / assigned_costs
        
        sorted_drop_idx = can_drop[np.argsort(-efficiency)]
        current_cost = rm_forced_count * 2000000
        
        keep_list = list(forced_rm_idx)
        for idx in sorted_drop_idx:
            c = allocations[idx] @ np.array([5000, 50000, 2000000])
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
    f.write(df_profit.to_markdown())
    f.write("\n\nRM SLOTS MATRIX:\n")
    f.write(df_rm.to_markdown())

print("Done! Saved to heatmap_results.txt")
