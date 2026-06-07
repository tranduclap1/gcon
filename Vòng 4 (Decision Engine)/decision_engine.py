import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

print("Loading data...")
try:
    # 1. Load probabilities (IB) - Keeping Product Dimension
    path_ib_prob = os.path.join(BASE_DIR, "NBFO_IB", "saved_models", "gcon_test_scores_best_xgboost_calibrated_sigmoid.parquet")
    df_ib = pd.read_parquet(path_ib_prob)
    # Get the row with the max probability for each customer (Next Best Product)
    idx_max_prob = df_ib.groupby('CUSTOMER_NUMBER')['SUBSCRIPTION_PROPENSITY'].idxmax()
    df_ib_prob = df_ib.loc[idx_max_prob].copy()
    df_ib_prob = df_ib_prob[['CUSTOMER_NUMBER', 'PRODUCT_NAME', 'SUBSCRIPTION_PROPENSITY']]
    df_ib_prob.rename(columns={'SUBSCRIPTION_PROPENSITY': 'PROBABILITY', 'PRODUCT_NAME': 'RECOMMENDED_PRODUCT'}, inplace=True)
    
    # Load real IB Personas
    path_ib_personas = os.path.join(BASE_DIR, "NBFO_IB", "processed_data", "ib_final_personas.parquet")
    df_ib_personas = pd.read_parquet(path_ib_personas)
    df_ib_prob = df_ib_prob.merge(df_ib_personas[['CUSTOMER_NUMBER', 'PERSONA']], on='CUSTOMER_NUMBER', how='left')
    df_ib_prob['PERSONA'].fillna('Standard', inplace=True)
    
    # 2. Load Non-IB
    path_nonib = os.path.join(BASE_DIR, "Cluster_nonIB", "output", "nonib_final_personas.parquet")
    df_nonib = pd.read_parquet(path_nonib)
    df_nonib = df_nonib[['CUSTOMER_NUMBER', 'PERSONA_NAME']].copy()
    df_nonib.rename(columns={'PERSONA_NAME': 'PERSONA'}, inplace=True)
    
    # Unsupervised-to-Supervised Proxy logic for Non-IB
    base_cr_nonib = {
        'Senior High-Value Saver': 0.05,
        'Traditional': 0.01,
        'Young Active': 0.08,
        'Digital Native': 0.06
    }
    # Map, default to 0.02
    df_nonib['PROBABILITY'] = df_nonib['PERSONA'].map(base_cr_nonib).fillna(0.02)
    # Add GMM posterior simulation (variation) to make it unique per customer
    np.random.seed(42)
    df_nonib['PROBABILITY'] *= np.random.uniform(0.8, 1.2, len(df_nonib))
    df_nonib['RECOMMENDED_PRODUCT'] = 'Digital Onboarding'
    
    # Combine
    df_master = pd.concat([df_ib_prob, df_nonib], ignore_index=True)
    
except Exception as e:
    print("Error loading real data:", e)
    import sys; sys.exit(1)

print("Master data shape:", df_master.shape)

# Unit Economics
TP = 5_000_000
FP = -50_000
FN_VIP = -30_000_000

channels = {
    'SMS': {'cost': 5_000, 'cr': 0.02},
    'Telesales': {'cost': 50_000, 'cr': 0.05},
    'RM': {'cost': 2_000_000, 'cr': 0.15}
}

# Identify VIPs
vip_personas = ['Wealthy Passive', 'Digital VIP', 'Senior High-Value Saver']
df_master['IS_VIP'] = df_master['PERSONA'].isin(vip_personas)

# TIE BREAKER: Asset Proxy (simulated from customer ID for stable ranking)
df_master['ASSET_SCORE'] = (df_master['CUSTOMER_NUMBER'] % 10000) / 10000.0

def calculate_emu(df, fp_multiplier=1.0, cr_multiplier=1.0):
    P_base = df['PROBABILITY']
    
    # UPLIFT MODELING: P_final = P_base + Uplift
    # Persuadable Curve: Uplift is max at P=0.5, 0 at P=0 and P=1
    # Uplift(P) = 4 * P * (1-P) * CR_channel
    uplift_sms = 4 * P_base * (1 - P_base) * channels['SMS']['cr'] * cr_multiplier
    uplift_tele = 4 * P_base * (1 - P_base) * channels['Telesales']['cr'] * cr_multiplier
    uplift_rm = 4 * P_base * (1 - P_base) * channels['RM']['cr'] * cr_multiplier
    
    # Dynamic FN (Only VIPs suffer -30M if we miss uplifting them)
    fn_array = np.where(df['IS_VIP'], FN_VIP, 0)
    fp_array = np.where(df['IS_VIP'], FP * fp_multiplier, FP)
    
    # EMU = Uplift * (TP - FN) + (1 - P_base - Uplift) * FP - Cost
    emu_sms = uplift_sms * (TP - fn_array) + (1 - P_base - uplift_sms) * fp_array - channels['SMS']['cost']
    emu_tele = uplift_tele * (TP - fn_array) + (1 - P_base - uplift_tele) * fp_array - channels['Telesales']['cost']
    emu_rm = uplift_rm * (TP - fn_array) + (1 - P_base - uplift_rm) * fp_array - channels['RM']['cost']
    
    # RM is only for VIPs
    emu_rm = np.where(df['IS_VIP'], emu_rm, -999_999_999)
    
    # Add Tie-breaker
    tie_breaker = 1e-6 * df['ASSET_SCORE']
    
    return np.vstack([emu_sms + tie_breaker, emu_tele + tie_breaker, emu_rm + tie_breaker]).T

print("Calculating Baseline EMU (Uplift Mode)...")
emu_baseline = calculate_emu(df_master)

def solve_allocation_with_sunkcost(emu_matrix):
    N = len(df_master)
    allocations = np.full((N, 3), 0)
    best_channels = np.argmax(emu_matrix, axis=1)
    max_emus = np.max(emu_matrix, axis=1)
    
    valid = max_emus > 0
    
    tele_rm_indices = np.where(valid & ((best_channels == 1) | (best_channels == 2)))[0]
    tele_rm_indices = tele_rm_indices[np.argsort(-max_emus[tele_rm_indices])]
    
    # --- SUNK COST LOWER BOUND CONSTRAINT (RM >= 100) ---
    # Force allocate top 100 RM even if we have to displace some Tele
    rm_valid_indices = np.where(valid & (best_channels == 2))[0]
    rm_valid_indices = rm_valid_indices[np.argsort(-max_emus[rm_valid_indices])]
    
    rm_forced_count = min(100, len(rm_valid_indices))
    forced_rm_idx = rm_valid_indices[:rm_forced_count]
    
    for idx in forced_rm_idx:
        allocations[idx, 2] = 1
        
    # Remove forced from the general pool
    tele_rm_remaining = np.setdiff1d(tele_rm_indices, forced_rm_idx, assume_unique=True)
    # Sort remaining
    tele_rm_remaining = tele_rm_remaining[np.argsort(-max_emus[tele_rm_remaining])]
    
    # Fill up to 10k capacity
    remaining_capacity = 10000 - rm_forced_count
    selected = tele_rm_remaining[:remaining_capacity]
    rejected = tele_rm_remaining[remaining_capacity:]
    
    for idx in selected:
        allocations[idx, best_channels[idx]] = 1
        
    # Rejected fallback to SMS
    sms_emu = emu_matrix[:, 0]
    fallback_sms = rejected[sms_emu[rejected] > 0]
    allocations[fallback_sms, 0] = 1
    
    # Assign native SMS
    native_sms = np.where(valid & (best_channels == 0))[0]
    # Do not overwrite if already forced (though disjoint logically)
    allocations[native_sms, 0] = 1
    
    # Check budget
    total_cost = np.sum(allocations * np.array([5000, 50000, 2000000]))
    if total_cost > 1_000_000_000:
        assigned = np.where(allocations.sum(axis=1) > 0)[0]
        # Never drop forced RM to maintain Sunk Cost Constraint
        can_drop = np.setdiff1d(assigned, forced_rm_idx, assume_unique=True)
        
        assigned_costs = allocations[can_drop] @ np.array([5000, 50000, 2000000])
        assigned_emus = emu_matrix[can_drop, np.argmax(allocations[can_drop], axis=1)]
        efficiency = assigned_emus / assigned_costs
        
        sorted_drop_idx = can_drop[np.argsort(-efficiency)]
        # We start with cost of forced RM
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
        
    total_profit = np.sum(allocations * emu_matrix)
    counts = np.sum(allocations, axis=0)
    cost = np.sum(allocations * np.array([5000, 50000, 2000000]))
    
    return total_profit, counts, cost, allocations

profit_base, counts_base, cost_base, alloc_base = solve_allocation_with_sunkcost(emu_baseline)

# Assign channels based on the allocations matrix
channel_names = np.array(['SMS', 'Telesales', 'RM'])
assigned_indices = np.where(alloc_base == 1)
df_master['RECOMMENDED_CHANNEL'] = 'None'
df_master.loc[assigned_indices[0], 'RECOMMENDED_CHANNEL'] = channel_names[assigned_indices[1]]

# Save output
out_alloc = os.path.join(BASE_DIR, "final_allocations.csv")
df_master[['CUSTOMER_NUMBER', 'PERSONA', 'RECOMMENDED_PRODUCT', 'PROBABILITY', 'RECOMMENDED_CHANNEL']].to_csv(out_alloc, index=False)

print("Baseline Results:")
print(f"Profit: {profit_base:,.0f}")
print(f"Cost: {cost_base:,.0f}")
print(f"Allocations: SMS={counts_base[0]}, Tele={counts_base[1]}, RM={counts_base[2]}")

print("\nSample Allocation Output:")
print(df_master[['CUSTOMER_NUMBER', 'PERSONA', 'RECOMMENDED_PRODUCT', 'PROBABILITY', 'RECOMMENDED_CHANNEL']].head(5).to_markdown(index=False))

print("\nCalculating Stress-Test (FP VIP cost +20%, CR -15%)...")
emu_stress = calculate_emu(df_master, fp_multiplier=1.2, cr_multiplier=0.85)
profit_stress, counts_stress, cost_stress, alloc_stress = solve_allocation_with_sunkcost(emu_stress)

print("Stress Results:")
print(f"Profit: {profit_stress:,.0f}")
print(f"Cost: {cost_stress:,.0f}")
print(f"Allocations: SMS={counts_stress[0]}, Tele={counts_stress[1]}, RM={counts_stress[2]}")

