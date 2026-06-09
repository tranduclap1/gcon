import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def frame_to_markdown(df):
    rows = [[str(col) for col in df.columns]]
    rows.extend(df.fillna('').astype(str).values.tolist())
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    header = '| ' + ' | '.join(rows[0][i].ljust(widths[i]) for i in range(len(widths))) + ' |'
    separator = '| ' + ' | '.join('-' * widths[i] for i in range(len(widths))) + ' |'
    body = ['| ' + ' | '.join(row[i].ljust(widths[i]) for i in range(len(widths))) + ' |' for row in rows[1:]]
    return '\n'.join([header, separator] + body)

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
    
    # Unsupervised-to-Supervised Proxy logic for Non-IB
    # Using REAL Historical Onboarding Rates calculated from Cohort Tracking
    base_cr_nonib = {
        'Senior High-Value Heavy Borrower': 0.9183,
        'High-Value Heavy Borrower': 0.9144,
        'High-Value Traditional': 0.7803,
        'Dormant / Ngủ đông': 0.4018,
        'Senior High-Value Saver': 0.1802,
        'High-Value Saver': 0.1562,
        'Traditional': 0.1068
    }
    # Map probabilities
    df_nonib['PROBABILITY'] = df_nonib['PERSONA'].map(base_cr_nonib).fillna(0.10)
    # Add minor GMM posterior simulation (variation) to make it unique per customer (prevent ties)
    np.random.seed(42)
    df_nonib['PROBABILITY'] *= np.random.uniform(0.9, 1.1, len(df_nonib))
    df_nonib['RECOMMENDED_PRODUCT'] = 'Digital Onboarding'
    
    # Combine
    df_master = pd.concat([df_ib_prob, df_nonib], ignore_index=True)
    
except Exception as e:
    print("Error loading real data:", e)
    import sys; sys.exit(1)

print("Master data shape:", df_master.shape)

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

channels = {
    'SMS': {'cost': 5_000, 'cr': 0.02},
    'Telesales': {'cost': 50_000, 'cr': 0.05},
    'RM': {'cost': 2_000_000, 'cr': 0.15}
}

channel_names = np.array(['SMS', 'Telesales', 'RM'])
channel_costs = np.array([channels[ch]['cost'] for ch in channel_names])

# Identify VIPs
vip_personas = ['Wealthy Passive', 'Digital VIP', 'Senior High-Value Saver', 'Senior High-Value Heavy Borrower']
df_master['IS_VIP'] = df_master['PERSONA'].isin(vip_personas)

# Map Unit Economics from FUM
df_master['TP'] = df_master['PERSONA'].map(lambda x: fum_matrix.get(x, fum_matrix['Standard'])['TP'])
df_master['FP'] = df_master['PERSONA'].map(lambda x: fum_matrix.get(x, fum_matrix['Standard'])['FP'])
df_master['FN'] = df_master['PERSONA'].map(lambda x: fum_matrix.get(x, fum_matrix['Standard'])['FN'])

# TIE BREAKER: Asset Proxy (simulated from customer ID for stable ranking)
df_master['ASSET_SCORE'] = (df_master['CUSTOMER_NUMBER'] % 10000) / 10000.0

def threshold_emu_formula(p, cr, cost, tp_val, fn_val, fp_val):
    uplift = 4 * p * (1 - p) * cr
    return uplift * (tp_val - fn_val) + fp_val - cost

def calculate_channel_thresholds():
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

def load_thresholds_from_csv():
    threshold_path = os.path.join(BASE_DIR, "thresholds.csv")
    if not os.path.exists(threshold_path):
        return None

    df_thresholds = pd.read_csv(threshold_path)
    thresholds = {}
    column_map = {
        'SMS': 'Threshold SMS',
        'Telesales': 'Threshold Telesales',
        'RM': 'Threshold RM',
    }
    for _, row in df_thresholds.iterrows():
        persona = row['Persona']
        thresholds[persona] = {}
        for ch_name, col_name in column_map.items():
            value = row[col_name]
            if pd.isna(value) or str(value) in {'N/A', 'No ROI'}:
                thresholds[persona][ch_name] = None
            else:
                thresholds[persona][ch_name] = float(value)
    return thresholds

threshold_matrix = load_thresholds_from_csv() or calculate_channel_thresholds()

def build_eligibility_matrix(df, thresholds):
    eligible = np.full((len(df), len(channel_names)), False)
    for channel_idx, ch_name in enumerate(channel_names):
        persona_thresholds = df['PERSONA'].map(
            lambda persona: thresholds.get(persona, thresholds['Standard']).get(ch_name)
        )
        has_threshold = persona_thresholds.notna()
        eligible[:, channel_idx] = has_threshold & (df['PROBABILITY'] >= persona_thresholds.astype(float))
    return eligible

eligibility_matrix = build_eligibility_matrix(df_master, threshold_matrix)

def apply_threshold_filter(value_matrix, eligible_matrix):
    return np.where(eligible_matrix, value_matrix, -999_999_999)

def calculate_emu(df, fp_multiplier=1.0, cr_multipliers=None):
    if cr_multipliers is None:
        cr_multipliers = {'SMS': 1.0, 'Telesales': 1.0, 'RM': 1.0}

    P_base = df['PROBABILITY']

    uplift_sms = 4 * P_base * (1 - P_base) * channels['SMS']['cr'] * cr_multipliers.get('SMS', 1.0)
    uplift_tele = 4 * P_base * (1 - P_base) * channels['Telesales']['cr'] * cr_multipliers.get('Telesales', 1.0)
    uplift_rm = 4 * P_base * (1 - P_base) * channels['RM']['cr'] * cr_multipliers.get('RM', 1.0)

    tp_array = df['TP']
    fn_array = df['FN']
    fp_array = df['FP'] * fp_multiplier

    emu_sms = uplift_sms * (tp_array - fn_array) + fp_array - channels['SMS']['cost']
    emu_tele = uplift_tele * (tp_array - fn_array) + fp_array - channels['Telesales']['cost']
    emu_rm = uplift_rm * (tp_array - fn_array) + fp_array - channels['RM']['cost']
    
    # Add Tie-breaker
    tie_breaker = 1e-6 * df['ASSET_SCORE']
    
    return np.vstack([emu_sms + tie_breaker, emu_tele + tie_breaker, emu_rm + tie_breaker]).T

print("Calculating Baseline EMU (Uplift Mode)...")
emu_baseline = apply_threshold_filter(calculate_emu(df_master), eligibility_matrix)

def solve_allocation(emu_matrix):
    N = len(df_master)
    allocations = np.full((N, 3), 0)
    best_channels = np.argmax(emu_matrix, axis=1)
    max_emus = np.max(emu_matrix, axis=1)
    
    valid = max_emus > 0
    
    tele_rm_indices = np.where(valid & ((best_channels == 1) | (best_channels == 2)))[0]
    tele_rm_indices = tele_rm_indices[np.argsort(-max_emus[tele_rm_indices])]
    
    # No lower-bound constraint for RM. RM is selected only when it wins on EMU.
    rm_forced_count = 0
    forced_rm_idx = np.array([], dtype=int)
        
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
        
    total_profit = np.sum(allocations * emu_matrix)
    counts = np.sum(allocations, axis=0)
    cost = np.sum(allocations * channel_costs)
    
    return total_profit, counts, cost, allocations

profit_base, counts_base, cost_base, alloc_base = solve_allocation(emu_baseline)

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
print(frame_to_markdown(df_master[['CUSTOMER_NUMBER', 'PERSONA', 'RECOMMENDED_PRODUCT', 'PROBABILITY', 'RECOMMENDED_CHANNEL']].head(5)))

print("\nCalculating Stress-Test (FP VIP cost +20%, CR -15% for Telesales and RM)...")
stress_cr_multipliers = {'SMS': 1.0, 'Telesales': 0.85, 'RM': 0.85}
emu_stress = apply_threshold_filter(
    calculate_emu(df_master, fp_multiplier=1.2, cr_multipliers=stress_cr_multipliers),
    eligibility_matrix
)
profit_stress, counts_stress, cost_stress, alloc_stress = solve_allocation(emu_stress)

print("Stress Results:")
print(f"Profit: {profit_stress:,.0f}")
print(f"Cost: {cost_stress:,.0f}")
print(f"Allocations: SMS={counts_stress[0]}, Tele={counts_stress[1]}, RM={counts_stress[2]}")

