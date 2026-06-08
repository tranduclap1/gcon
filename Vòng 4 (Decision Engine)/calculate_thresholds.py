import pandas as pd
import numpy as np
from scipy.optimize import fsolve
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

channels = {
    'SMS': {'cost': 5_000, 'cr': 0.02},
    'Telesales': {'cost': 50_000, 'cr': 0.05},
    'RM': {'cost': 2_000_000, 'cr': 0.15}
}

# Unique Financial Utility Matrix (FUM) for each Persona
fum_matrix = {
    # IB Personas (Cross-sell: TP ~ 5M)
    'Wealthy Passive': {'TP': 5_000_000, 'FP': -50_000, 'FN': -30_000_000},
    'Digital VIP':     {'TP': 5_000_000, 'FP': -80_000, 'FN': -20_000_000},
    'Mass Active':     {'TP': 5_000_000, 'FP': -20_000, 'FN': 0},
    'Young Digital':   {'TP': 5_000_000, 'FP': -100_000, 'FN': 0}, # Spam sensitive
    'Standard':        {'TP': 5_000_000, 'FP': -40_000, 'FN': 0},
    
    # Non-IB Personas (Onboarding: TP ~ 1M)
    'Senior High-Value Saver':          {'TP': 1_000_000, 'FP': -10_000, 'FN': -5_000_000},
    'Traditional':                      {'TP': 1_000_000, 'FP': -5_000,  'FN': 0},
    'Dormant / Ngủ đông':               {'TP': 1_000_000, 'FP': -2_000,  'FN': 0},
    'High-Value Saver':                 {'TP': 1_000_000, 'FP': -15_000, 'FN': -1_000_000},
    'High-Value Heavy Borrower':        {'TP': 1_000_000, 'FP': -15_000, 'FN': -2_000_000},
    'Senior High-Value Heavy Borrower': {'TP': 1_000_000, 'FP': -10_000, 'FN': -4_000_000},
    'High-Value Traditional':           {'TP': 1_000_000, 'FP': -8_000,  'FN': -500_000},
}

vip_personas = ['Wealthy Passive', 'Digital VIP', 'Senior High-Value Saver', 'Senior High-Value Heavy Borrower']

print("Calculating Thresholds by Persona x Channel...")

def emu_func(p, cr, cost, fn_val, fp_val, tp_val):
    uplift = 4 * p * (1 - p) * cr
    return uplift * (tp_val - fn_val) + (1 - p - uplift) * fp_val - cost

results = []
for persona, economics in fum_matrix.items():
    is_vip = persona in vip_personas
    tp_val = economics['TP']
    fp_val = economics['FP']
    fn_val = economics['FN']
    
    thresholds = {}
    for ch_name, ch_data in channels.items():
        if ch_name == 'RM' and not is_vip:
            thresholds[ch_name] = 'N/A'
            continue
            
        def target(p):
            return emu_func(p, ch_data['cr'], ch_data['cost'], fn_val, fp_val, tp_val)
        
        ps = np.linspace(0, 1, 5000)
        emus = target(ps)
        
        valid_ps = ps[emus >= 0]
        if len(valid_ps) > 0:
            thresholds[ch_name] = f"{valid_ps[0]:.4f}"
        else:
            thresholds[ch_name] = "No ROI"
            
    results.append({
        'Persona': persona,
        'Threshold SMS': thresholds['SMS'],
        'Threshold Telesales': thresholds['Telesales'],
        'Threshold RM': thresholds.get('RM', 'N/A')
    })

df_thresh = pd.DataFrame(results)

out_path = os.path.join(BASE_DIR, "thresholds.md")
with open(out_path, "w", encoding="utf-8") as f:
    f.write(df_thresh.to_markdown(index=False))
