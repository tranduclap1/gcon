import pandas as pd
import numpy as np
from scipy.optimize import fsolve

# Unit Economics
TP = 5_000_000
FP = -50_000
FN_VIP = -30_000_000
FN_NORM = 0

channels = {
    'SMS': {'cost': 5_000, 'cr': 0.02},
    'Telesales': {'cost': 50_000, 'cr': 0.05},
    'RM': {'cost': 2_000_000, 'cr': 0.15}
}

vip_personas = ['Wealthy Passive', 'Digital VIP', 'Senior High-Value Saver']
all_personas = [
    'Wealthy Passive', 'Digital VIP', 'Mass Active', 'Young Digital', 'Standard',
    'Senior High-Value Saver', 'Traditional', 'Young Active', 'Digital Native', 
    'Low-Income Earner', 'Student', 'Inactive', 'Unknown' # Adjust based on actual non-IB personas
]

print("Calculating Thresholds...")

def emu_func(p, cr, cost, fn_val, fp_val):
    uplift = 4 * p * (1 - p) * cr
    return uplift * (TP - fn_val) + (1 - p - uplift) * fp_val - cost

results = []
for persona in all_personas:
    is_vip = persona in vip_personas
    fn_val = FN_VIP if is_vip else FN_NORM
    fp_val = FP
    
    thresholds = {}
    for ch_name, ch_data in channels.items():
        if ch_name == 'RM' and not is_vip:
            thresholds[ch_name] = 'N/A'
            continue
            
        # We want to find the lowest P in [0, 1] where EMU >= 0.
        # Since EMU is a parabola opening downwards (due to -P^2 in uplift), it might have 2 roots.
        # The threshold is the lower root.
        def target(p):
            return emu_func(p, ch_data['cr'], ch_data['cost'], fn_val, fp_val)
        
        # Test values from 0 to 1
        ps = np.linspace(0, 1, 1000)
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
print(df_thresh.to_markdown(index=False))

import os
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

with open(os.path.join(BASE_DIR, "thresholds.md"), "w", encoding="utf-8") as f:
    f.write(df_thresh.to_markdown(index=False))
