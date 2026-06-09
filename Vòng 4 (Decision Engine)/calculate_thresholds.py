import pandas as pd
import numpy as np
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
    'Senior High-Value Saver':          {'TP': 2_060_000, 'FP': -10_000, 'FN': -5_000_000},
    'Traditional':                      {'TP': 2_060_000, 'FP': -5_000,  'FN': 0},
    'Dormant / Ngủ đông':               {'TP': 2_060_000, 'FP': -2_000,  'FN': 0},
    'High-Value Saver':                 {'TP': 2_060_000, 'FP': -15_000, 'FN': -1_000_000},
    'High-Value Heavy Borrower':        {'TP': 2_060_000, 'FP': -15_000, 'FN': -2_000_000},
    'Senior High-Value Heavy Borrower': {'TP': 2_060_000, 'FP': -10_000, 'FN': -4_000_000},
    'High-Value Traditional':           {'TP': 2_060_000, 'FP': -8_000,  'FN': -500_000},
}

vip_personas = ['Wealthy Passive', 'Digital VIP', 'Senior High-Value Saver', 'Senior High-Value Heavy Borrower']

print("Calculating Thresholds by Persona x Channel...")

def threshold_emu_func(p, cr, cost, tp_val, fn_val, fp_val):
    uplift = 4 * p * (1 - p) * cr
    return uplift * (tp_val - fn_val) + fp_val - cost

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
            return threshold_emu_func(p, ch_data['cr'], ch_data['cost'], tp_val, fn_val, fp_val)
        
        ps = np.linspace(0, 1, 5000)
        emus = target(ps)
        
        valid_ps = ps[emus >= 0]
        if len(valid_ps) > 0:
            thresholds[ch_name] = float(valid_ps[0])
        else:
            thresholds[ch_name] = "No ROI"
            
    results.append({
        'Persona': persona,
        'Threshold SMS': thresholds['SMS'],
        'Threshold Telesales': thresholds['Telesales'],
        'Threshold RM': thresholds.get('RM', 'N/A')
    })

df_thresh = pd.DataFrame(results)
df_thresh_display = df_thresh.copy()
for col in ['Threshold SMS', 'Threshold Telesales', 'Threshold RM']:
    df_thresh_display[col] = df_thresh_display[col].map(
        lambda value: f"{value:.4f}" if isinstance(value, float) else value
    )

out_path = os.path.join(BASE_DIR, "thresholds.md")
with open(out_path, "w", encoding="utf-8") as f:
    f.write(frame_to_markdown(df_thresh_display))

csv_path = os.path.join(BASE_DIR, "thresholds.csv")
df_thresh.to_csv(csv_path, index=False)
