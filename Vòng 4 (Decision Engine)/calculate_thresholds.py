import os

import numpy as np
import pandas as pd

from decision_config import FUM_MATRIX, VIP_SEGMENTS


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
    'RM': {'cost': 2_000_000, 'cr': 0.15},
}


def threshold_emu_func(p, cr, cost, tp_val, fn_val, fp_val):
    uplift = 4 * p * (1 - p) * cr
    fp_probability = 1 - p - uplift
    return uplift * (tp_val - fn_val) + fp_probability * fp_val - cost


print("Calculating Thresholds by Segment x Channel...")

results = []
ps = np.linspace(0, 1, 5000)

for segment, economics in FUM_MATRIX.items():
    thresholds = {}
    for ch_name, ch_data in channels.items():
        if ch_name == 'RM' and segment not in VIP_SEGMENTS:
            thresholds[ch_name] = 'N/A'
            continue

        emus = threshold_emu_func(
            ps,
            ch_data['cr'],
            ch_data['cost'],
            economics['TP'],
            economics['FN'],
            economics['FP'],
        )
        valid_ps = ps[emus >= 0]
        thresholds[ch_name] = float(valid_ps[0]) if len(valid_ps) > 0 else "No ROI"

    results.append({
        'Segment': segment,
        'Threshold SMS': thresholds['SMS'],
        'Threshold Telesales': thresholds['Telesales'],
        'Threshold RM': thresholds['RM'],
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

print(f"Saved thresholds to {csv_path} and {out_path}")
