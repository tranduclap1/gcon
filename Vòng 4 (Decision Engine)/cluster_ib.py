import os

import pandas as pd

from decision_config import add_ib_segments, attach_ib_register_date


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

print("Loading IB customer data...")
model_input_path = os.path.join(BASE_DIR, "NBFO_IB", "processed_data", "gcon_model_input.parquet")
df = pd.read_parquet(model_input_path)

if 'MONTH' in df.columns:
    df_cust = df.sort_values('MONTH').groupby('CUSTOMER_NUMBER').last().reset_index()
else:
    df_cust = df.groupby('CUSTOMER_NUMBER').last().reset_index()

df_cust = attach_ib_register_date(df_cust, BASE_DIR)
df_cust = add_ib_segments(df_cust)
register_year = pd.to_datetime(df_cust['IB_REGISTER_DATE'], errors='coerce').dt.year
df_cust['IB_COHORT'] = 'UNKNOWN'
df_cust.loc[register_year <= 2019, 'IB_COHORT'] = '2019'
df_cust.loc[register_year.isin([2020, 2021]), 'IB_COHORT'] = '2020/21'

print("\nRule-based IB segment distribution:")
print(df_cust['SEGMENT'].value_counts().to_string())
print("\nRule-based IB segment distribution by cohort:")
print(pd.crosstab(df_cust['IB_COHORT'], df_cust['SEGMENT']).to_string())

out_path = os.path.join(BASE_DIR, "NBFO_IB", "processed_data", "ib_final_personas.parquet")
df_cust[['CUSTOMER_NUMBER', 'SEGMENT', 'MAPPED_IB_SEGMENT', 'CUSTOMER_TYPE', 'IB_COHORT']].to_parquet(out_path)
print(f"\nSaved IB segments to {out_path}")
