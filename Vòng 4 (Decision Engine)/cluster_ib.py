import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

print("Loading IB customer data...")
in_path = os.path.join(BASE_DIR, "NBFO_IB", "processed_data", "gcon_customer_month_clean.parquet")
df = pd.read_parquet(in_path)

# Aggregate to customer level (take latest month)
df_cust = df.sort_values('MONTH').groupby('CUSTOMER_NUMBER').last().reset_index()

# Select features for clustering
features = ['AVG_CA_BALANCE', 'AVG_TD_BALANCE', 'TRANS_AMOUNT_SUM', 'TRANS_NO_SUM', 'AGE']
X = df_cust[features].copy()

# Impute and scale
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)

# We use log transform for heavy tailed financial features
X_log = np.log1p(np.maximum(X_imputed, 0))

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_log)

print("Running GMM (K=5)...")
gmm = GaussianMixture(n_components=5, random_state=42, n_init=3)
clusters = gmm.fit_predict(X_scaled)
df_cust['CLUSTER'] = clusters

print("\nCluster Profiles (Medians):")
profile = df_cust.groupby('CLUSTER')[features].median()
print(profile)

# Name personas based on profile
# 0, 1, 2, 3, 4
# Let's map dynamically based on logic:
# Highest TD Balance -> Wealthy Passive
# Highest Trans Amount -> Digital VIP
# Lowest Age -> Young Digital
# etc.

# For simplicity, we just assign the names we used in the report directly to the clusters based on their stats
td_ranks = profile['AVG_TD_BALANCE'].rank()
trans_ranks = profile['TRANS_AMOUNT_SUM'].rank()

# Define a mapping logic based on actual cluster results
persona_map = {}
for c in range(5):
    if profile.loc[c, 'AVG_TD_BALANCE'] == profile['AVG_TD_BALANCE'].max():
        persona_map[c] = 'Wealthy Passive'
    elif profile.loc[c, 'TRANS_AMOUNT_SUM'] == profile['TRANS_AMOUNT_SUM'].max():
        persona_map[c] = 'Digital VIP'
    elif profile.loc[c, 'AGE'] == profile['AGE'].min():
        persona_map[c] = 'Young Digital'
    elif c not in persona_map:
        persona_map[c] = 'Mass Active' if len(persona_map) % 2 == 0 else 'Standard'

# Ensure all 5 have distinct names if logic overlaps
used_names = list(persona_map.values())
default_names = ['Wealthy Passive', 'Digital VIP', 'Young Digital', 'Mass Active', 'Standard']
for c in range(5):
    if c not in persona_map:
        for name in default_names:
            if name not in used_names:
                persona_map[c] = name
                used_names.append(name)
                break

df_cust['PERSONA'] = df_cust['CLUSTER'].map(persona_map)

print("\nPersona mapping:")
for c, name in persona_map.items():
    print(f"Cluster {c} -> {name}")

out_path = os.path.join(BASE_DIR, "NBFO_IB", "processed_data", "ib_final_personas.parquet")
df_cust[['CUSTOMER_NUMBER', 'CLUSTER', 'PERSONA']].to_parquet(out_path)
print(f"\nSaved IB personas to {out_path}")
