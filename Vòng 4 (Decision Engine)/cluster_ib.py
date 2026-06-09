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
clean_path = os.path.join(BASE_DIR, "NBFO_IB", "processed_data", "gcon_customer_month_clean.parquet")
model_input_path = os.path.join(BASE_DIR, "NBFO_IB", "processed_data", "gcon_model_input.parquet")
in_path = clean_path if os.path.exists(clean_path) else model_input_path
print(f"Using input: {in_path}")
df = pd.read_parquet(in_path)

# Aggregate to customer level (take latest month)
df_cust = df.sort_values('MONTH').groupby('CUSTOMER_NUMBER').last().reset_index()

# Select features for clustering
age_col = 'AGE' if 'AGE' in df_cust.columns else 'AGE_CLEAN'
features = ['AVG_CA_BALANCE', 'AVG_TD_BALANCE', 'TRANS_AMOUNT_SUM', 'TRANS_NO_SUM', age_col]
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
# Define a deterministic one-to-one mapping from clusters to the 5 IB personas.
persona_map = {}
remaining = set(profile.index)

wealth_cluster = profile['AVG_TD_BALANCE'].idxmax()
persona_map[wealth_cluster] = 'Wealthy Passive'
remaining.remove(wealth_cluster)

digital_cluster = profile.loc[list(remaining), 'TRANS_AMOUNT_SUM'].idxmax()
persona_map[digital_cluster] = 'Digital VIP'
remaining.remove(digital_cluster)

young_cluster = profile.loc[list(remaining), age_col].idxmin()
persona_map[young_cluster] = 'Young Digital'
remaining.remove(young_cluster)

activity_score = (
    profile['TRANS_NO_SUM'].rank(pct=True)
    + profile['TRANS_AMOUNT_SUM'].rank(pct=True)
    + profile['AVG_CA_BALANCE'].rank(pct=True)
)
mass_cluster = activity_score.loc[list(remaining)].idxmax()
persona_map[mass_cluster] = 'Mass Active'
remaining.remove(mass_cluster)

standard_cluster = next(iter(remaining))
persona_map[standard_cluster] = 'Standard'

df_cust['PERSONA'] = df_cust['CLUSTER'].map(persona_map)

print("\nPersona mapping:")
for c, name in persona_map.items():
    print(f"Cluster {c} -> {name}")

out_path = os.path.join(BASE_DIR, "NBFO_IB", "processed_data", "ib_final_personas.parquet")
df_cust[['CUSTOMER_NUMBER', 'CLUSTER', 'PERSONA']].to_parquet(out_path)
print(f"\nSaved IB personas to {out_path}")
