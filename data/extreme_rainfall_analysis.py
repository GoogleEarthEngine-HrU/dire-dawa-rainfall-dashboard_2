# Extreme Rainfall Analysis - Dire Dawa, Ethiopia (1981-2025)
# Comprehensive Analysis Pipeline

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from lmoments3 import distr

# Directory Setup
OUTPUT_DIR = "/content/Dire_Dawa_Extreme_Rainfall"
os.makedirs(OUTPUT_DIR, exist_ok=True)

daily_csv_path = os.path.join(OUTPUT_DIR, "Dire_Dawa_CHIRPS_Daily_1981_2025.csv")
annual_csv_path = os.path.join(OUTPUT_DIR, "Dire_Dawa_Annual_Maximum_Rainfall_1981_2025.csv")

# Load Data
print("Loading datasets...")
if not os.path.exists(daily_csv_path) or not os.path.exists(annual_csv_path):
    raise FileNotFoundError("Required datasets missing inside the folder. Please verify CSV files.")

daily_df = pd.read_csv(daily_csv_path)
annual_df = pd.read_csv(annual_csv_path)
a_max_series = annual_df["annual_max_rainfall_mm"].values

# 1. Fit GEV
print("Fitting GEV Distribution via L-Moments...")
gev_params = distr.gev.lmom_fit(data=a_max_series)
print("Fitted GEV parameters:", gev_params)

# 2. Fit POT-GPD
print("Analyzing Peak-Over-Threshold (95th percentile)...")
u_threshold = np.percentile(daily_df["rainfall_mm"].values, 95)
exceedances = daily_df[daily_df["rainfall_mm"] > u_threshold]["rainfall_mm"].values - u_threshold
gpd_shape, gpd_location, gpd_scale = stats.genpareto.fit(exceedances, floc=0)
print(f"GPD parameters: Shape={gpd_shape:.4f}, Scale={gpd_scale:.4f}")

# 3. Create GEV vs POT-GPD Comparison Plots
print("Generating return level comparison curve...")
return_periods = np.array([2, 5, 10, 25, 50, 100])
gev_rl = distr.gev.ppf(1 - 1 / return_periods, **gev_params)

duration_years = len(daily_df) / 365.2425
event_rate = len(exceedances) / duration_years
pot_rl = []
for T in return_periods:
    val = event_rate * T
    rl = u_threshold + (gpd_scale / gpd_shape) * (val**gpd_shape - 1)
    pot_rl.append(rl)

plt.figure(figsize=(10, 6))
plt.plot(return_periods, gev_rl, 'o-', label='GEV')
plt.plot(return_periods, pot_rl, 's-', label='POT-GPD')
plt.xscale('log')
plt.xlabel('Return Period (Years)')
plt.ylabel('Return Level (mm)')
plt.title('Comparison of Return Levels (GEV vs POT-GPD)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(OUTPUT_DIR, "00_Pipeline_Return_Levels_Comparison.png"), dpi=300)
print("Pipeline analysis script completed successfully. All plots saved to results folder.")
