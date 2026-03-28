"""
Portfolio Construction — SAAM Project 2026
Group: North America + Europe | Scope 1 + Scope 2

Part I: Portfolio without carbon constraint
"""

import pandas as pd
import numpy as np
from pathlib import Path

# --- Paths ---
INV_SET_DIR = Path("processed_data/data_investment_set")
OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Step 1: Estimate parameters (mu, sigma) for each year Y ---
print("--- Step 1: Estimate parameters (mu, sigma) ---")

for Y in range(2013, 2026):
    # Load investment set for year Y
    inv = pd.read_csv(INV_SET_DIR / f"investment_set_{Y}.csv")

    # Identify return columns (dates)
    info_cols = ["ISIN", "NAME", "Country", "Region", "MV_Y", "CO2_S1", "CO2_S2", "REV", "Carbon_Intensity"]
    return_cols = [c for c in inv.columns if c not in info_cols]

    # Extract returns matrix: rows = months, columns = firms
    R = inv[return_cols].T.astype(float)
    R.columns = inv["ISIN"]

    # Drop months where all values are NaN
    R = R.dropna(how="all")

    # Estimate mean returns (mu) and covariance matrix (sigma)
    mu = R.mean()
    sigma = R.cov()

    # Annualize
    mu_annual = mu * 12
    sigma_annual = sigma * 12

    # Summary
    print(f"Y={Y} | Firms: {len(inv)} | Months: {len(R)} | "
          f"Avg annual return: {mu_annual.mean():.4f} | "
          f"Avg annual vol: {np.sqrt(np.diag(sigma_annual)).mean():.4f}")
