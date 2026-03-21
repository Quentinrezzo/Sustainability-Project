"""
Investment Set — SAAM Project 2026
Group: North America + Europe | Scope 1 + Scope 2

Determines the investable firms for each year Y (2013-2024)
"""

import pandas as pd
from pathlib import Path

# --- Paths ---
CLEANED_DIR = Path("processed_data/data_cleaned")
OUTPUT_DIR = Path("processed_data/data_investment_set")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Step 1: Load cleaned data and calculate monthly returns ---
print("--- Step 1: Load cleaned data and calculate monthly returns ---")
static = pd.read_csv(CLEANED_DIR / "Static_filtered.csv")
ri_m = pd.read_csv(CLEANED_DIR / "RI_M_filtered.csv")
mv_y = pd.read_csv(CLEANED_DIR / "MV_Y_filtered.csv")
co2_s1 = pd.read_csv(CLEANED_DIR / "CO2_S1_filtered.csv")
co2_s2 = pd.read_csv(CLEANED_DIR / "CO2_S2_filtered.csv")
rev = pd.read_csv(CLEANED_DIR / "REV_filtered.csv")

# Calculate monthly returns from RI_M
date_cols = [c for c in ri_m.columns if c not in ["NAME", "ISIN"]]
prices = ri_m[date_cols].apply(pd.to_numeric, errors="coerce")
returns = prices.pct_change(axis=1)
returns.insert(0, "NAME", ri_m["NAME"])
returns.insert(1, "ISIN", ri_m["ISIN"])

print(f"Returns calculated: {len(returns)} firms, {len(date_cols)-1} months")

# --- Step 2: Exclude firms with < 48 months of returns (rolling 10-year window) ---
print("\n--- Step 2: Exclude firms with < 48 months of returns ---")

return_date_cols = [c for c in returns.columns if c not in ["NAME", "ISIN"]]
return_dates = pd.to_datetime(return_date_cols)

for Y in range(2013, 2025):
    # 10-year window: Y-9 to Y (e.g. 2004-2013)
    start_year = Y - 9
    window_cols = [c for c, d in zip(return_date_cols, return_dates) if d.year >= start_year and d.year <= Y]

    # Count non-NaN returns per firm
    valid_counts = returns[window_cols].notna().sum(axis=1)

    # Firms with >= 48 months
    eligible = valid_counts >= 48
    excluded = (~eligible).sum()

    print(f"Y={Y} | Window {start_year}-{Y} | Eligible: {eligible.sum()} | Excluded: {excluded} (< 48 months)")

# --- Step 3: Exclude firms without price at end of year Y ---
print("\n--- Step 3: Exclude firms without price at end of year Y ---")

ri_date_cols = [c for c in ri_m.columns if c not in ["NAME", "ISIN"]]
ri_dates = pd.to_datetime(ri_date_cols)
ri_prices = ri_m[ri_date_cols].apply(pd.to_numeric, errors="coerce")

for Y in range(2013, 2025):
    # Find the last month of year Y (December)
    dec_cols = [c for c, d in zip(ri_date_cols, ri_dates) if d.year == Y and d.month == 12]
    if not dec_cols:
        continue
    dec_col = dec_cols[0]

    # Firms with a valid price in December of year Y
    has_price = ri_prices[dec_col].notna()
    excluded = (~has_price).sum()

    print(f"Y={Y} | With price Dec {Y}: {has_price.sum()} | Excluded: {excluded}")

# --- Step 4: Exclude stale firms (> 50% zero returns over 10-year window) ---
print("\n--- Step 4: Exclude stale firms (> 50% zero returns) ---")

for Y in range(2013, 2025):
    start_year = Y - 9
    window_cols = [c for c, d in zip(return_date_cols, return_dates) if d.year >= start_year and d.year <= Y]

    # For each firm, count % of returns = 0 among valid returns
    window_returns = returns[window_cols]
    valid_counts = window_returns.notna().sum(axis=1)
    zero_counts = (window_returns == 0).sum(axis=1)

    # % of zero returns (avoid division by 0)
    pct_zero = zero_counts / valid_counts.replace(0, 1) * 100

    # Firms with <= 50% zero returns
    not_stale = pct_zero <= 50
    excluded = (~not_stale).sum()

    print(f"Y={Y} | Not stale: {not_stale.sum()} | Excluded: {excluded} (> 50% zero returns)")

# --- Step 5: Exclude firms without carbon data or revenue at end of year Y ---
print("\n--- Step 5: Exclude firms without carbon/revenue data at end of year Y ---")

co2_s1_cols = [c for c in co2_s1.columns if c not in ["NAME", "ISIN"]]
co2_s2_cols = [c for c in co2_s2.columns if c not in ["NAME", "ISIN"]]
rev_cols = [c for c in rev.columns if c not in ["NAME", "ISIN"]]

for Y in range(2013, 2025):
    y_str = str(Y)

    # Check CO2 Scope 1
    has_s1 = co2_s1[y_str].notna() if y_str in co2_s1_cols else pd.Series(False, index=co2_s1.index)
    # Check CO2 Scope 2
    has_s2 = co2_s2[y_str].notna() if y_str in co2_s2_cols else pd.Series(False, index=co2_s2.index)
    # Check Revenue
    has_rev = rev[y_str].notna() if y_str in rev_cols else pd.Series(False, index=rev.index)

    # Need all three
    has_all = has_s1.sum(), has_s2.sum(), has_rev.sum()

    print(f"Y={Y} | S1: {has_all[0]} | S2: {has_all[1]} | REV: {has_all[2]}")

# --- Final: Combine all filters and save investment set per year Y ---
print("\n--- Final: Combine all filters and save investment set ---")

for Y in range(2013, 2025):
    start_year = Y - 9
    y_str = str(Y)

    # Step 2: >= 48 months of returns
    window_cols = [c for c, d in zip(return_date_cols, return_dates) if d.year >= start_year and d.year <= Y]
    valid_counts = returns[window_cols].notna().sum(axis=1)
    filter_returns = valid_counts >= 48

    # Step 3: Price available in December Y
    dec_cols = [c for c, d in zip(ri_date_cols, ri_dates) if d.year == Y and d.month == 12]
    filter_price = ri_prices[dec_cols[0]].notna() if dec_cols else pd.Series(True, index=ri_m.index)

    # Step 4: Not stale (<=50% zero returns)
    window_returns = returns[window_cols]
    valid = window_returns.notna().sum(axis=1)
    zeros = (window_returns == 0).sum(axis=1)
    pct_zero = zeros / valid.replace(0, 1) * 100
    filter_stale = pct_zero <= 50

    # Step 5: Carbon + Revenue available
    has_s1 = co2_s1[y_str].notna() if y_str in co2_s1_cols else pd.Series(False, index=co2_s1.index)
    has_s2 = co2_s2[y_str].notna() if y_str in co2_s2_cols else pd.Series(False, index=co2_s2.index)
    has_rev = rev[y_str].notna() if y_str in rev_cols else pd.Series(False, index=rev.index)
    filter_carbon = has_s1 & has_s2 & has_rev

    # Combine all filters
    eligible = filter_returns & filter_price & filter_stale & filter_carbon
    eligible_isins = ri_m.loc[eligible, "ISIN"].tolist()

    # Build combined dataframe
    inv_set = static[static["ISIN"].isin(eligible_isins)].copy()

    # Add MV, CO2, REV for year Y
    inv_set = inv_set.merge(mv_y[["ISIN", y_str]].rename(columns={y_str: "MV_Y"}), on="ISIN", how="left")
    inv_set = inv_set.merge(co2_s1[["ISIN", y_str]].rename(columns={y_str: "CO2_S1"}), on="ISIN", how="left")
    inv_set = inv_set.merge(co2_s2[["ISIN", y_str]].rename(columns={y_str: "CO2_S2"}), on="ISIN", how="left")
    inv_set = inv_set.merge(rev[["ISIN", y_str]].rename(columns={y_str: "REV"}), on="ISIN", how="left")

    # Add carbon intensity
    inv_set["Carbon_Intensity"] = (inv_set["CO2_S1"] + inv_set["CO2_S2"]) / inv_set["REV"]

    # Add monthly returns (at the end)
    firm_returns = returns.loc[eligible, ["ISIN"] + window_cols]
    inv_set = inv_set.merge(firm_returns, on="ISIN", how="left")

    inv_set = inv_set.round(4)
    inv_set.to_csv(OUTPUT_DIR / f"investment_set_{Y}.csv", index=False)

    print(f"Y={Y} | Returns: -{(~filter_returns).sum()} | Price: -{(~filter_price).sum()} | Stale: -{(~filter_stale).sum()} | Carbon: -{(~filter_carbon).sum()} | Final: {eligible.sum()} firms")
