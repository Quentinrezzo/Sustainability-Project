"""
Data Cleaning — SAAM Project 2026
Group: North America + Europe | Scope 1 + Scope 2

Step 1: Load data and filter by region (AMER + EUR)
"""

import pandas as pd
import re
from pathlib import Path

# --- Paths ---
RAW_CSV = Path("raw_data/csv")
OUTPUT_DIR = Path("processed_data/data_cleaned")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Step 1: Load Static and filter by region ---
static = pd.read_csv(RAW_CSV / "Static_2025.csv")
regions = ["AMER", "EUR"]
static = static[static["Region"].isin(regions)].copy()
valid_isins = set(static["ISIN"].tolist())

print(f"Firms after region filter (AMER + EUR): {len(static)}")
print(f"  AMER: {(static['Region'] == 'AMER').sum()}")
print(f"  EUR:  {(static['Region'] == 'EUR').sum()}")

# --- Load all data files and filter by ISIN ---
files = {
    "RI_M":    "DS_RI_T_USD_M_2025.csv",
    "RI_Y":    "DS_RI_T_USD_Y_2025.csv",
    "MV_M":    "DS_MV_T_USD_M_2025.csv",
    "MV_Y":    "DS_MV_T_USD_Y_2025.csv",
    "CO2_S1":  "DS_CO2_SCOPE_1_Y_2025.csv",
    "CO2_S2":  "DS_CO2_SCOPE_2_Y_2025.csv",
    "REV":     "DS_REV_Y_2025.csv",
}

data = {}
for key, filename in files.items():
    df = pd.read_csv(RAW_CSV / filename)
    # Remove Datastream error rows ($$ER)
    df = df[df["ISIN"].notna() & ~df["NAME"].str.contains(r"\$\$ER", na=True)]
    # Filter by region
    df = df[df["ISIN"].isin(valid_isins)].copy()
    data[key] = df
    print(f"{key}: {len(df)} firms loaded")

# --- Step 2: Remove date columns before 2003 ---
def filter_columns_from_2003(df):
    """Keep NAME, ISIN + all date columns from 2003 onwards."""
    keep = []
    for col in df.columns:
        if col in ["NAME", "ISIN"]:
            keep.append(col)
        else:
            try:
                date = pd.to_datetime(col)
                if date.year >= 2003:
                    keep.append(col)
            except:
                keep.append(col)
    return df[keep]

print("\n--- Step 2: Remove date columns before 2003 ---")
for key in data:
    before = len(data[key].columns)
    data[key] = filter_columns_from_2003(data[key])
    after = len(data[key].columns)
    print(f"{key}: {before - after} columns removed, {after - 2} date columns kept")

# --- Step 3: Replace empty cells with NaN ---
print("\n--- Step 3: Replace empty cells with NaN ---")
for key in data:
    data[key] = data[key].fillna(float("nan"))
    print(f"{key}: empty cells replaced with NaN")

# --- Step 4: Handle delisted firms ---
print("\n--- Step 4: Handle delisted firms ---")

def get_delist_date(name):
    """Extract delisting date from firm name (e.g. 'DEAD - DELIST.10/01/22')"""
    match = re.search(r'(\d{2})/(\d{2})/(\d{2})', str(name))
    if match:
        day, month, year = int(match.group(1)), int(match.group(2)), int(match.group(3))
        year = 2000 + year if year < 50 else 1900 + year
        return pd.Timestamp(year=year, month=month, day=day)
    return None

def set_zero_after_delist(df, label):
    """Set 0 on first period after delisting, NaN on all subsequent."""
    date_cols = [c for c in df.columns if c not in ["NAME", "ISIN"]]
    dates = pd.to_datetime(date_cols)
    dead = df[df["NAME"].str.contains("DEAD", case=False, na=False)]
    count = 0
    for idx in dead.index:
        delist_date = get_delist_date(df.loc[idx, "NAME"])
        if delist_date is None:
            continue
        first_after = True
        for col, d in zip(date_cols, dates):
            if d >= delist_date:
                if first_after:
                    df.loc[idx, col] = 0
                    first_after = False
                else:
                    df.loc[idx, col] = float("nan")
        count += 1
    print(f"{label}: {count} delisted firms processed")
    return df

for key in ["RI_M", "RI_Y", "MV_M", "MV_Y"]:
    data[key] = set_zero_after_delist(data[key], key)

# --- Step 5: Forward-fill missing values between available values (CO2, REV) ---
print("\n--- Step 5: Forward-fill missing values (CO2_S1, CO2_S2, REV) ---")
for key in ["CO2_S1", "CO2_S2", "REV"]:
    date_cols = [c for c in data[key].columns if c not in ["NAME", "ISIN"]]
    before_nan = data[key][date_cols].isna().sum().sum()
    for idx in data[key].index:
        vals = data[key].loc[idx, date_cols]
        first_valid = vals.first_valid_index()
        if first_valid is not None:
            start = date_cols.index(first_valid)
            data[key].loc[idx, date_cols[start:]] = vals.iloc[start:].ffill()
    after_nan = data[key][date_cols].isna().sum().sum()
    print(f"{key}: {before_nan - after_nan} NaN filled ({before_nan} -> {after_nan})")

# --- Step 6: Replace low prices (< 0.5) with NaN (keep delisting zeros) ---
print("\n--- Step 6: Replace low prices (< 0.5) with NaN ---")
for key in ["RI_M", "RI_Y"]:
    date_cols = [c for c in data[key].columns if c not in ["NAME", "ISIN"]]
    dates = pd.to_datetime(date_cols)
    low_count = 0
    for idx in data[key].index:
        name = data[key].loc[idx, "NAME"]
        delist_date = get_delist_date(name) if "DEAD" in str(name).upper() else None
        for col, d in zip(date_cols, dates):
            val = data[key].loc[idx, col]
            if pd.notna(val) and val < 0.5:
                # Skip if this is the delisting zero
                if delist_date is not None and d >= delist_date:
                    continue
                data[key].loc[idx, col] = float("nan")
                low_count += 1
    print(f"{key}: {low_count} values < 0.5 replaced with NaN (delisting zeros kept)")

# --- Save filtered data ---
static.to_csv(OUTPUT_DIR / "Static_filtered.csv", index=False)
for key, df in data.items():
    df.to_csv(OUTPUT_DIR / f"{key}_filtered.csv", index=False)

print(f"\nFiltered data saved to {OUTPUT_DIR}/")
