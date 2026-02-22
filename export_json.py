#!/usr/bin/env python3
"""
Export EMP pipeline output to docs/data.json for GitHub Pages dashboard.
Includes DFM confidence band columns (emp_dfm, emp_dfm_lo, emp_dfm_hi).
"""
import json, math, os
import pandas as pd

def clean(v):
    if isinstance(v, float) and math.isnan(v):
        return None
    return round(v, 6) if isinstance(v, float) else v

daily = pd.read_csv("emp_output_daily.csv")

# All columns exported to dashboard
EXPORT_COLS = [
    "date", "mid", "spread_pct", "gap", "log_depth",
    "z_gap", "z_spread", "z_liq",
    "emp", "emp_rolling",
    "emp_dfm", "emp_dfm_lo", "emp_dfm_hi",
    "prob_depr_14d", "prob_depr_30d",
    "alarm_mass_14d", "alarm_mass_30d",
]
# Only include columns that actually exist in the CSV
export_cols = [c for c in EXPORT_COLS if c in daily.columns]
records = [
    {k: clean(v) for k, v in row.items()}
    for _, row in daily[export_cols].iterrows()
]

payload = {
    "updated":      daily.date.max(),
    "official_peg": 6.96,
    "series":       records,
}

os.makedirs("docs", exist_ok=True)
with open("docs/data.json", "w") as f:
    json.dump(payload, f, indent=2)

# Also export daily CSV for direct download
daily[export_cols].to_csv("docs/emp_daily.csv", index=False)

print(f"Exported {len(records)} rows → docs/data.json")
print(f"Columns: {export_cols}")
print(f"Latest:  {records[-1]}")
