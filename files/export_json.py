#!/usr/bin/env python3
"""Export EMP pipeline output to docs/data.json for GitHub Pages dashboard."""
import json, math, os
import pandas as pd

def clean(v):
    if isinstance(v, float) and math.isnan(v):
        return None
    return round(v, 6) if isinstance(v, float) else v

daily = pd.read_csv("emp_output_daily.csv")
records = [{k: clean(v) for k, v in row.items()} for _, row in daily.iterrows()]

payload = {
    "updated":      daily.date.max(),
    "official_peg": 6.96,
    "series":       records
}

os.makedirs("docs", exist_ok=True)
with open("docs/data.json", "w") as f:
    json.dump(payload, f, indent=2)

# Also copy the daily CSV for download
daily.to_csv("docs/emp_daily.csv", index=False)

print(f"Exported {len(records)} rows → docs/data.json")
print(f"Latest: {records[-1]}")
