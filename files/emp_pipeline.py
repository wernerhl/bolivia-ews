#!/usr/bin/env python3
"""
Bolivia P2P Dollar Early Warning System
Computes EMP index and depreciation warning probabilities from Binance P2P data.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import kagglehub
from kagglehub import KaggleDatasetAdapter
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from supabase import create_client

# ── Constants ─────────────────────────────────────────────────────────────────
OFFICIAL_PEG    = 6.96          # BOB per USD
ROLLING_WINDOW  = 30 * 48       # ~30 days of 30-min snapshots
DEPR_THRESHOLD  = 0.05          # 5% depreciation = event
DEPR_HORIZON    = 14            # days ahead for 14-day warning
DEPR_HORIZON30  = 30            # days ahead for 30-day warning
DAILY_FREQ      = "30min"

# ── Supabase ──────────────────────────────────────────────────────────────────
def supabase_client():
    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
    return create_client(url, key)

def upsert(client, table, df, keys):
    import math
    def clean_record(r):
        return {k: (None if isinstance(v, float) and math.isnan(v) else v)
                for k, v in r.items()}
    records = [clean_record(r) for r in df.to_dict(orient="records")]
    on_conflict = ",".join(keys)
    CHUNK = 3000
    for i in range(0, len(records), CHUNK):
        client.table(table).upsert(records[i:i+CHUNK], on_conflict=on_conflict).execute()
    print(f"  Upserted {len(records)} rows → {table}")

# ── Data loading ──────────────────────────────────────────────────────────────
def load_from_kaggle():
    print("Downloading from Kaggle...")
    kg = kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        "andreschirinos/p2p-bob-exchange",
        "advice.parquet",
    )
    return kg

def compute_raw_series(kg):
    """From raw Kaggle data compute ask/bid VWAP and depth series."""
    results = {}
    for trade_type, side in [("BUY", "bid"), ("SELL", "ask")]:
        df = kg[(kg.asset == "USDT") & (kg.tradetype == trade_type)][
            ["advertiser_userno", "timestamp", "price", "tradablequantity"]
        ].copy()

        # Advertisement VWAP
        vwap_advs = pd.Series({
            t: np.average(g.price, weights=g.tradablequantity)
            for t, g in df.groupby("timestamp")
        })
        supply = df.groupby("timestamp").tradablequantity.sum()
        results[side] = pd.DataFrame({"price": vwap_advs, "depth": supply})
    return results

def load_from_csvs(buy_csv, sell_csv):
    """Load from pre-computed CSVs (dolar_buy = ASK, dolar_sell = BID)."""
    buy  = pd.read_csv(buy_csv,  parse_dates=["timestamp"])
    sell = pd.read_csv(sell_csv, parse_dates=["timestamp"])
    ask  = buy.set_index("timestamp")[["vwap_advs", "supply"]].rename(
        columns={"vwap_advs": "price", "supply": "depth"})
    bid  = sell.set_index("timestamp")[["vwap_advs", "supply"]].rename(
        columns={"vwap_advs": "price", "supply": "depth"})
    return {"ask": ask, "bid": bid}

# ── EMP Construction ──────────────────────────────────────────────────────────
def build_microstructure(sides):
    ask = sides["ask"].sort_index()
    bid = sides["bid"].sort_index()

    # Align to nearest snapshot
    df = pd.merge_asof(
        ask, bid,
        left_index=True, right_index=True,
        suffixes=("_ask", "_bid"),
        tolerance=pd.Timedelta("45min"),
        direction="nearest",
    ).dropna()

    df["mid"]        = (df["price_ask"] + df["price_bid"]) / 2
    df["spread_abs"] = df["price_ask"] - df["price_bid"]
    df["spread_pct"] = df["spread_abs"] / df["mid"]
    df["gap"]        = df["mid"] - OFFICIAL_PEG
    df["log_depth"]  = np.log(df[["depth_ask", "depth_bid"]].min(axis=1))

    # Resample to 30-min to get regular grid
    df = df.resample("30min").last().dropna()
    return df

def rolling_zscore(series, window):
    mu  = series.rolling(window, min_periods=window // 3).mean()
    sig = series.rolling(window, min_periods=window // 3).std()
    return (series - mu) / sig.where(sig > 1e-8, 1e-8)

def compute_emp(df, window=ROLLING_WINDOW):
    """
    Three standardised components (following paper §5.1):
      z_gap      : shadow rate deviation above official peg  (+= more pressure)
      z_spread   : percentage bid-ask spread                 (+= more pressure)
      z_liq      : liquidity stress = -log(depth) z-scored   (+= more pressure)
    EMP = first principal component of the three z-scores.
    """
    df = df.copy()

    # Full-sample z-scores (static EMP, tracks level)
    for col, sign, raw in [
        ("z_gap",    +1, "gap"),
        ("z_spread", +1, "spread_pct"),
        ("z_liq",    -1, "log_depth"),
    ]:
        mu  = df[raw].mean()
        sig = df[raw].std()
        df[col] = sign * (df[raw] - mu) / max(sig, 1e-8)

    # Rolling z-scores (rolling EMP, tracks short-run stress)
    for col, sign, raw in [
        ("rz_gap",    +1, "gap"),
        ("rz_spread", +1, "spread_pct"),
        ("rz_liq",    -1, "log_depth"),
    ]:
        df[col] = sign * rolling_zscore(df[raw], window)

    # PCA on static components → EMP
    z_cols = ["z_gap", "z_spread", "z_liq"]
    valid  = df[z_cols].dropna()
    if len(valid) > 10:
        pca = PCA(n_components=1)
        pc  = pca.fit_transform(valid.values)
        # Orient so higher = more pressure (gap has dominant positive loading)
        if pca.components_[0, 0] < 0:
            pc = -pc
        df.loc[valid.index, "emp"] = pc.ravel()
        print(f"  PCA variance explained: {pca.explained_variance_ratio_[0]*100:.1f}%")

    # Rolling EMP (same orientation)
    rz_cols = ["rz_gap", "rz_spread", "rz_liq"]
    valid_r = df[rz_cols].dropna()
    if len(valid_r) > 10:
        pca_r = PCA(n_components=1)
        pc_r  = pca_r.fit_transform(valid_r.values)
        if pca_r.components_[0, 0] < 0:
            pc_r = -pc_r
        df.loc[valid_r.index, "emp_rolling"] = pc_r.ravel()

    return df

# ── Early Warning ─────────────────────────────────────────────────────────────
def define_depr_events(df, horizon_days, threshold=DEPR_THRESHOLD):
    """
    Binary event: mid price increases >threshold% within horizon_days ahead.
    Positive sign = depreciation (BOB weakens, more BOB per USDT).
    """
    horizon_steps = horizon_days * 48  # 30-min steps per day
    fwd_return    = df["mid"].shift(-horizon_steps) / df["mid"] - 1
    return (fwd_return > threshold).astype(float)

def fit_early_warning(df):
    """
    Logit model: P(depreciation event in next H days | EMP components).
    Returns fitted probabilities for 14d and 30d horizons.
    """
    features = ["z_gap", "z_spread", "z_liq"]
    df = df.copy()

    for horizon, col in [(DEPR_HORIZON, "prob_depr_14d"), (DEPR_HORIZON30, "prob_depr_30d")]:
        df[f"event_{horizon}d"] = define_depr_events(df, horizon)

        # Training rows: have features + known outcome
        train = df[features + [f"event_{horizon}d"]].dropna()
        if len(train) < 50 or train[f"event_{horizon}d"].nunique() < 2:
            print(f"  Warning: insufficient events for {horizon}d model, using EMP proxy.")
            # Fallback: sigmoid of scaled EMP
            if "emp" in df.columns:
                df[col] = 1 / (1 + np.exp(-df["emp"].fillna(0)))
            continue

        X = train[features].values
        y = train[f"event_{horizon}d"].values

        model = LogisticRegression(C=1.0, max_iter=500, class_weight="balanced")
        model.fit(X, y)

        # Predict on all rows with features
        pred_idx = df[features].dropna().index
        df.loc[pred_idx, col] = model.predict_proba(
            df.loc[pred_idx, features].values
        )[:, 1]

        # Count events for reporting
        n_events = int(y.sum())
        print(f"  {horizon}d model: {n_events} events / {len(train)} obs")

    return df

def compute_alarm_mass(series, span=7 * 48):
    """Deviation of probability from its EMA baseline."""
    baseline = series.ewm(span=span).mean()
    return series - baseline

# ── Aggregation ───────────────────────────────────────────────────────────────
def to_daily(df):
    """Aggregate 30-min series to daily for Supabase storage."""
    daily_agg = {
        "mid":           "last",
        "spread_pct":    "mean",
        "gap":           "last",
        "log_depth":     "mean",
        "z_gap":         "last",
        "z_spread":      "last",
        "z_liq":         "last",
        "emp":           "last",
        "emp_rolling":   "last",
        "prob_depr_14d": "last",
        "prob_depr_30d": "last",
        "alarm_mass_14d":"last",
        "alarm_mass_30d":"last",
    }
    # Only aggregate columns that exist
    agg = {k: v for k, v in daily_agg.items() if k in df.columns}
    daily = df.resample("D").agg(agg).dropna(subset=["mid"])
    daily.index.name = "date"
    daily = daily.reset_index()
    daily["date"] = daily["date"].dt.strftime("%Y-%m-%d")
    return daily

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Bolivia EMP Early Warning Pipeline")
    parser.add_argument("--csv",    action="store_true", help="Use local CSVs instead of Kaggle")
    parser.add_argument("--upload", action="store_true", help="Upload results to Supabase")
    parser.add_argument("--buy-csv",  default="dolar_buy.csv")
    parser.add_argument("--sell-csv", default="dolar_sell.csv")
    parser.add_argument("--out",    default="emp_output.csv")
    args = parser.parse_args()

    # 1. Load data
    if args.csv:
        print("Loading from local CSVs...")
        sides = load_from_csvs(args.buy_csv, args.sell_csv)
    else:
        kg = load_from_kaggle()
        sides = compute_raw_series(kg)

    # 2. Build microstructure
    print("Building microstructure series...")
    df = build_microstructure(sides)
    print(f"  {len(df)} observations from {df.index.min().date()} to {df.index.max().date()}")

    # 3. EMP components and index
    print("Computing EMP index...")
    df = compute_emp(df)

    # 4. Early warning probabilities
    print("Fitting early warning models...")
    df = fit_early_warning(df)

    # 5. Alarm mass
    for col, out in [("prob_depr_14d", "alarm_mass_14d"), ("prob_depr_30d", "alarm_mass_30d")]:
        if col in df.columns:
            df[out] = compute_alarm_mass(df[col])

    # 6. Print current readings
    latest = df.iloc[-1]
    print("\n── CURRENT READINGS ──────────────────────────────")
    print(f"  Date:          {df.index[-1]}")
    print(f"  Mid price:     {latest.mid:.4f} BOB/USDT  (peg: {OFFICIAL_PEG})")
    print(f"  Spread:        {latest.spread_pct*100:.1f}%")
    print(f"  Shadow gap:    {latest.gap:.4f} BOB")
    print(f"  EMP:           {latest.get('emp', float('nan')):.3f}")
    print(f"  EMP (rolling): {latest.get('emp_rolling', float('nan')):.3f}")
    print(f"  P(depr 14d):   {latest.get('prob_depr_14d', float('nan')):.3f}")
    print(f"  P(depr 30d):   {latest.get('prob_depr_30d', float('nan')):.3f}")
    print("──────────────────────────────────────────────────\n")

    # 7. Save intraday CSV
    df.reset_index().to_csv(args.out, index=False)
    print(f"Saved intraday output → {args.out}")

    # 8. Daily aggregation + Supabase upload
    daily = to_daily(df)
    daily_path = args.out.replace(".csv", "_daily.csv")
    daily.to_csv(daily_path, index=False)
    print(f"Saved daily output → {daily_path}")

    if args.upload:
        print("Uploading to Supabase...")
        sb = supabase_client()

        # Upload daily EMP
        upsert(sb, "emp_daily", daily, ["date"])

        # Upload latest snapshot for live dashboard
        snap = df.reset_index().tail(1).copy()
        snap["timestamp"] = snap["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        snap_cols = [c for c in ["timestamp","mid","spread_pct","gap",
                                  "emp","emp_rolling","prob_depr_14d","prob_depr_30d",
                                  "alarm_mass_14d","alarm_mass_30d"] if c in snap.columns]
        upsert(sb, "emp_latest", snap[snap_cols], ["timestamp"])

    return df, daily

if __name__ == "__main__":
    main()
