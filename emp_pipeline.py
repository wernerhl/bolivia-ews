#!/usr/bin/env python3
"""
Bolivia P2P Dollar Early Warning System — v2
=============================================
Changes from v1:
  - Historical baseline merge: loads dolar_buy_full.csv + dolar_sell_full.csv
    as the Aug 2024 → present baseline and appends any new Kaggle observations
    for dates beyond the baseline's last timestamp.
  - DFM confidence band: adds emp_dfm (Kalman-filtered real-time estimate),
    emp_dfm_lo, emp_dfm_hi (95% CI) alongside the static PCA EMP.
  - New columns propagated through daily aggregation and Supabase upload.
"""

import os
import math
import argparse
import warnings
import numpy as np
import pandas as pd
import kagglehub
from kagglehub import KaggleDatasetAdapter
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from supabase import create_client

warnings.filterwarnings("ignore")

# ── Constants ─────────────────────────────────────────────────────────────────
OFFICIAL_PEG      = 6.96
ROLLING_WINDOW    = 30 * 48        # 30-day window in 30-min steps
DEPR_THRESHOLD    = 0.05
DEPR_HORIZON      = 14
DEPR_HORIZON30    = 30

# Historical baseline CSVs (committed to repo, never change)
BASELINE_BUY_CSV  = "dolar_buy_full.csv"   # ASK side  (Aug 2024 → Nov 2025)
BASELINE_SELL_CSV = "dolar_sell_full.csv"  # BID side

# ── Supabase ──────────────────────────────────────────────────────────────────
def supabase_client():
    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
    return create_client(url, key)

def upsert(client, table, df, keys):
    def clean_record(r):
        return {k: (None if isinstance(v, float) and math.isnan(v) else v)
                for k, v in r.items()}
    records = [clean_record(r) for r in df.to_dict(orient="records")]
    on_conflict = ",".join(keys)
    CHUNK = 3000
    for i in range(0, len(records), CHUNK):
        client.table(table).upsert(
            records[i:i + CHUNK], on_conflict=on_conflict
        ).execute()
    print(f"  Upserted {len(records)} rows → {table}")

# ── Data loading ──────────────────────────────────────────────────────────────
def load_kaggle_raw():
    print("Downloading from Kaggle...")
    return kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        "andreschirinos/p2p-bob-exchange",
        "advice.parquet",
    )

def kaggle_to_sides(kg):
    """Convert raw Kaggle parquet to {ask, bid} price+depth DataFrames."""
    results = {}
    for trade_type, side in [("BUY", "bid"), ("SELL", "ask")]:
        df = kg[
            (kg.asset == "USDT") & (kg.tradetype == trade_type)
        ][["advertiser_userno", "timestamp", "price", "tradablequantity"]].copy()
        vwap = pd.Series({
            t: np.average(g.price, weights=g.tradablequantity)
            for t, g in df.groupby("timestamp")
        })
        depth = df.groupby("timestamp").tradablequantity.sum()
        results[side] = pd.DataFrame({"price": vwap, "depth": depth})
    return results

def load_baseline_sides():
    """Load the committed historical CSV baseline (buy_full / sell_full)."""
    buy  = pd.read_csv(BASELINE_BUY_CSV,  parse_dates=["timestamp"])
    sell = pd.read_csv(BASELINE_SELL_CSV, parse_dates=["timestamp"])
    ask = buy.set_index("timestamp")[["vwap_advs", "supply"]].rename(
        columns={"vwap_advs": "price", "supply": "depth"})
    bid = sell.set_index("timestamp")[["vwap_advs", "supply"]].rename(
        columns={"vwap_advs": "price", "supply": "depth"})
    return {"ask": ask, "bid": bid}

def merge_sides(baseline_sides, kaggle_sides):
    """
    Merge historical baseline with new Kaggle data.
    For each side, take baseline up to its last timestamp, then append
    any Kaggle observations strictly after that cutoff.
    """
    merged = {}
    for side in ("ask", "bid"):
        base = baseline_sides[side].sort_index()
        new  = kaggle_sides[side].sort_index()
        cutoff = base.index.max()
        new_tail = new[new.index > cutoff]
        if len(new_tail):
            combined = pd.concat([base, new_tail])
            print(f"  {side}: baseline={len(base):,}  new={len(new_tail):,}  "
                  f"total={len(combined):,}  "
                  f"({base.index.min().date()} → {combined.index.max().date()})")
        else:
            combined = base
            print(f"  {side}: baseline={len(base):,}  (no new Kaggle rows beyond "
                  f"{cutoff.date()})")
        merged[side] = combined
    return merged

# ── Microstructure ────────────────────────────────────────────────────────────
def build_microstructure(sides):
    ask = sides["ask"].sort_index()
    bid = sides["bid"].sort_index()
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
    df = df.resample("30min").last().dropna()
    return df

# ── EMP: Static PCA ───────────────────────────────────────────────────────────
def rolling_zscore(series, window):
    mu  = series.rolling(window, min_periods=window // 3).mean()
    sig = series.rolling(window, min_periods=window // 3).std()
    return (series - mu) / sig.where(sig > 1e-8, 1e-8)

def compute_emp_pca(df, window=ROLLING_WINDOW):
    """
    Full-sample standardised components + PCA-based EMP (static and rolling).
    Orientation: higher EMP = more pressure.
    """
    df = df.copy()

    # Full-sample z-scores
    for col, sign, raw in [
        ("z_gap",    +1, "gap"),
        ("z_spread", +1, "spread_pct"),
        ("z_liq",    -1, "log_depth"),
    ]:
        mu  = df[raw].mean()
        sig = df[raw].std()
        df[col] = sign * (df[raw] - mu) / max(sig, 1e-8)

    # Rolling z-scores
    for col, sign, raw in [
        ("rz_gap",    +1, "gap"),
        ("rz_spread", +1, "spread_pct"),
        ("rz_liq",    -1, "log_depth"),
    ]:
        df[col] = sign * rolling_zscore(df[raw], window)

    # Static PCA → EMP
    z_cols = ["z_gap", "z_spread", "z_liq"]
    valid  = df[z_cols].dropna()
    if len(valid) > 10:
        pca = PCA(n_components=1)
        pc  = pca.fit_transform(valid.values)
        if pca.components_[0, 0] < 0:
            pc = -pc
        df.loc[valid.index, "emp"] = pc.ravel()
        print(f"  PCA variance explained: {pca.explained_variance_ratio_[0]*100:.1f}%")

    # Rolling PCA → emp_rolling
    rz_cols = ["rz_gap", "rz_spread", "rz_liq"]
    valid_r = df[rz_cols].dropna()
    if len(valid_r) > 10:
        pca_r = PCA(n_components=1)
        pc_r  = pca_r.fit_transform(valid_r.values)
        if pca_r.components_[0, 0] < 0:
            pc_r = -pc_r
        df.loc[valid_r.index, "emp_rolling"] = pc_r.ravel()

    return df

# ── EMP: DFM Kalman Filter ────────────────────────────────────────────────────
def compute_emp_dfm(df):
    """
    Dynamic Factor Model (AR(1) latent factor, diagonal idiosyncratic noise).
    Adds three columns:
      emp_dfm     — Kalman-filtered (real-time / causal) factor estimate
      emp_dfm_lo  — 95% CI lower bound  (emp_dfm − 1.96 × filter SE)
      emp_dfm_hi  — 95% CI upper bound  (emp_dfm + 1.96 × filter SE)
    Orientation: same sign convention as PCA EMP (positive = more pressure).
    Falls back gracefully if statsmodels is not available.
    """
    try:
        import statsmodels.api as sm
    except ImportError:
        print("  statsmodels not available — skipping DFM (pip install statsmodels)")
        df["emp_dfm"]    = np.nan
        df["emp_dfm_lo"] = np.nan
        df["emp_dfm_hi"] = np.nan
        return df

    df = df.copy()
    z_cols = ["z_gap", "z_spread", "z_liq"]
    valid  = df[z_cols].dropna()

    if len(valid) < 60:
        print("  DFM: insufficient data (<60 obs) — skipping")
        df["emp_dfm"]    = np.nan
        df["emp_dfm_lo"] = np.nan
        df["emp_dfm_hi"] = np.nan
        return df

    # Daily aggregated z-scores for DFM (runs on daily data; intraday
    # would be too large for the state-space solver and adds no value
    # for a daily surveillance dashboard)
    daily_z = valid.resample("D").last().dropna()

    try:
        model = sm.tsa.DynamicFactor(
            daily_z, k_factors=1, factor_order=1
        )
        res = model.fit(disp=False, maxiter=500)

        emp_filtered_raw = res.filtered_state[0].copy()      # causal, raw scale
        emp_filt_se_raw  = np.sqrt(res.filtered_state_cov[0, 0, :])

        # Orient and rescale to match PCA EMP sign and σ-units
        from scipy.stats import pearsonr
        pca_daily = df["emp"].resample("D").last().reindex(daily_z.index).dropna()
        pca_aligned = pca_daily.values
        if pearsonr(emp_filtered_raw, pca_aligned)[0] < 0:
            emp_filtered_raw = -emp_filtered_raw

        # Affine rescale: match mean and std of PCA daily series
        scale = pca_aligned.std() / max(emp_filtered_raw.std(), 1e-8)
        emp_filtered = (emp_filtered_raw - emp_filtered_raw.mean()) * scale + pca_aligned.mean()
        emp_filt_se  = emp_filt_se_raw * scale

        # Reindex back to 30-min intraday (forward-fill within each day)
        dfm_daily = pd.DataFrame({
            "emp_dfm":    emp_filtered,
            "emp_dfm_lo": emp_filtered - 1.96 * emp_filt_se,
            "emp_dfm_hi": emp_filtered + 1.96 * emp_filt_se,
        }, index=daily_z.index)

        dfm_intraday = dfm_daily.reindex(
            df.index.normalize().unique().union(dfm_daily.index)
        ).ffill().reindex(df.index, method="ffill")

        df["emp_dfm"]    = dfm_intraday["emp_dfm"]
        df["emp_dfm_lo"] = dfm_intraday["emp_dfm_lo"]
        df["emp_dfm_hi"] = dfm_intraday["emp_dfm_hi"]

        print(f"  DFM: factor AR(1)={res.params['L1.f1.f1']:.4f}  "
              f"mean filter SE=±{emp_filt_se.mean():.3f}σ")

    except Exception as exc:
        print(f"  DFM fitting failed ({exc}) — filling with NaN")
        df["emp_dfm"]    = np.nan
        df["emp_dfm_lo"] = np.nan
        df["emp_dfm_hi"] = np.nan

    return df

# ── Early Warning ─────────────────────────────────────────────────────────────
def define_depr_events(df, horizon_days, threshold=DEPR_THRESHOLD):
    horizon_steps = horizon_days * 48
    fwd_return    = df["mid"].shift(-horizon_steps) / df["mid"] - 1
    return (fwd_return > threshold).astype(float)

def fit_early_warning(df):
    features = ["z_gap", "z_spread", "z_liq"]
    df = df.copy()
    for horizon, col in [
        (DEPR_HORIZON,   "prob_depr_14d"),
        (DEPR_HORIZON30, "prob_depr_30d"),
    ]:
        df[f"event_{horizon}d"] = define_depr_events(df, horizon)
        train = df[features + [f"event_{horizon}d"]].dropna()
        if len(train) < 50 or train[f"event_{horizon}d"].nunique() < 2:
            print(f"  Warning: insufficient events for {horizon}d model, "
                  "using EMP sigmoid fallback.")
            if "emp" in df.columns:
                df[col] = 1 / (1 + np.exp(-df["emp"].fillna(0)))
            continue
        X = train[features].values
        y = train[f"event_{horizon}d"].values
        model = LogisticRegression(C=1.0, max_iter=500, class_weight="balanced")
        model.fit(X, y)
        pred_idx = df[features].dropna().index
        df.loc[pred_idx, col] = model.predict_proba(
            df.loc[pred_idx, features].values
        )[:, 1]
        print(f"  {horizon}d model: {int(y.sum())} events / {len(train)} obs")
    return df

def compute_alarm_mass(series, span=7 * 48):
    baseline = series.ewm(span=span).mean()
    return series - baseline

# ── Daily aggregation ─────────────────────────────────────────────────────────
def to_daily(df):
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
        "emp_dfm":       "last",
        "emp_dfm_lo":    "last",
        "emp_dfm_hi":    "last",
        "prob_depr_14d": "last",
        "prob_depr_30d": "last",
        "alarm_mass_14d":"last",
        "alarm_mass_30d":"last",
    }
    agg = {k: v for k, v in daily_agg.items() if k in df.columns}
    daily = df.resample("D").agg(agg).dropna(subset=["mid"])
    daily.index.name = "date"
    daily = daily.reset_index()
    daily["date"] = daily["date"].dt.strftime("%Y-%m-%d")
    return daily

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Bolivia EMP Early Warning Pipeline v2"
    )
    parser.add_argument(
        "--no-baseline", action="store_true",
        help="Skip historical CSV baseline; use Kaggle data only "
             "(reproduces v1 behaviour)"
    )
    parser.add_argument(
        "--upload", action="store_true",
        help="Upload results to Supabase"
    )
    parser.add_argument("--out", default="emp_output.csv")
    args = parser.parse_args()

    # ── 1. Load data ──────────────────────────────────────────────────────────
    kg = load_kaggle_raw()
    kaggle_sides = kaggle_to_sides(kg)

    if args.no_baseline:
        print("Using Kaggle data only (--no-baseline flag set)")
        sides = kaggle_sides
    else:
        print("Loading historical baseline CSVs...")
        baseline_sides = load_baseline_sides()
        print("Merging baseline with new Kaggle data...")
        sides = merge_sides(baseline_sides, kaggle_sides)

    # ── 2. Microstructure ──────────────────────────────────────────────────────
    print("Building microstructure series...")
    df = build_microstructure(sides)
    print(f"  {len(df):,} intraday obs  "
          f"({df.index.min().date()} → {df.index.max().date()})")

    # ── 3. Static PCA EMP ──────────────────────────────────────────────────────
    print("Computing PCA EMP index...")
    df = compute_emp_pca(df)

    # ── 4. DFM EMP + confidence bands ─────────────────────────────────────────
    print("Computing DFM EMP (Kalman filter)...")
    df = compute_emp_dfm(df)

    # ── 5. Early warning probabilities ────────────────────────────────────────
    print("Fitting early warning models...")
    df = fit_early_warning(df)

    # ── 6. Alarm mass ──────────────────────────────────────────────────────────
    for col, out in [
        ("prob_depr_14d", "alarm_mass_14d"),
        ("prob_depr_30d", "alarm_mass_30d"),
    ]:
        if col in df.columns:
            df[out] = compute_alarm_mass(df[col])

    # ── 7. Current readings ────────────────────────────────────────────────────
    lr = df.iloc[-1]
    print("\n── CURRENT READINGS ─────────────────────────────────────────────")
    print(f"  Date:            {df.index[-1]}")
    print(f"  Mid price:       {lr.mid:.4f} BOB/USDT  (peg: {OFFICIAL_PEG})")
    print(f"  Shadow premium:  {(lr.mid/OFFICIAL_PEG - 1)*100:.1f}%")
    print(f"  Spread (panic):  {lr.spread_pct*100:.1f}%")
    print(f"  Log depth:       {lr.log_depth:.3f}")
    print(f"  EMP (PCA):       {lr.get('emp', float('nan')):.3f}σ")
    print(f"  EMP (DFM):       {lr.get('emp_dfm', float('nan')):.3f}σ  "
          f"[{lr.get('emp_dfm_lo', float('nan')):.2f}, "
          f"{lr.get('emp_dfm_hi', float('nan')):.2f}]")
    print(f"  EMP (rolling):   {lr.get('emp_rolling', float('nan')):.3f}σ")
    print(f"  P(depr 14d):     {lr.get('prob_depr_14d', float('nan')):.1%}")
    print(f"  P(depr 30d):     {lr.get('prob_depr_30d', float('nan')):.1%}")
    print("─────────────────────────────────────────────────────────────────\n")

    # ── 8. Save CSVs ───────────────────────────────────────────────────────────
    df.reset_index().to_csv(args.out, index=False)
    print(f"Saved intraday output → {args.out}")

    daily = to_daily(df)
    daily_path = args.out.replace(".csv", "_daily.csv")
    daily.to_csv(daily_path, index=False)
    print(f"Saved daily output    → {daily_path}")

    # ── 9. Supabase upload ─────────────────────────────────────────────────────
    if args.upload:
        print("Uploading to Supabase...")
        sb = supabase_client()

        upsert(sb, "emp_daily", daily, ["date"])

        def _c(v):
            if isinstance(v, float) and math.isnan(v): return None
            return round(float(v), 6) if isinstance(v, (float, int, np.floating)) else v

        snap = {
            "id":             1,
            "timestamp":      df.index[-1].strftime("%Y-%m-%dT%H:%M:%SZ"),
            "mid":            _c(lr.get("mid")),
            "spread_pct":     _c(lr.get("spread_pct")),
            "gap":            _c(lr.get("gap")),
            "emp":            _c(lr.get("emp")),
            "emp_rolling":    _c(lr.get("emp_rolling")),
            "emp_dfm":        _c(lr.get("emp_dfm")),
            "emp_dfm_lo":     _c(lr.get("emp_dfm_lo")),
            "emp_dfm_hi":     _c(lr.get("emp_dfm_hi")),
            "prob_depr_14d":  _c(lr.get("prob_depr_14d")),
            "prob_depr_30d":  _c(lr.get("prob_depr_30d")),
            "alarm_mass_14d": _c(lr.get("alarm_mass_14d")),
            "alarm_mass_30d": _c(lr.get("alarm_mass_30d")),
            "updated_at":     pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        sb.table("emp_latest").upsert(snap, on_conflict="id").execute()
        print("  Upserted latest snapshot → emp_latest")

    return df, daily


if __name__ == "__main__":
    main()
