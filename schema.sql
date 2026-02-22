-- ============================================================
-- Bolivia P2P Dollar Early Warning System — Schema v2 Migration
-- Adds DFM Kalman-filter confidence band columns.
--
-- Run in Supabase SQL editor:
-- https://supabase.com/dashboard/project/ccevluxyjokafotaokwn/editor
-- ============================================================

-- ── emp_daily: add DFM columns ────────────────────────────────────────────────
ALTER TABLE emp_daily
    ADD COLUMN IF NOT EXISTS emp_dfm     FLOAT,  -- DFM Kalman-filtered EMP (real-time)
    ADD COLUMN IF NOT EXISTS emp_dfm_lo  FLOAT,  -- 95% CI lower bound (emp_dfm − 1.96·SE)
    ADD COLUMN IF NOT EXISTS emp_dfm_hi  FLOAT;  -- 95% CI upper bound (emp_dfm + 1.96·SE)

COMMENT ON COLUMN emp_daily.emp_dfm    IS
    'DFM Kalman-filtered EMP factor (real-time / causal). '
    'Derived from a Dynamic Factor Model with AR(1) latent factor. '
    'Smooths out idiosyncratic noise in z_spread and z_liq; '
    'closely tracks z_gap (R²=0.999 with common factor).';

COMMENT ON COLUMN emp_daily.emp_dfm_lo IS
    '95% confidence interval lower bound for emp_dfm: emp_dfm − 1.96 × Kalman filter SE.';

COMMENT ON COLUMN emp_daily.emp_dfm_hi IS
    '95% confidence interval upper bound for emp_dfm: emp_dfm + 1.96 × Kalman filter SE.';

-- ── emp_latest: add DFM columns ───────────────────────────────────────────────
ALTER TABLE emp_latest
    ADD COLUMN IF NOT EXISTS emp_dfm     FLOAT,
    ADD COLUMN IF NOT EXISTS emp_dfm_lo  FLOAT,
    ADD COLUMN IF NOT EXISTS emp_dfm_hi  FLOAT;

-- ── Full schema reference (v2) ────────────────────────────────────────────────
-- Run this block only if creating the tables from scratch on a new project.

/*
CREATE TABLE IF NOT EXISTS emp_daily (
    date            DATE PRIMARY KEY,
    mid             FLOAT,   -- Shadow mid-price (BOB/USDT)
    spread_pct      FLOAT,   -- Bid-ask spread as fraction of mid (panic premium)
    gap             FLOAT,   -- Shadow rate − official peg (BOB)
    log_depth       FLOAT,   -- Log of min(ask depth, bid depth) in USDT
    z_gap           FLOAT,   -- Standardised gap component
    z_spread        FLOAT,   -- Standardised spread component
    z_liq           FLOAT,   -- Standardised liquidity stress component (−log depth)
    emp             FLOAT,   -- EMP index (PCA first component, full-sample)
    emp_rolling     FLOAT,   -- Rolling 30-day PCA EMP
    emp_dfm         FLOAT,   -- DFM Kalman-filtered EMP (real-time)
    emp_dfm_lo      FLOAT,   -- 95% CI lower bound
    emp_dfm_hi      FLOAT,   -- 95% CI upper bound
    prob_depr_14d   FLOAT,   -- P(≥5% shadow depreciation in next 14 days)
    prob_depr_30d   FLOAT,   -- P(≥5% shadow depreciation in next 30 days)
    alarm_mass_14d  FLOAT,   -- prob_14d deviation from EMA baseline
    alarm_mass_30d  FLOAT,   -- prob_30d deviation from EMA baseline
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

ALTER TABLE emp_daily ENABLE ROW LEVEL SECURITY;
CREATE POLICY IF NOT EXISTS "Public read access"
    ON emp_daily FOR SELECT USING (true);

CREATE TABLE IF NOT EXISTS emp_latest (
    id              INT PRIMARY KEY DEFAULT 1,
    timestamp       TIMESTAMPTZ,
    mid             FLOAT,
    spread_pct      FLOAT,
    gap             FLOAT,
    emp             FLOAT,
    emp_rolling     FLOAT,
    emp_dfm         FLOAT,
    emp_dfm_lo      FLOAT,
    emp_dfm_hi      FLOAT,
    prob_depr_14d   FLOAT,
    prob_depr_30d   FLOAT,
    alarm_mass_14d  FLOAT,
    alarm_mass_30d  FLOAT,
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

ALTER TABLE emp_latest ENABLE ROW LEVEL SECURITY;
CREATE POLICY IF NOT EXISTS "Public read access"
    ON emp_latest FOR SELECT USING (true);
*/
