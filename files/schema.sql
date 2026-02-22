-- ============================================================
-- Bolivia P2P Dollar Early Warning System — Supabase Schema
-- Run this once in:
-- https://supabase.com/dashboard/project/ccevluxyjokafotaokwn/editor
-- ============================================================

CREATE TABLE IF NOT EXISTS emp_daily (
    date            DATE PRIMARY KEY,
    mid             FLOAT,   -- Shadow mid-price (BOB/USDT)
    spread_pct      FLOAT,   -- Bid-ask spread as % of mid (panic premium)
    gap             FLOAT,   -- Shadow rate - official peg (2.190 = 31% premium)
    log_depth       FLOAT,   -- Log of min order book depth
    z_gap           FLOAT,   -- Standardised gap component
    z_spread        FLOAT,   -- Standardised spread component
    z_liq           FLOAT,   -- Standardised liquidity stress component
    emp             FLOAT,   -- EMP index (PCA first component, full-sample)
    emp_rolling     FLOAT,   -- Rolling 30-day EMP (short-run stress)
    prob_depr_14d   FLOAT,   -- P(≥5% depreciation in next 14 days)
    prob_depr_30d   FLOAT,   -- P(≥5% depreciation in next 30 days)
    alarm_mass_14d  FLOAT,   -- Deviation of prob_14d from EMA baseline
    alarm_mass_30d  FLOAT,   -- Deviation of prob_30d from EMA baseline
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Enable Row Level Security with public read access
ALTER TABLE emp_daily ENABLE ROW LEVEL SECURITY;

CREATE POLICY IF NOT EXISTS "Public read access"
    ON emp_daily FOR SELECT
    USING (true);

-- Latest snapshot (single-row, for live gauge)
CREATE TABLE IF NOT EXISTS emp_latest (
    id              INT PRIMARY KEY DEFAULT 1,
    timestamp       TIMESTAMPTZ,
    mid             FLOAT,
    spread_pct      FLOAT,
    gap             FLOAT,
    emp             FLOAT,
    emp_rolling     FLOAT,
    prob_depr_14d   FLOAT,
    prob_depr_30d   FLOAT,
    alarm_mass_14d  FLOAT,
    alarm_mass_30d  FLOAT,
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

ALTER TABLE emp_latest ENABLE ROW LEVEL SECURITY;

CREATE POLICY IF NOT EXISTS "Public read access"
    ON emp_latest FOR SELECT
    USING (true);
