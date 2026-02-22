# Bolivia P2P Dollar — Early Warning System

Real-time Exchange Market Pressure (EMP) index and depreciation warning probabilities
derived from Binance P2P USDT/BOB order book data. Implements and extends
Hernani-Limarino (2025) methodology.

---

## Architecture

```
Kaggle (P2P data) → emp_pipeline.py → Supabase (storage)
                                     → docs/data.json   → GitHub Pages (dashboard)
```

- **Pipeline** runs daily via GitHub Actions at 10:00 AM Bolivia time
- **Dashboard** is a static HTML page served from GitHub Pages, reads `docs/data.json`
- **Supabase** stores the full history for additional querying

---

## One-time Setup

### 1. Create Supabase tables

Go to: https://supabase.com/dashboard/project/ccevluxyjokafotaokwn/editor

Paste and run the contents of `schema.sql`.

### 2. Create GitHub repository

```bash
git init bolivia-ews
cd bolivia-ews
cp -r /path/to/these/files/* .
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/bolivia-ews.git
git push -u origin main
```

### 3. Enable GitHub Pages

- Go to repo Settings → Pages
- Source: **Deploy from branch**
- Branch: `main` / folder: `/docs`
- Save → your dashboard will be at: `https://YOUR_USERNAME.github.io/bolivia-ews`

### 4. Add GitHub Secrets

Go to repo Settings → Secrets and variables → Actions → New repository secret:

| Name                      | Value                          |
|---------------------------|--------------------------------|
| `KAGGLE_USERNAME`         | `whl666`                       |
| `KAGGLE_KEY`              | `7949129fa132d2390e90cd2e6542358f` |
| `SUPABASE_URL`            | `https://ccevluxyjokafotaokwn.supabase.co` |
| `SUPABASE_SERVICE_ROLE_KEY` | (your service role key)      |

### 5. Trigger first run

Go to repo Actions → Daily EMP Update → Run workflow → Run workflow

---

## Local usage

```bash
pip install -r requirements.txt

# Run with Kaggle download
python emp_pipeline.py --upload

# Run with local CSVs (faster, for testing)
python emp_pipeline.py --csv --buy-csv dolar_buy.csv --sell-csv dolar_sell.csv

# Export dashboard JSON
python export_json.py
```

---

## EMP Methodology

Three standardised components (following paper §5):

| Component   | Raw variable          | Direction |
|-------------|----------------------|-----------|
| z_gap       | mid − 6.96 (peg)     | + = more pressure |
| z_spread    | (ask − bid) / mid    | + = more pressure |
| z_liq       | − log(min depth)     | + = more pressure |

**EMP** = first principal component of [z_gap, z_spread, z_liq]

**Early warning** = logistic regression of EMP components on binary depreciation events
(≥5% change in mid price over the next 14/30 days).

**Alarm mass** = deviation of predicted probability from its 7-day EMA baseline.

---

## Files

```
emp_pipeline.py          — Main computation + Supabase upload
export_json.py           — Exports docs/data.json for dashboard
schema.sql               — Supabase table definitions (run once)
requirements.txt         — Python dependencies
upload.py                — Supabase upsert helper
docs/
  index.html             — Dashboard (served by GitHub Pages)
  data.json              — Latest EMP series (committed daily)
  emp_daily.csv          — Daily data download
.github/workflows/
  daily.yml              — GitHub Actions schedule
```
