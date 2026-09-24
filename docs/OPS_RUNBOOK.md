# StockAI — Ops Runbook

Practical guide for diagnosing and operating the LIVE trading system.
No secrets in this file — see "Credentials" for where they live.

---

## Infrastructure

| Thing | Value |
|---|---|
| Host | AWS Lightsail `stockai-scheduler`, region **us-west-2**, public IP **54.244.151.95** |
| OS / user | Ubuntu, user `ubuntu`, app at `/home/ubuntu/stock-ai-system` |
| Process | systemd service **`stockai`** (`scripts/stockai.service`) → runs `scripts/run_paper_trading.py` |
| Logs | **journald only** (no `.log` files): `journalctl -u stockai` |
| Deploy | `scripts/auto_deploy.sh` via cron every 5 min — pulls `main`, rebuilds, restarts service. Writes `/home/ubuntu/deploy.log` |
| Live S3 bucket | `s3://stock-ai-system-live` (us-west-2). Paper/old: `stock-ai-system-data` |
| Trading mode | **LIVE** (real money), ~$500 account. `config.yaml: trading.mode: live` → uses `live_risk` block |

### What syncs to S3 (and what does NOT)
Synced: `news/`, `raw/`, `discovery/`, `notifications/`, `audit/`, `signals/`.
**NOT synced:** `data/state/*` (risk_state, trailing_stops, take_profits) — kill-switch
state lives ONLY on the box. To check kill state you MUST read the box, not S3.

---

## Credentials (where they live — never commit these)
- **Alpaca live keys:** box `.env` → `ALPACA_API_KEY_LIVE`, `ALPACA_SECRET_KEY_LIVE`
- **Polygon / Finnhub / Anthropic:** box `.env`
- **AWS:** box uses instance/env creds for S3 sync (these expired 2026-09-23 → S3 backups broke; see Known Issues)
- **Local diagnostics:** a read-only IAM user + a throwaway SSH key (`~/.ssh/stockai_diag`) may be created ad hoc — **rotate/revoke after use.**

---

## Diagnostic commands

### SSH in (read-only diagnostics)
```bash
ssh -i ~/.ssh/stockai_diag ubuntu@54.244.151.95
```
(To grant a new machine: generate a keypair, append its `.pub` to the box's
`~/.ssh/authorized_keys`, connect with `-i <privkey>`. Remove the line to revoke.)

### Is it alive / what commit is deployed?
```bash
systemctl is-active stockai
cd stock-ai-system && git log --oneline -1
tail -6 /home/ubuntu/deploy.log
```

### Kill-switch / risk state (box only)
```bash
cat /home/ubuntu/stock-ai-system/data/state/risk_state.json
# {"paused":bool, "killed":bool, "peak_value_usd":float}
# killed=true → NOT trading until reset. peak_value_usd must reflect the REAL account.
```

### Live account + positions (source of truth)
```bash
cd stock-ai-system && .venv/bin/python -c "
from src.ingestion.alpaca_client import AlpacaClient
c=AlpacaClient(); a=c._trading_client.get_account()
print('equity',a.equity,'cash',a.cash,'last_equity',a.last_equity)
for p in c._trading_client.get_all_positions(): print(p.symbol,p.qty,p.market_value,p.unrealized_pl)"
```

### Recent trading activity (market hours only)
```bash
journalctl -u stockai --since "today 13:30" --until "today 20:10" --no-pager \
  | grep -iE "signal|BUY|SELL|approv|block|confidence|decision|EXIT" | grep -v TelegramBot
```

### Read live data from S3 (no SSH needed) — needs read-only AWS creds
```bash
aws s3 ls s3://stock-ai-system-live/ --recursive --region us-west-2 | sort -k1,2 | tail
aws s3 cp s3://stock-ai-system-live/market-data/audit/2026-09.parquet /tmp/a.parquet --region us-west-2
```

---

## Deploy a change
1. Commit to `main` locally, run tests: `pytest tests/test_phase1.py -v` (full suite hits live APIs / hangs — run logic suites).
2. `git push origin main` (HTTPS needs a PAT or keychain; SSH needs a GitHub key).
3. `auto_deploy.sh` picks it up within ~5 min. Verify: box HEAD == your commit, `systemctl is-active stockai` == active, and `journalctl` shows `✓ LIVE trading started` with no traceback.

---

## Known issues / findings (2026-09-24 audit)

| Area | Finding | Status |
|---|---|---|
| Early churn (Sep 1–3) | Positions held ~4.5 min (median), 73 trades ≈ −$0.64, win 5.5% — trailing stops self-triggered | Fixed by commits d2482a0/d26d870 |
| Idle since Sep 14 | Confidence gate blocked ~all signals in non-bull regime (×0.85 penalty) | Softened to ×0.92 in `188f1b0` |
| Audit accounting | `broker_closed` closed still-held positions using a stale prior-round-trip SELL | Fixed in `188f1b0` (SELL must post-date entry) |
| Learning loop | ConfidenceScorer read an `outcome` column nobody wrote → stuck on seed win-rates forever | Enabled in `188f1b0` (WIN/LOSS now written on close); switch scorer to read it after ~30+ clean trades accrue |
| Gate config bug | `confidence_scorer` reads `config.risk` (paper 0.60) not `config.effective_risk` (live 0.62) | Open — left as-is (fixing raises the gate, reduces trades) |
| Long-only | SELL/death-cross signals generated but unusable | Open — decide whether to enable shorts |
| Polygon 429s | Rate-limit errors flood discovery volume checks | Open — throttle or paid tier |
| S3 backups | AWS creds expired 2026-09-23 → audit/state stopped uploading | Open — refresh box AWS creds |
| Config vs docs | Risk rules are runtime-configurable via `live_risk`, contradicting CLAUDE.md's "hardcoded" claim; `auto_approve: true` contradicts "always False" (kept intentionally) | Open — reconcile docs |

## Ideas to improve profitability (ranked)
1. Close the learning loop (switch scorer to real win-rates once history exists).
2. Keep tuning the confidence gate / regime penalty; measure trade frequency vs win rate.
3. Decide long-only vs shorts.
4. Fix Polygon rate-limiting for reliable data.
5. Let winners run to the ~8.7-day backtest horizon (now that churn is fixed).
6. Only scale size after positive expectancy over ~30+ clean live trades.
