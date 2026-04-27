"""
Manually trigger any scheduler job from the command line.

Usage:
    python scripts/run_job.py data_sync
    python scripts/run_job.py signal_pipeline
    python scripts/run_job.py discovery_scan
    python scripts/run_job.py position_check
    python scripts/run_job.py drift_check
"""
import sys

JOBS = ["data_sync", "signal_pipeline", "discovery_scan", "position_check", "drift_check", "recalibrate"]

if len(sys.argv) < 2 or sys.argv[1] not in JOBS:
    print(f"Usage: python scripts/run_job.py <job>")
    print(f"Available jobs: {', '.join(JOBS)}")
    sys.exit(1)

job_name = sys.argv[1]

from src.scheduler.scheduler import TradingScheduler

sched = TradingScheduler()
job_fn = getattr(sched, f"job_{job_name}")
print(f"Running {job_name}...")
job_fn()
print(f"Done.")
