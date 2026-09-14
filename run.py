#!/usr/bin/env python3
"""Run after installation, or use start.sh/start.bat to install automatically."""
import os

# Small CPU linear algebra workloads often slow down with many BLAS threads.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")

try:
    from flytrade.cli import main
except ModuleNotFoundError as exc:
    raise SystemExit('Dependencies missing. Run start.sh/start.bat or: python -m pip install -e "."') from exc

if __name__ == "__main__":
    main()
