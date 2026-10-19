"""Shared pytest setup."""

import os

# The legacy model module imports pyplot at import time. Agg keeps that import
# headless on CI and stops a GUI backend from loading on a laptop.
os.environ.setdefault("MPLBACKEND", "Agg")
