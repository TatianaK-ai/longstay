"""Longstay — shelter length-of-stay prediction.

We predict DURATION — how long an animal waits — not outcome type, which is
the usual thing people build on this dataset. The primary model asks whether
an animal will still be here in 90 days; the day estimate is secondary and
scores worse than a constant. See CLAUDE.md.
"""

__version__ = "0.1.0"
