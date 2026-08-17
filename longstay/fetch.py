"""Download the Austin Animal Center intake and outcome tables from Socrata.

The API is public, anonymous, and rate limited. We page through it, cache the
raw result to CSV, and do not re-download unless explicitly forced. Running
`fetch` many times during development should cost the city nothing.
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests

from . import config


class FetchError(RuntimeError):
    """Raised when the API could not be read after exhausting retries."""


# A trailing UTC offset or "Z" on an ISO timestamp.
_TZ_SUFFIX = re.compile(r"(?:Z|[+-]\d{2}:?\d{2})\s*$")


def parse_socrata_datetime(values: pd.Series) -> pd.Series:
    """Parse Socrata timestamps to naive Austin local time.

    The feed is not internally consistent. Most rows are naive local
    wall-clock ("2013-10-01T07:51:00.000"), but a minority — 3,584 outcome
    rows at the time of writing, all at exactly midnight — carry an explicit
    "-05:00" offset. Both are the same thing: the local time the shelter
    wrote down.

    So we strip the offset and keep the wall clock. Parsing with `utc=True`
    instead would convert only the offset-bearing minority, shifting them five
    hours and making their length of stay inconsistent with everyone else's
    for no reason the data supports.
    """
    text = values.astype("string").str.strip()
    return pd.to_datetime(
        text.str.replace(_TZ_SUFFIX, "", regex=True),
        format="ISO8601",
        errors="coerce",
    )


def count_offset_timestamps(values: pd.Series) -> int:
    """How many raw timestamps carried an explicit UTC offset."""
    text = values.astype("string").str.strip()
    return int(text.str.contains(_TZ_SUFFIX, regex=True, na=False).sum())


def _sleep_seconds(attempt: int, retry_after: str | None) -> float:
    """Backoff duration for a given attempt, honouring Retry-After if sent."""
    if retry_after:
        try:
            return min(float(retry_after), config.BACKOFF_MAX_SECONDS)
        except ValueError:
            pass
    return min(
        config.BACKOFF_BASE_SECONDS * (2 ** attempt),
        config.BACKOFF_MAX_SECONDS,
    )


def _get_page(url: str, offset: int, session: requests.Session) -> list[dict]:
    """Fetch one page, retrying with exponential backoff on rate limits."""
    params = {
        "$limit": config.PAGE_SIZE,
        "$offset": offset,
        "$order": config.SOCRATA_ORDER,
    }

    last_error: str = "no attempt made"

    for attempt in range(config.MAX_RETRIES):
        retry_after: str | None = None

        try:
            response = session.get(
                url, params=params, timeout=config.REQUEST_TIMEOUT_SECONDS
            )
        except requests.RequestException as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        else:
            if response.status_code == 200:
                return response.json()

            # 429 = rate limited, 5xx = transient server trouble. Both retry.
            if response.status_code == 429 or response.status_code >= 500:
                last_error = f"HTTP {response.status_code}"
                retry_after = response.headers.get("Retry-After")
            else:
                raise FetchError(
                    f"{url} returned HTTP {response.status_code}: "
                    f"{response.text[:200]}"
                )

        delay = _sleep_seconds(attempt, retry_after)
        print(
            f"    retry {attempt + 1}/{config.MAX_RETRIES} after {last_error} "
            f"— waiting {delay:.0f}s",
            flush=True,
        )
        time.sleep(delay)

    raise FetchError(
        f"Gave up on {url} at offset {offset} after "
        f"{config.MAX_RETRIES} attempts. Last error: {last_error}"
    )


def _download(url: str, label: str) -> pd.DataFrame:
    """Page through a Socrata endpoint until it stops returning rows."""
    print(f"  downloading {label} from {url}", flush=True)

    pages: list[pd.DataFrame] = []
    total = 0
    offset = 0
    page_number = 0

    with requests.Session() as session:
        session.headers.update({"User-Agent": "longstay/0.1 (coursework)"})

        while True:
            page_number += 1
            rows = _get_page(url, offset, session)
            if not rows:
                break

            pages.append(pd.DataFrame(rows))
            total += len(rows)
            print(
                f"    page {page_number:>2}: +{len(rows):>6,} rows "
                f"({total:,} total)",
                flush=True,
            )

            if len(rows) < config.PAGE_SIZE:
                break  # short page means we reached the end
            offset += config.PAGE_SIZE

    if not pages:
        raise FetchError(f"{label}: API returned no rows at all")

    return pd.concat(pages, ignore_index=True)


def _report(frame: pd.DataFrame, label: str, path: Path) -> None:
    """Print row count and the actual date range retrieved."""
    stamps = parse_socrata_datetime(frame["datetime"])
    valid = stamps.notna().sum()
    offsets = count_offset_timestamps(frame["datetime"])

    print(f"  {label}: {len(frame):,} rows -> {path}")
    print(
        f"    date range: {stamps.min():%Y-%m-%d} to {stamps.max():%Y-%m-%d}"
        f"  ({valid:,} parseable datetimes)"
    )
    if offsets:
        print(
            f"    {offsets:,} timestamps carried an explicit UTC offset; "
            "offset stripped, local wall clock kept"
        )
    print(f"    columns: {', '.join(frame.columns)}")


def fetch(force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download (or load from cache) the intakes and outcomes tables."""
    config.ensure_dirs()

    targets = [
        (config.INTAKES_URL, config.INTAKES_CSV, "intakes"),
        (config.OUTCOMES_URL, config.OUTCOMES_CSV, "outcomes"),
    ]

    frames: list[pd.DataFrame] = []

    for url, path, label in targets:
        if path.exists() and not force:
            frame = read_raw(path)
            print(f"  {label}: cached — {len(frame):,} rows at {path}")
            print("    (pass --force to re-download)")
        else:
            frame = _download(url, label)
            frame.to_csv(path, index=False)
            _report(frame, label, path)

        frames.append(frame)

    return frames[0], frames[1]


def read_raw(path: Path) -> pd.DataFrame:
    """Read a cached raw CSV without pandas guessing at types.

    Everything stays text until clean.py parses it deliberately.
    `keep_default_na=False` matters: an animal genuinely named "NA" must not
    silently become a null and flip `has_name`.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python main.py fetch` first."
        )
    return pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
        na_values=[""],
        low_memory=False,
    )


if __name__ == "__main__":  # pragma: no cover
    fetch(force="--force" in sys.argv)
