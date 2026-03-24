from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import requests
from sgp4.api import Satrec


CELESTRAK_STARLINK_URL = "https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=tle"


class TLELoadError(RuntimeError):
    pass


def _parse_tle_text(tle_text: str) -> List[Dict[str, str]]:
    lines = [line.strip() for line in tle_text.splitlines() if line.strip()]
    records: List[Dict[str, str]] = []

    i = 0
    while i + 2 < len(lines):
        name = lines[i]
        line1 = lines[i + 1]
        line2 = lines[i + 2]
        if line1.startswith("1 ") and line2.startswith("2 "):
            records.append({"name": name, "line1": line1, "line2": line2})
            i += 3
        else:
            i += 1
    return records


def fetch_starlink_tles(limit: int = 100, cache_path: str = "data/starlink.tle") -> List[Dict[str, str]]:
    cache_file = Path(cache_path)
    cache_file.parent.mkdir(parents=True, exist_ok=True)

    tle_text = None
    try:
        response = requests.get(CELESTRAK_STARLINK_URL, timeout=20)
        response.raise_for_status()
        tle_text = response.text
        cache_file.write_text(tle_text, encoding="utf-8")
    except requests.RequestException:
        if cache_file.exists():
            tle_text = cache_file.read_text(encoding="utf-8")
        else:
            raise TLELoadError(
                "Failed to fetch Starlink TLE data from Celestrak and no cache is available."
            )

    records = _parse_tle_text(tle_text)
    if len(records) < limit:
        raise TLELoadError(
            f"Only found {len(records)} Starlink TLE records, but {limit} were requested."
        )

    return records[:limit]


def build_satrecs(tle_records: List[Dict[str, str]]) -> List[Satrec]:
    return [Satrec.twoline2rv(rec["line1"], rec["line2"]) for rec in tle_records]
