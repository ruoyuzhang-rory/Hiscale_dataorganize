"""Build organized/catalogs/flight_index.csv from raw HI-SCALE filenames.

Crosswalks three filename conventions used in the campaign:
  AIMMS/CCN/CPC/CPC_CVI/CDP -> *_YYYYMMDDhhmmss_*.ict
  FIMS                      -> *_YYYYMMDD[_L1|_L2]_*.txt
  Merged                    -> aaf.g1.hiscale.mergedSD.YYYYMMDD[a|b].txt

Each row in the output is one flight leg, keyed by canonical flight_id YYYYMMDD{a,b,...}.
Legs are assigned per date by start-time order (a = earliest, b = next, ...).
Missing instruments for a leg get an empty cell.
"""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # SGP_HISCALE_2.0/
OUT = ROOT / "organized" / "catalogs" / "flight_index.csv"

# Folder -> (column_name, regex pulling start datetime, kind)
TIMESTAMP_SOURCES = {
    "hubbe-aimms":   ("aimms_file",   re.compile(r"_G1_(\d{14})_"), "datetime14"),
    "mei-ccn":       ("ccn_file",     re.compile(r"_G1_(\d{14})_"), "datetime14"),
    "mei-cpc":       ("cpc_file",     re.compile(r"_G1_(\d{14})_"), "datetime14"),
    "mei-cpc_cvi":   ("cpc_cvi_file", re.compile(r"_G1_(\d{14})_"), "datetime14"),
    "matthews-cdp":  ("cdp_file",     re.compile(r"_G1_(\d{14})_"), "datetime14"),
}
# FIMS: YYYYMMDD with revision tag _R\d+ and optional _L1/_L2 (L1->'a', L2->'b').
FIMS_RE = re.compile(r"FIMS_G1_(\d{8})_R\d+(?:_L(\d))?_")
# Merged: YYYYMMDD with trailing letter a|b
MERGED_RE = re.compile(r"mergedSD\.(\d{8})([a-z])\.txt$")

IOP_RANGES = {1: ("2016-04-25", "2016-05-20"), 2: ("2016-08-29", "2016-09-22")}


def iop_for(date_str: str) -> int:
    d = datetime.strptime(date_str, "%Y-%m-%d").date()
    for iop, (lo, hi) in IOP_RANGES.items():
        if datetime.strptime(lo, "%Y-%m-%d").date() <= d <= datetime.strptime(hi, "%Y-%m-%d").date():
            return iop
    return 0


def scan_timestamp_folder(folder: Path, regex: re.Pattern, suffix_filter: str | None = None):
    """Return list of (date_str, start_seconds_of_day, filename)."""
    out = []
    if not folder.exists():
        return out
    for p in sorted(folder.iterdir()):
        if not p.is_file():
            continue
        if suffix_filter and not p.name.lower().endswith(suffix_filter):
            continue
        m = regex.search(p.name)
        if not m:
            continue
        ts = m.group(1)
        dt = datetime.strptime(ts, "%Y%m%d%H%M%S")
        date_str = dt.strftime("%Y-%m-%d")
        seconds = dt.hour * 3600 + dt.minute * 60 + dt.second
        out.append((date_str, seconds, p.name))
    return out


def scan_fims(folder: Path):
    """Return list of (date_str, leg_letter_or_None, filename).

    FIMS legs: _L1 -> 'a', _L2 -> 'b'. No suffix means the only leg that day,
    which will be reconciled against other instruments later.
    """
    out = []
    if not folder.exists():
        return out
    for p in sorted(folder.iterdir()):
        if not p.is_file():
            continue
        m = FIMS_RE.search(p.name)
        if not m:
            continue
        date_str = datetime.strptime(m.group(1), "%Y%m%d").strftime("%Y-%m-%d")
        leg_num = m.group(2)
        leg_letter = {None: None, "1": "a", "2": "b"}.get(leg_num)
        out.append((date_str, leg_letter, p.name))
    return out


def scan_merged(folder: Path):
    """Return list of (date_str, leg_letter, filename)."""
    out = []
    if not folder.exists():
        return out
    # merged files live in HiScale_2016_merged/
    sub = folder / "HiScale_2016_merged"
    if not sub.exists():
        return out
    for p in sorted(sub.iterdir()):
        if not p.is_file():
            continue
        m = MERGED_RE.search(p.name)
        if not m:
            continue
        date_str = datetime.strptime(m.group(1), "%Y%m%d").strftime("%Y-%m-%d")
        out.append((date_str, m.group(2), p.name))
    return out


def build_index():
    # Step 1: collect timestamp-style entries per folder; assign leg letters per date by sort order.
    # date -> list of (start_seconds, folder_col_name, filename)
    timestamped: dict[str, list[tuple[int, str, str]]] = defaultdict(list)
    for folder_name, (col, regex, _) in TIMESTAMP_SOURCES.items():
        entries = scan_timestamp_folder(ROOT / folder_name, regex)
        for date_str, secs, fname in entries:
            timestamped[date_str].append((secs, col, fname))

    # Step 2: discover canonical (date, leg) keys by clustering distinct start times per date
    # within a tolerance (small differences across instruments on the same takeoff).
    TOLERANCE_S = 600  # 10 min: well below any IOP gap, well above instrument start jitter
    legs_per_date: dict[str, list[int]] = {}  # date -> sorted list of canonical start_seconds (one per leg)
    for date_str, entries in timestamped.items():
        starts = sorted({e[0] for e in entries})
        clusters: list[list[int]] = []
        for s in starts:
            if clusters and s - clusters[-1][-1] <= TOLERANCE_S:
                clusters[-1].append(s)
            else:
                clusters.append([s])
        legs_per_date[date_str] = [min(c) for c in clusters]

    # Step 3: build rows.
    rows: dict[tuple[str, str], dict] = {}  # (date, leg_letter) -> row dict

    def leg_letter_for(date_str: str, secs: int) -> str:
        canonicals = legs_per_date[date_str]
        # nearest canonical within tolerance
        best = min(canonicals, key=lambda c: abs(c - secs))
        idx = canonicals.index(best)
        return chr(ord("a") + idx)

    for date_str, entries in timestamped.items():
        for secs, col, fname in entries:
            leg = leg_letter_for(date_str, secs)
            key = (date_str, leg)
            row = rows.setdefault(key, _blank_row(date_str, leg))
            row[col] = fname
            # record earliest takeoff time across instruments for this leg
            if row["takeoff_utc_s"] == "" or secs < int(row["takeoff_utc_s"]):
                row["takeoff_utc_s"] = secs

    # Step 4: FIMS — attach to existing (date, leg) keys when possible.
    fims = scan_fims(ROOT / "FIMS")
    fims_by_date: dict[str, list[tuple[str | None, str]]] = defaultdict(list)
    for d, leg_letter, fname in fims:
        fims_by_date[d].append((leg_letter, fname))
    for d, items in fims_by_date.items():
        existing_legs = sorted({lg for (dd, lg) in rows if dd == d})
        if len(items) == 1 and items[0][0] is None:
            # Single FIMS file with no L-suffix: attach to leg 'a' if it exists, else create.
            target_leg = existing_legs[0] if existing_legs else "a"
            key = (d, target_leg)
            row = rows.setdefault(key, _blank_row(d, target_leg))
            row["fims_file"] = items[0][1]
        else:
            for leg_letter, fname in items:
                # explicit _L1 -> 'a', _L2 -> 'b'; None -> 'a'
                lg = leg_letter or "a"
                key = (d, lg)
                row = rows.setdefault(key, _blank_row(d, lg))
                row["fims_file"] = fname

    # Step 5: merged — letter already canonical (a/b in filename).
    merged = scan_merged(ROOT / "mei-merged")
    for d, leg, fname in merged:
        key = (d, leg)
        row = rows.setdefault(key, _blank_row(d, leg))
        row["merged_file"] = fname

    # Step 6: fill flight_id and iop, then write sorted by (date, leg).
    sorted_keys = sorted(rows.keys())
    fields = [
        "flight_id", "date", "leg", "iop", "takeoff_utc_s",
        "aimms_file", "ccn_file", "cpc_file", "cpc_cvi_file",
        "cdp_file", "fims_file", "merged_file",
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for key in sorted_keys:
            r = rows[key]
            r["flight_id"] = r["date"].replace("-", "") + r["leg"]
            r["iop"] = iop_for(r["date"])
            w.writerow({k: r.get(k, "") for k in fields})
    print(f"Wrote {len(sorted_keys)} legs to {OUT}")
    return OUT


def _blank_row(date_str: str, leg: str) -> dict:
    return {
        "flight_id": "",
        "date": date_str,
        "leg": leg,
        "iop": "",
        "takeoff_utc_s": "",
        "aimms_file": "",
        "ccn_file": "",
        "cpc_file": "",
        "cpc_cvi_file": "",
        "cdp_file": "",
        "fims_file": "",
        "merged_file": "",
    }


if __name__ == "__main__":
    build_index()
