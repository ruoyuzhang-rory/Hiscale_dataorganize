"""Build one netCDF per flight leg from raw HI-SCALE files.

Usage:
  python build_flight_nc.py --flight 20160425a
  python build_flight_nc.py --all
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import io_readers as io  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]  # SGP_HISCALE_2.0/
INDEX = ROOT / "organized" / "catalogs" / "flight_index.csv"
OUT_DIR = ROOT / "organized" / "flight_days"
PROCESSING_VERSION = "v1"

# CCN columns A and B -> stack into a 2D var (time, ccn_col).
# Output var name -> (column A source, column B source).
CCN_PAIRS = {
    "ccn_dT": ("DT_A", "DT_B"),
    "ccn_ss": ("SS_A", "SS_B"),
    "ccn_conc": ("CCN_Conc_A", "CCN_Conc_B"),
    "ccn_conc_isok": ("CCN_Conc_A_Isok", "CCN_Conc_B_Isok"),
}


def _open_for(row: pd.Series, col: str):
    folder = ROOT / io.FOLDERS[col]
    fname = row.get(col, None)
    if pd.isna(fname) or not fname:
        return None
    path = folder / str(fname)
    if not path.exists():
        print(f"  [warn] file not found: {path}", file=sys.stderr)
        return None
    return io.READERS[col](path)


def _restructure_ccn(ds: xr.Dataset) -> xr.Dataset:
    """Fold A/B column pairs into 2D (time, ccn_col) variables."""
    if ds is None:
        return None
    new = xr.Dataset(coords={"time": ds["time"]})
    new = new.assign_coords({"ccn_col": (("ccn_col",), np.array(["A", "B"]))})

    for out_name, (a, b) in CCN_PAIRS.items():
        if a in ds and b in ds:
            stacked = np.stack([ds[a].values, ds[b].values], axis=1)
            new[out_name] = (("time", "ccn_col"), stacked,
                             {"units": ds[a].attrs.get("units", ""),
                              "long_name": ds[a].attrs.get("long_name", out_name)})

    # Carry the scalar ambient P, T, flag, isokinetic P/T if present.
    for src, dst in [("Flag", "ccn_flag"),
                     ("P_mbar", "ccn_P_mbar"),
                     ("T_C", "ccn_T_C"),
                     ("IsokP_mbar", "ccn_isokP_mbar"),
                     ("IsokT_C", "ccn_isokT_C")]:
        if src in ds:
            new[dst] = ds[src]
    return new


def build_one(flight_id: str, index: pd.DataFrame) -> Path:
    rows = index[index.flight_id == flight_id]
    if len(rows) != 1:
        raise SystemExit(f"flight_id {flight_id!r} not unique in index")
    row = rows.iloc[0]
    print(f"[{flight_id}] building...")

    sources: dict[str, xr.Dataset] = {}
    missing: list[str] = []
    for col in io.READERS:
        ds = _open_for(row, col)
        if ds is None:
            missing.append(col.replace("_file", ""))
            continue
        if col == "ccn_file":
            ds = _restructure_ccn(ds)
        sources[col] = ds

    if not sources:
        raise SystemExit(f"[{flight_id}] no readable source files")

    # Build common 1 s time grid spanning union of all sources.
    t_min = min(int(s["time"].values.min()) for s in sources.values())
    t_max = max(int(s["time"].values.max()) for s in sources.values())
    grid = np.arange(t_min, t_max + 1, dtype=np.int64)

    aligned = []
    for col, ds in sources.items():
        # ensure no duplicate times
        ds = ds.drop_duplicates("time").reindex(time=grid)
        aligned.append(ds)

    merged = xr.merge(aligned, compat="override", join="exact")

    # Convert time coord (seconds since midnight) to a CF datetime by attaching
    # a units attribute referencing the flight date midnight.
    date_str = row["date"]
    merged["time"].attrs["units"] = f"seconds since {date_str} 00:00:00"
    merged["time"].attrs["calendar"] = "proleptic_gregorian"
    merged["time"].attrs["long_name"] = "time"

    # Global attributes
    src_files = {col.replace("_file", ""): str(row.get(col, "")) for col in io.READERS
                 if isinstance(row.get(col, None), str) and row.get(col)}
    merged.attrs.update(
        flight_id=flight_id,
        date=date_str,
        leg=row["leg"],
        iop=int(row["iop"]),
        takeoff_utc_s=int(row["takeoff_utc_s"]) if not pd.isna(row["takeoff_utc_s"]) else -1,
        location="SGP, Bartlesville, OK",
        platform="ARM G-1 (N701BN)",
        campaign="HI-SCALE 2016",
        processing_version=PROCESSING_VERSION,
        created_utc=datetime.utcnow().isoformat(timespec="seconds") + "Z",
        missing_instruments=",".join(missing) if missing else "(none)",
        source_files="; ".join(f"{k}={v}" for k, v in src_files.items()),
        pi_contacts=("FIMS: Jian Wang (BNL); AIMMS: John Hubbe (PNNL); "
                     "CCN/CPC/CPC-CVI/Merged: Fan Mei (PNNL); CDP: Alyssa Matthews (PNNL)"),
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"hiscale_g1_{flight_id}.nc"

    # encode time as int seconds (the attr "units" will let xarray decode on read)
    encoding = {"time": {"dtype": "i4", "_FillValue": None}}
    for v in merged.data_vars:
        encoding[v] = {"zlib": True, "complevel": 4}

    merged.to_netcdf(out, encoding=encoding)
    print(f"[{flight_id}] wrote {out}  ({len(grid)} time samples, "
          f"{len(merged.data_vars)} vars, missing={missing or 'none'})")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--flight", help="flight_id (e.g. 20160425a)")
    p.add_argument("--all", action="store_true", help="process every leg in the index")
    args = p.parse_args()

    if not (args.flight or args.all):
        p.error("specify --flight FLIGHT_ID or --all")

    idx = pd.read_csv(INDEX)
    if args.all:
        for fid in idx.flight_id:
            build_one(fid, idx)
    else:
        build_one(args.flight, idx)


if __name__ == "__main__":
    main()
