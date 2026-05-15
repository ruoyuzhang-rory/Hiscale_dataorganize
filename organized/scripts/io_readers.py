"""Readers for HI-SCALE 2016 raw instrument files.

Each `read_*` function returns an `xarray.Dataset` with:
  - `time` coord: int seconds since midnight UTC (matches the ICARTT independent
    variable directly for 1 Hz files; AIMMS is downsampled to 1 Hz here)
  - data variables with sanitized netCDF-safe names
  - NaN substituted for the source missing value
  - `units` and `long_name` attributes on every data variable

Sources:
  AIMMS, CCN, CPC, CPC-CVI, CDP, FIMS  -> ICARTT 1001 (.ict / .txt)
  Merged                               -> plain CSV (.txt)
"""

from __future__ import annotations

import re
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

# icartt issues UserWarning for variable names with special chars (e.g. "Conc(#/L)").
# The data still loads correctly. Silence the warning class globally for this module.
warnings.filterwarnings("ignore", category=UserWarning, module=r"icartt(\.|$)")

import icartt  # noqa: E402


_HEADER_LINE1_RE = re.compile(r"^\s*(\d+)\s*,\s*1001\s*$")


def _read_text_lenient(path: Path) -> list[str]:
    with open(path, "rb") as f:
        raw = f.read()
    return raw.decode("utf-8", errors="replace").splitlines(keepends=True)


def _strip_duplicate_header(lines: list[str]) -> list[str] | None:
    """If the file has multiple '<n>,1001' markers (concatenated headers),
    keep everything from the LAST marker onward. Returns None if only one
    marker is found.
    """
    starts = [i for i, ln in enumerate(lines) if _HEADER_LINE1_RE.match(ln)]
    if len(starts) <= 1:
        return None
    return lines[starts[-1]:]


def _open_icartt_robust(path: Path) -> icartt.Dataset:
    """Wrapper around icartt.Dataset that tolerates non-UTF-8 bytes and
    duplicated ICARTT headers (both seen in the HI-SCALE 2016 data product)."""
    try:
        return icartt.Dataset(path)
    except (UnicodeDecodeError, IndexError):
        pass
    # Fallback: read leniently, optionally strip duplicate header, write tempfile.
    lines = _read_text_lenient(path)
    fixed = _strip_duplicate_header(lines)
    if fixed is not None:
        lines = fixed
    with tempfile.NamedTemporaryFile("w", suffix=".ict", delete=False,
                                     encoding="utf-8", newline="") as tmp:
        tmp.writelines(lines)
        tmp_path = tmp.name
    try:
        return icartt.Dataset(tmp_path)
    finally:
        try:
            Path(tmp_path).unlink()
        except OSError:
            pass


# ---------- utilities ----------

_NAME_FIX = re.compile(r"[^A-Za-z0-9_]+")


def _sanitize(name: str) -> str:
    """Convert an ICARTT variable shortname to a netCDF-safe identifier."""
    s = _NAME_FIX.sub("_", name).strip("_")
    if s and s[0].isdigit():
        s = "v_" + s
    return s


def _icartt_to_dataset(path: Path, rename: dict[str, str] | None = None) -> xr.Dataset:
    """Read an ICARTT 1001 file into an xr.Dataset.

    - Time variable -> int 'time' coord (seconds since midnight UTC).
    - Each other variable -> data variable with NaN-substituted missing values.
    - Variable names sanitized; `rename` allows explicit override (post-sanitize).
    """
    ds_src = _open_icartt_robust(path)
    arr = ds_src.data.data  # numpy structured array
    ivar = ds_src.data.ivarname

    # Time: round to int seconds (some files have sub-second resolution recorded
    # as float64; we coerce to 1 Hz integer for joining).
    t_raw = arr[ivar].astype(np.float64)
    t = np.rint(t_raw).astype(np.int64)

    # icartt stores varnames with spaces preserved in `varnames`, but the
    # structured-array dtype names have spaces replaced with underscores. Build
    # a mapping varname -> dtype-field-name for safe lookup.
    dtype_names = list(arr.dtype.names)
    by_underscored = {n.replace(" ", "_"): n for n in dtype_names}

    data_vars: dict[str, xr.DataArray] = {}
    for vname in ds_src.data.varnames:
        if vname == ivar:
            continue
        var_meta = ds_src.variables[vname]
        # Map source varname -> actual field in the structured array.
        field = vname if vname in dtype_names else by_underscored.get(vname.replace(" ", "_"))
        if field is None:
            continue
        col = arr[field].astype(np.float64).copy()
        col[col == var_meta.miss] = np.nan
        nc_name = _sanitize(vname)
        if rename and nc_name in rename:
            nc_name = rename[nc_name]
        da = xr.DataArray(
            col,
            dims=("time",),
            attrs={
                "units": str(var_meta.units),
                "long_name": str(var_meta.longname or var_meta.shortname),
                "icartt_shortname": vname,
            },
        )
        data_vars[nc_name] = da

    ds = xr.Dataset(data_vars, coords={"time": ("time", t)})
    ds["time"].attrs["units"] = "seconds since midnight UTC"
    ds["time"].attrs["long_name"] = "UTC time-of-day"

    # Average any duplicate timestamps (rare but possible after rounding).
    if not _is_strictly_increasing(ds["time"].values):
        ds = ds.groupby("time").mean(skipna=True)
    return ds


def _is_strictly_increasing(a: np.ndarray) -> bool:
    return bool(np.all(np.diff(a) > 0))


def _stack_bins(ds: xr.Dataset, var_pattern: re.Pattern, dp_dim: str,
                output_var: str, edge_extractor) -> xr.Dataset:
    """Collapse N per-bin scalar variables into a single 2D (time, dp) variable.

    `edge_extractor(name) -> (lower, upper)` parses bin edges from each var name.
    Removes the per-bin scalars; adds `output_var`, `<dp_dim>_lower`,
    `<dp_dim>_upper` and a midpoint coord `<dp_dim>`.
    """
    matches = [(v, var_pattern.fullmatch(v)) for v in ds.data_vars]
    bin_vars = [(v, m) for v, m in matches if m is not None]
    if not bin_vars:
        return ds

    edges = []
    for v, _ in bin_vars:
        lo, hi = edge_extractor(v)
        edges.append((lo, hi, v))
    edges.sort(key=lambda t: 0.5 * (t[0] + t[1]))

    lower = np.array([e[0] for e in edges])
    upper = np.array([e[1] for e in edges])
    mid = 0.5 * (lower + upper)
    stack = np.stack([ds[e[2]].values for e in edges], axis=1)  # (time, dp)

    # carry units from the first bin variable
    units = ds[edges[0][2]].attrs.get("units", "")

    dp_units = "um" if "um" in units.lower() else "nm"
    ds = ds.drop_vars([e[2] for e in edges])
    ds = ds.assign_coords({dp_dim: ((dp_dim,), mid)})
    ds[dp_dim].attrs.update(units=dp_units, long_name="bin midpoint diameter")
    ds[f"{dp_dim}_lower"] = ((dp_dim,), lower, {"units": dp_units, "long_name": "bin lower edge"})
    ds[f"{dp_dim}_upper"] = ((dp_dim,), upper, {"units": dp_units, "long_name": "bin upper edge"})
    ds[output_var] = (("time", dp_dim), stack, {"units": units,
                                                "long_name": "size distribution"})
    return ds


# ---------- per-instrument readers ----------

def read_aimms(path: Path) -> xr.Dataset:
    """20 Hz AIMMS -> 1 Hz (mean for all vars, plus std for w_w)."""
    ds = _icartt_to_dataset(path, rename={
        "Temperature": "T_amb",
        "Humidity": "RH",
        "Pressure": "P_amb",
        "u_w": "u_w",
        "v_w": "v_w",
        "w_w": "w_w",
        "Lat": "lat",
        "Lon": "lon",
        "Alt": "alt",
        "u_i": "u_aircraft",
        "v_i": "v_aircraft",
        "w_i": "w_aircraft",
        "Roll": "roll",
        "Pitch": "pitch",
        "Yaw": "yaw",
        "TAS": "TAS",
        "Wind_Status_Flag": "aimms_wind_flag",
    })
    # Build 1 s bins on the unique integer-second values present.
    # groupby('time') with the int-second coord gives 1 Hz means directly.
    if "w_w" in ds:
        w_std = ds["w_w"].groupby("time").std(skipna=True)
    ds_1hz = ds.groupby("time").mean(skipna=True)
    if "w_w" in ds_1hz:
        ds_1hz["w_w_std_1s"] = w_std
        ds_1hz["w_w_std_1s"].attrs.update(
            units="m/s",
            long_name="standard deviation of vertical wind within each 1 s bin",
        )
    return ds_1hz


def read_ccn(path: Path) -> xr.Dataset:
    return _icartt_to_dataset(path)


def read_cpc(path: Path) -> xr.Dataset:
    """Isokinetic-inlet CPC 3010 + 3025."""
    return _icartt_to_dataset(path, rename={
        "CPC_Conc_3010": "cpc3010",
        "CPC_Conc_3025": "cpc3025",
        "Flag_a": "cpc_flag",
        "IsokP_mbar": "cpc_isokP_mbar",
        "IsokT_C": "cpc_isokT_C",
    })


def read_cpc_cvi(path: Path) -> xr.Dataset:
    """CPC behind the CVI inlet (cloud-droplet residuals)."""
    return _icartt_to_dataset(path, rename={
        "N_CVI": "cpc_cvi_n",
        "E_factor": "cpc_cvi_e_factor",
        "CVI_mode": "cvi_mode",
        "N_uk": "cpc_cvi_n_uk",
        "D_factor": "cpc_cvi_d_factor",
    })


def read_cdp(path: Path) -> xr.Dataset:
    """CDP-2000 cloud droplet PSD, 2-50 um (30 bins).

    NOTE: The CDP file header advertises units '#/liter/um' but the per-bin
    columns are actually concentration *within* each bin in #/L (verified by
    summing all bins and matching the reported total). The narrow 1-um bins
    happen to give the same number either way, but the wider 2-um bins
    (>=14 um) reveal the convention. We expose the values as 'nbin_cdp' in
    #/L (concentration per bin) and additionally compute true dN/dDp.
    """
    ds = _icartt_to_dataset(path, rename={
        "Conc_L": "Nconc_cdp_total",  # 'Conc(#/L)' after _sanitize
    })

    bin_re = re.compile(r"C_(\d+)_(\d+)")

    def extract(name: str):
        m = bin_re.fullmatch(name)
        return (float(m.group(1)), float(m.group(2)))

    ds = _stack_bins(ds, bin_re, dp_dim="dp_cdp", output_var="nbin_cdp",
                     edge_extractor=extract)

    # Re-label units to reflect ground truth, and derive true dN/dDp.
    if "nbin_cdp" in ds:
        ds["nbin_cdp"].attrs.update(
            units="L^-1",
            long_name="CDP concentration per bin (in-bin droplet count per liter)",
        )
        width = (ds["dp_cdp_upper"] - ds["dp_cdp_lower"]).values  # um
        # broadcast dN/dDp = nbin / dDp
        dndDp = ds["nbin_cdp"].values / width[np.newaxis, :]
        ds["n_dp_cdp"] = (("time", "dp_cdp"), dndDp, {
            "units": "L^-1 um^-1",
            "long_name": "CDP cloud droplet size distribution dN/dDp",
            "note": "Derived from nbin_cdp / bin_width; original file column labelled #/L/um is misleading.",
        })
    if "Nconc_cdp_total" in ds:
        ds["Nconc_cdp_total"].attrs.setdefault("long_name", "CDP total droplet number concentration")
    return ds


def read_fims(path: Path) -> xr.Dataset:
    """FIMS aerosol PSD 10-400 nm (30 bins, dN/dlnDp in cm^-3) + P_amb, T_amb."""
    ds = _icartt_to_dataset(path, rename={
        "P_amb": "P_amb_fims",
        "T_amb": "T_amb_fims",
    })
    bin_re = re.compile(r"n_Dp_(\d+)")

    # FIMS bin edges are documented in the README PDF but not on each file; we
    # store the bin INDEX as the dp coord placeholder. A separate calibration
    # table can replace this later.
    def extract(name: str):
        m = bin_re.fullmatch(name)
        idx = int(m.group(1))
        return (float(idx), float(idx))

    ds = _stack_bins(ds, bin_re, dp_dim="dp_fims_bin", output_var="n_dp_fims",
                     edge_extractor=extract)
    return ds


def read_merged(path: Path) -> xr.Dataset:
    """Merged FCDP + 2DS + HVPS hydrometeor PSD CSV.

    Bin names look like 'dndDp_<lo>_<hi>' with 'p' as decimal point and Dp in um.
    Bin values are dN/dDp in cm^-3 um^-1 (verified by integrating to N_total).
    Missing value: -9999.
    First column is a human-readable timestamp; convert to seconds since midnight.
    """
    df = pd.read_csv(path, na_values=[-9999, -9999.0])
    tcol = df.columns[0]
    # parse '25-Apr-2016 15:58:10' -> seconds since midnight UTC
    t = pd.to_datetime(df[tcol], format="%d-%b-%Y %H:%M:%S", utc=True)
    secs = (t.dt.hour * 3600 + t.dt.minute * 60 + t.dt.second).astype(np.int64).values
    df = df.drop(columns=[tcol])

    # Total concentration is column N_total (cm^-3 per file convention).
    n_total = df.pop("N_total").values.astype(np.float64) if "N_total" in df.columns else None

    bin_re = re.compile(r"dndDp_(\d+(?:p\d+)?)_(\d+(?:p\d+)?)")
    bin_cols = [c for c in df.columns if bin_re.fullmatch(c)]
    edges = []
    for c in bin_cols:
        m = bin_re.fullmatch(c)
        lo = float(m.group(1).replace("p", "."))
        hi = float(m.group(2).replace("p", "."))
        edges.append((lo, hi, c))
    edges.sort(key=lambda x: 0.5 * (x[0] + x[1]))
    lower = np.array([e[0] for e in edges])
    upper = np.array([e[1] for e in edges])
    mid = 0.5 * (lower + upper)
    stack = np.stack([df[e[2]].values.astype(np.float64) for e in edges], axis=1)

    ds = xr.Dataset(
        {
            "n_dp_merged": (("time", "dp_merged"), stack,
                            {"units": "L^-1 um^-1",
                             "long_name": "merged hydrometeor size distribution dN/dDp",
                             "note": "Units inferred to be L^-1 um^-1 (not cm^-3) by cross-comparing in-cloud N_total with CDP (#/L) — see provenance."}),
            "dp_merged_lower": (("dp_merged",), lower, {"units": "um"}),
            "dp_merged_upper": (("dp_merged",), upper, {"units": "um"}),
        },
        coords={"time": ("time", secs),
                "dp_merged": (("dp_merged",), mid,
                              {"units": "um", "long_name": "bin midpoint diameter"})},
    )
    if n_total is not None:
        ds["Ntotal_merged"] = (("time",), n_total,
                              {"units": "L^-1", "long_name": "merged total number concentration"})

    ds["time"].attrs["units"] = "seconds since midnight UTC"

    # Handle any duplicate timestamps from sub-second rounding (none expected, but safe).
    if not _is_strictly_increasing(ds["time"].values):
        ds = ds.groupby("time").mean(skipna=True)
    return ds


READERS = {
    "aimms_file":   read_aimms,
    "ccn_file":     read_ccn,
    "cpc_file":     read_cpc,
    "cpc_cvi_file": read_cpc_cvi,
    "cdp_file":     read_cdp,
    "fims_file":    read_fims,
    "merged_file":  read_merged,
}


FOLDERS = {
    "aimms_file":   "hubbe-aimms",
    "ccn_file":     "mei-ccn",
    "cpc_file":     "mei-cpc",
    "cpc_cvi_file": "mei-cpc_cvi",
    "cdp_file":     "matthews-cdp",
    "fims_file":    "FIMS",
    "merged_file":  "mei-merged/HiScale_2016_merged",
}
