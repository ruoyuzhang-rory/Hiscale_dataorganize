"""Compute analysis-ready ACI variables on a built flight netCDF.

Adds:
  Nd_cdp, LWC_cdp, MVD_cdp                                  - CDP-derived
  Nd_fcdp, LWC_fcdp, MVD_fcdp                               - FCDP-via-merged
  in_cloud_flag, in_cloud_flag_fcdp                         - cloud masks
  cvi_active_flag                                           - CVI sampling mask
  N_cn_stp, N_ccn_ssA_stp, N_ccn_ssB_stp                    - STP-normalized
  act_frac_ssA, act_frac_ssB                                - CCN/CN in clear air
  w_updraft                                                 - alias for w_w

Re-writes the netCDF in place via atomic temp-file swap.

Usage:
  python derive_aci.py --flight 20160425a
  python derive_aci.py --all
  # optional thresholds:
  python derive_aci.py --flight 20160425a --nd-threshold 10 --lwc-threshold 0.01
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "organized" / "flight_days"
INDEX = ROOT / "organized" / "catalogs" / "flight_index.csv"

# Physical constants
RHO_WATER_G_CM3 = 1.0        # g cm^-3
P_STP_PA = 101325.0          # Pa
T_STP_K = 273.15             # K
DROPLET_RANGE_UM = (2.0, 50.0)  # window for Nd / LWC integration


# -------- spectral integrals --------

def _cdp_integrals(ds: xr.Dataset):
    """Return (Nd_cdp [cm^-3], LWC_cdp [g/m^3], MVD_cdp [um]) from CDP nbin (#/L per bin)."""
    if "nbin_cdp" not in ds:
        return None, None, None
    nbin = ds["nbin_cdp"].values  # (time, dp) in #/L per bin
    dp = ds["dp_cdp"].values      # midpoint, um

    # Nd in cm^-3:  #/L -> #/cm^3 is /1000.
    Nd_per_L = np.nansum(nbin, axis=1)
    Nd = Nd_per_L / 1000.0  # cm^-3

    # LWC [g/m^3]:
    # mass_per_droplet = (pi/6) * Dp^3 [um^3] * 1e-12 cm^3/um^3 * 1 g/cm^3
    # nbin [#/L] -> nbin * 1000 [#/m^3]; total mass = sum_bins (nbin * 1000 * (pi/6) * Dp^3 * 1e-12)
    #             = (pi/6) * 1e-9 * sum(nbin * Dp^3)
    coeff_lwc = (np.pi / 6.0) * 1e-9 * RHO_WATER_G_CM3
    LWC = coeff_lwc * np.nansum(nbin * dp[np.newaxis, :] ** 3, axis=1)  # g/m^3

    # MVD: cumulative volume distribution; find Dp where cum reaches half of total volume.
    vol = nbin * dp[np.newaxis, :] ** 3  # proportional to volume in each bin
    cum = np.cumsum(np.nan_to_num(vol), axis=1)
    total = cum[:, -1]
    half = total / 2.0
    MVD = np.full(Nd.shape, np.nan)
    valid = total > 0
    for i in np.where(valid)[0]:
        # locate first bin where cum >= half
        c = cum[i]
        k = np.searchsorted(c, half[i])
        k = min(max(k, 0), len(dp) - 1)
        if k == 0:
            MVD[i] = dp[0]
        else:
            # linear interp on cum-vs-Dp
            x0, x1 = dp[k - 1], dp[k]
            y0, y1 = c[k - 1], c[k]
            if y1 > y0:
                MVD[i] = x0 + (half[i] - y0) * (x1 - x0) / (y1 - y0)
            else:
                MVD[i] = dp[k]
    return Nd, LWC, MVD


def _fcdp_integrals(ds: xr.Dataset, dp_lo: float, dp_hi: float):
    """Integrate the merged dN/dDp (cm^-3 um^-1) over [dp_lo, dp_hi] um using
    partial-bin contributions for bins that straddle the window edges.

    This avoids spuriously including sub-cloud-droplet aerosol from bin 1 of the
    merged product (1.5-3 um), which has midpoint 2.25 um but extends to 1.5 um.
    """
    if "n_dp_merged" not in ds:
        return None, None, None
    dp_mid = ds["dp_merged"].values
    lo = ds["dp_merged_lower"].values
    hi = ds["dp_merged_upper"].values
    # effective overlap of each bin with the integration window
    eff_lo = np.maximum(lo, dp_lo)
    eff_hi = np.minimum(hi, dp_hi)
    overlap = np.clip(eff_hi - eff_lo, 0.0, None)
    if not (overlap > 0).any():
        return None, None, None

    n = ds["n_dp_merged"].values   # L^-1 um^-1 (see io_readers note)
    contrib = n * overlap[np.newaxis, :]  # L^-1 per bin within window
    Nd_per_L = np.nansum(contrib, axis=1)
    Nd = Nd_per_L / 1000.0  # cm^-3

    # Effective Dp for the LWC moment in each bin: use midpoint of overlap segment.
    eff_mid = np.where(overlap > 0, 0.5 * (eff_lo + eff_hi), dp_mid)
    # LWC: contrib [#/L] * Dp^3 [um^3] * 1e-12 [cm^3/um^3] * rho_w [g/cm^3] * 1000 [L/m^3]
    coeff = (np.pi / 6.0) * 1e-9 * RHO_WATER_G_CM3
    LWC = coeff * np.nansum(contrib * eff_mid[np.newaxis, :] ** 3, axis=1)

    # MVD over the same partial-bin contributions
    active = overlap > 0
    Dp_sub = eff_mid[active]
    contrib_sub = contrib[:, active]
    vol = contrib_sub * Dp_sub[np.newaxis, :] ** 3
    cum = np.cumsum(np.nan_to_num(vol), axis=1)
    total = cum[:, -1]
    half = total / 2.0
    MVD = np.full(Nd.shape, np.nan)
    valid = total > 0
    for i in np.where(valid)[0]:
        c = cum[i]
        k = np.searchsorted(c, half[i])
        k = min(max(k, 0), len(Dp_sub) - 1)
        if k == 0:
            MVD[i] = Dp_sub[0]
        else:
            x0, x1 = Dp_sub[k - 1], Dp_sub[k]
            y0, y1 = c[k - 1], c[k]
            if y1 > y0:
                MVD[i] = x0 + (half[i] - y0) * (x1 - x0) / (y1 - y0)
            else:
                MVD[i] = Dp_sub[k]
    return Nd, LWC, MVD


# -------- flags and masking --------

def _in_cloud(Nd, LWC, nd_thr, lwc_thr):
    if Nd is None or LWC is None:
        return None
    flag = np.zeros(Nd.shape, dtype=np.int8)
    in_cloud = (Nd > nd_thr) & (LWC > lwc_thr)
    flag[in_cloud] = 1
    # mark NaN-rows as -1 ("unknown")
    bad = ~(np.isfinite(Nd) & np.isfinite(LWC))
    flag[bad] = -1
    return flag


def _cvi_active(ds: xr.Dataset):
    if "cvi_mode" not in ds:
        return None
    flag = np.zeros(ds.sizes["time"], dtype=np.int8)
    cvi = ds["cvi_mode"].values
    flag[cvi > 0] = 1
    flag[~np.isfinite(cvi)] = -1
    return flag


def _stp_correct(n_ambient, P_amb_pa, T_amb_k):
    """Convert ambient-air concentration to STP using P/T from AIMMS."""
    if n_ambient is None:
        return None
    ratio = (P_STP_PA / P_amb_pa) * (T_amb_k / T_STP_K)
    return n_ambient * ratio


# -------- main derive routine --------

def derive(flight_id: str, nd_thr: float, lwc_thr: float) -> Path:
    path = OUT_DIR / f"hiscale_g1_{flight_id}.nc"
    if not path.exists():
        raise SystemExit(f"netCDF not found: {path}")
    # load fully so we can rewrite the file
    ds = xr.load_dataset(path)
    n = ds.sizes["time"]

    added: dict[str, xr.DataArray] = {}

    # ---- CDP integrals ----
    Nd_cdp, LWC_cdp, MVD_cdp = _cdp_integrals(ds)
    if Nd_cdp is not None:
        added["Nd_cdp"] = xr.DataArray(
            Nd_cdp, dims="time",
            attrs={"units": "cm-3", "long_name": "CDP cloud droplet number concentration",
                   "provenance": "sum(nbin_cdp)/1000"})
        added["LWC_cdp"] = xr.DataArray(
            LWC_cdp, dims="time",
            attrs={"units": "g m-3", "long_name": "CDP liquid water content",
                   "provenance": "(pi/6) * rho_w * sum(nbin_cdp * Dp^3); ground-truth bin convention"})
        added["MVD_cdp"] = xr.DataArray(
            MVD_cdp, dims="time",
            attrs={"units": "um", "long_name": "CDP median volume diameter"})

    # ---- merged (FCDP-derived) integrals ----
    Nd_f, LWC_f, MVD_f = _fcdp_integrals(ds, *DROPLET_RANGE_UM)
    if Nd_f is not None:
        added["Nd_fcdp"] = xr.DataArray(
            Nd_f, dims="time",
            attrs={"units": "cm-3",
                   "long_name": f"Merged-product droplet number concentration, {DROPLET_RANGE_UM[0]}-{DROPLET_RANGE_UM[1]} um",
                   "provenance": "sum(n_dp_merged * dDp) over [2,50] um"})
        added["LWC_fcdp"] = xr.DataArray(
            LWC_f, dims="time",
            attrs={"units": "g m-3", "long_name": "Merged-product liquid water content",
                   "provenance": "(pi/6) * rho_w * sum(n_dp_merged * dDp * Dp^3) over [2,50] um"})
        added["MVD_fcdp"] = xr.DataArray(
            MVD_f, dims="time",
            attrs={"units": "um", "long_name": "Merged-product median volume diameter"})

    # ---- in-cloud flags ----
    f_cdp = _in_cloud(Nd_cdp, LWC_cdp, nd_thr, lwc_thr)
    if f_cdp is not None:
        added["in_cloud_flag"] = xr.DataArray(
            f_cdp, dims="time",
            attrs={"long_name": "In-cloud flag from CDP (1=cloud, 0=clear, -1=unknown)",
                   "flag_values": np.array([-1, 0, 1], dtype=np.int8),
                   "flag_meanings": "unknown clear cloud",
                   "nd_threshold_cm-3": nd_thr,
                   "lwc_threshold_g_m-3": lwc_thr,
                   "criterion": "Nd_cdp > nd_thr AND LWC_cdp > lwc_thr"})

    f_fcdp = _in_cloud(Nd_f, LWC_f, nd_thr, lwc_thr)
    if f_fcdp is not None:
        added["in_cloud_flag_fcdp"] = xr.DataArray(
            f_fcdp, dims="time",
            attrs={"long_name": "In-cloud flag from merged FCDP (1=cloud, 0=clear, -1=unknown)",
                   "flag_values": np.array([-1, 0, 1], dtype=np.int8),
                   "flag_meanings": "unknown clear cloud",
                   "nd_threshold_cm-3": nd_thr,
                   "lwc_threshold_g_m-3": lwc_thr,
                   "criterion": "Nd_fcdp > nd_thr AND LWC_fcdp > lwc_thr"})

    # ---- CVI sampling flag ----
    f_cvi = _cvi_active(ds)
    if f_cvi is not None:
        added["cvi_active_flag"] = xr.DataArray(
            f_cvi, dims="time",
            attrs={"long_name": "CVI cloud-residual sampling active (1=CVI, 0=isokinetic, -1=unknown)",
                   "flag_values": np.array([-1, 0, 1], dtype=np.int8),
                   "flag_meanings": "unknown isokinetic cvi"})

    # ---- STP-corrected aerosol concentrations (clear-air only) ----
    # NOTE: CPC iso-kinetic and CCN counter use separate inlets from the CVI,
    # so cvi_active_flag does NOT affect their sampling. We mask only on
    # in_cloud_flag to avoid cloud-shattering artefacts on the iso inlet.
    if {"cpc3010", "P_amb", "T_amb"} <= set(ds.data_vars):
        P_amb_pa = ds["P_amb"].values
        T_amb_k = ds["T_amb"].values + 273.15

        cn_stp = _stp_correct(ds["cpc3010"].values, P_amb_pa, T_amb_k)
        clear = (f_cdp == 0) if f_cdp is not None else np.ones(n, dtype=bool)
        cn_amb = np.where(clear, cn_stp, np.nan)
        added["N_cn_stp"] = xr.DataArray(
            cn_amb, dims="time",
            attrs={"units": "cm-3 STP",
                   "long_name": "CPC 3010 condensation nuclei at STP, clear-air",
                   "provenance": "cpc3010 * (P_stp/P_amb) * (T_amb/T_stp); masked to in_cloud_flag==0 only (iso-CPC inlet is independent of CVI state)"})

        if "ccn_conc_isok" in ds:
            ccn_iso = ds["ccn_conc_isok"].values  # (time, ccn_col)
            for i, label in enumerate(["A", "B"]):
                ccn_stp = _stp_correct(ccn_iso[:, i], P_amb_pa, T_amb_k)
                ccn_amb = np.where(clear, ccn_stp, np.nan)
                added[f"N_ccn_ss{label}_stp"] = xr.DataArray(
                    ccn_amb, dims="time",
                    attrs={"units": "cm-3 STP",
                           "long_name": f"CCN column {label} concentration at STP, clear-air",
                           "provenance": f"ccn_conc_isok[:, {i}] * STP-correction; masked to in_cloud_flag==0 only (CCN inlet is independent of CVI state)"})
                # activation fraction
                with np.errstate(divide="ignore", invalid="ignore"):
                    actf = np.where(cn_amb > 0, ccn_amb / cn_amb, np.nan)
                added[f"act_frac_ss{label}"] = xr.DataArray(
                    actf, dims="time",
                    attrs={"units": "1",
                           "long_name": f"Activation fraction CCN/CN at SS column {label}, clear-air",
                           "provenance": f"N_ccn_ss{label}_stp / N_cn_stp"})

    # ---- updraft alias ----
    if "w_w" in ds:
        added["w_updraft"] = xr.DataArray(
            ds["w_w"].values, dims="time",
            attrs={"units": "m s-1", "long_name": "Vertical wind (alias of AIMMS w_w; positive up)",
                   "provenance": "alias of w_w"})

    # Merge in and write back atomically.
    for name, da in added.items():
        ds[name] = da

    ds.attrs["derive_aci_version"] = "v1"
    ds.attrs["derive_aci_nd_threshold_cm-3"] = nd_thr
    ds.attrs["derive_aci_lwc_threshold_g_m-3"] = lwc_thr

    tmp = path.with_suffix(".nc.tmp")
    encoding = {"time": {"dtype": "i4", "_FillValue": None}}
    for v in ds.data_vars:
        encoding[v] = {"zlib": True, "complevel": 4}
    ds.to_netcdf(tmp, encoding=encoding)
    ds.close()
    os.replace(tmp, path)
    print(f"[{flight_id}] derived {len(added)} variables -> {path.name}")
    return path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--flight", help="flight_id (e.g. 20160425a)")
    p.add_argument("--all", action="store_true")
    p.add_argument("--nd-threshold", type=float, default=10.0,
                   help="Nd > THIS (cm^-3) for in-cloud (default 10)")
    p.add_argument("--lwc-threshold", type=float, default=0.01,
                   help="LWC > THIS (g/m^3) for in-cloud (default 0.01)")
    args = p.parse_args()
    if not (args.flight or args.all):
        p.error("specify --flight FLIGHT_ID or --all")

    if args.all:
        idx = pd.read_csv(INDEX)
        for fid in idx.flight_id:
            try:
                derive(fid, args.nd_threshold, args.lwc_threshold)
            except SystemExit as e:
                print(f"  [skip] {fid}: {e}", file=sys.stderr)
    else:
        derive(args.flight, args.nd_threshold, args.lwc_threshold)


if __name__ == "__main__":
    main()
