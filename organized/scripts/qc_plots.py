"""Per-flight quicklook PNG: altitude, Nd, LWC, CCN/CN, w, in-cloud mask.

Usage:
  python qc_plots.py --flight 20160425a
  python qc_plots.py --all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

ROOT = Path(__file__).resolve().parents[2]
NC_DIR = ROOT / "organized" / "flight_days"
PNG_DIR = ROOT / "organized" / "flight_days" / "quicklook"
INDEX = ROOT / "organized" / "catalogs" / "flight_index.csv"


def plot_one(flight_id: str) -> Path:
    nc = NC_DIR / f"hiscale_g1_{flight_id}.nc"
    if not nc.exists():
        raise SystemExit(f"missing: {nc}")
    ds = xr.open_dataset(nc)
    t = ds.time.values

    fig, axes = plt.subplots(6, 1, figsize=(12, 14), sharex=True)
    title = f"HI-SCALE 2016 G-1 quicklook — flight {flight_id}  ({ds.attrs.get('date','?')}, IOP{ds.attrs.get('iop','?')})"
    fig.suptitle(title, y=0.995, fontsize=12)

    # 1) altitude
    ax = axes[0]
    if "alt" in ds:
        ax.plot(t, ds.alt.values, color="black", lw=0.8)
    ax.set_ylabel("Altitude (m)")
    ax.grid(alpha=0.3)
    # shade in-cloud
    if "in_cloud_flag" in ds:
        _shade_flag(ax, t, ds.in_cloud_flag.values == 1, color="lightblue", alpha=0.35, label="in cloud (CDP)")
    if "cvi_active_flag" in ds:
        _shade_flag(ax, t, ds.cvi_active_flag.values == 1, color="gold", alpha=0.35, label="CVI mode")
    _legend_top(ax)

    # 2) Nd (CDP + FCDP)
    ax = axes[1]
    if "Nd_cdp" in ds:
        ax.semilogy(t, _safe_log(ds.Nd_cdp.values), color="C0", lw=0.7, label="Nd_cdp")
    if "Nd_fcdp" in ds:
        ax.semilogy(t, _safe_log(ds.Nd_fcdp.values), color="C1", lw=0.7, label="Nd_fcdp")
    ax.set_ylabel("Nd (cm$^{-3}$)")
    ax.set_ylim(0.1, 5e3)
    ax.grid(alpha=0.3, which="both")
    _legend_top(ax)

    # 3) LWC (CDP + FCDP)
    ax = axes[2]
    if "LWC_cdp" in ds:
        ax.semilogy(t, _safe_log(ds.LWC_cdp.values), color="C0", lw=0.7, label="LWC_cdp")
    if "LWC_fcdp" in ds:
        ax.semilogy(t, _safe_log(ds.LWC_fcdp.values), color="C1", lw=0.7, label="LWC_fcdp")
    ax.set_ylabel("LWC (g m$^{-3}$)")
    ax.set_ylim(1e-4, 5)
    ax.grid(alpha=0.3, which="both")
    _legend_top(ax)

    # 4) CN and CCN
    ax = axes[3]
    if "N_cn_stp" in ds:
        ax.semilogy(t, _safe_log(ds.N_cn_stp.values), color="black", lw=0.7, label="CN (CPC 3010, STP)")
    if "N_ccn_ssA_stp" in ds:
        ax.semilogy(t, _safe_log(ds.N_ccn_ssA_stp.values), color="C2", lw=0.7, label="CCN ss-A")
    if "N_ccn_ssB_stp" in ds:
        ax.semilogy(t, _safe_log(ds.N_ccn_ssB_stp.values), color="C3", lw=0.7, label="CCN ss-B")
    ax.set_ylabel("Conc. (cm$^{-3}$ STP)")
    ax.grid(alpha=0.3, which="both")
    _legend_top(ax)

    # 5) Activation fraction
    ax = axes[4]
    if "act_frac_ssA" in ds:
        ax.plot(t, ds.act_frac_ssA.values, ".", ms=1.5, color="C2", label="act_frac ss-A")
    if "act_frac_ssB" in ds:
        ax.plot(t, ds.act_frac_ssB.values, ".", ms=1.5, color="C3", label="act_frac ss-B")
    ax.set_ylabel("CCN / CN")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    _legend_top(ax)

    # 6) vertical wind
    ax = axes[5]
    if "w_updraft" in ds:
        ax.plot(t, ds.w_updraft.values, color="gray", lw=0.5)
        ax.axhline(0, color="black", lw=0.4)
    ax.set_ylabel("w (m s$^{-1}$)")
    ax.set_ylim(-5, 5)
    ax.grid(alpha=0.3)
    ax.set_xlabel("UTC")

    PNG_DIR.mkdir(parents=True, exist_ok=True)
    out = PNG_DIR / f"hiscale_g1_{flight_id}.png"
    fig.autofmt_xdate()
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out, dpi=120)
    plt.close(fig)
    ds.close()
    print(f"[{flight_id}] wrote {out.name}")
    return out


def _safe_log(a):
    # log scale doesn't tolerate <=0; replace with tiny positive
    return np.where(np.isfinite(a) & (a > 0), a, np.nan)


def _shade_flag(ax, t, mask, **kw):
    if not mask.any():
        return
    # find runs of True
    runs = []
    i = 0
    while i < len(mask):
        if mask[i]:
            j = i
            while j < len(mask) and mask[j]:
                j += 1
            runs.append((i, j - 1))
            i = j
        else:
            i += 1
    for s, e in runs:
        ax.axvspan(t[s], t[e], color=kw.get("color", "lightblue"),
                   alpha=kw.get("alpha", 0.3), label=kw.get("label", None))
        kw["label"] = None  # only label once


def _legend_top(ax):
    h, l = ax.get_legend_handles_labels()
    # dedupe
    seen = {}
    for hi, li in zip(h, l):
        if li and li not in seen:
            seen[li] = hi
    if seen:
        ax.legend(seen.values(), seen.keys(), loc="upper right",
                  fontsize=8, framealpha=0.85, ncol=2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--flight", help="flight_id (e.g. 20160425a)")
    p.add_argument("--all", action="store_true")
    args = p.parse_args()
    if not (args.flight or args.all):
        p.error("specify --flight FLIGHT_ID or --all")

    if args.all:
        idx = pd.read_csv(INDEX)
        for fid in idx.flight_id:
            try:
                plot_one(fid)
            except SystemExit as e:
                print(f"  [skip] {fid}: {e}", file=sys.stderr)
    else:
        plot_one(args.flight)


if __name__ == "__main__":
    main()
