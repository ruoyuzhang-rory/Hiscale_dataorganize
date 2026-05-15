# HI-SCALE 2016 organized-flight netCDF schema

One netCDF per flight leg lives at `organized/flight_days/hiscale_g1_<flight_id>.nc`.
Files are CF-compliant; `time` decodes to `datetime64[ns]` via xarray.

## Dimensions

| Name           | Size  | Description                                                              |
|----------------|-------|--------------------------------------------------------------------------|
| `time`         | varies| 1 Hz UTC; spans the union of all available instruments for this leg       |
| `ccn_col`      | 2     | CCN columns A and B (different supersaturations per second)              |
| `dp_cdp`       | 30    | CDP bin midpoint diameters (µm)                                          |
| `dp_fims_bin`  | 30    | FIMS bin index (1–30); real Dp from cal table — not stored in file       |
| `dp_merged`    | 86    | Merged FCDP+2DS+HVPS bin midpoint diameters (µm)                         |

## Coordinates

`time` (CF), `dp_cdp`, `dp_merged`, plus auxiliary `dp_*_lower` / `dp_*_upper` edges.

## Raw data variables (per instrument)

### AIMMS (1 Hz downsampled from 20 Hz native)
`T_amb` (°C), `RH` (%RH/100), `P_amb` (Pa), `u_w` `v_w` `w_w` (m/s), `lat` `lon` (deg),
`alt` (m), `u_aircraft` `v_aircraft` `w_aircraft` (m/s), `roll` `pitch` `yaw` (deg),
`TAS` (m/s), `w_w_std_1s` (std of vertical wind within each 1 s bin, m/s),
`Dimensional_Sideslip_Angle`, `Non_dimensional_Angle_of_Attack`, `Non_dimensional_Sideslip_Angle`,
`aimms_wind_flag`.

### CCN dual-column
`ccn_dT(time, ccn_col)` (°C), `ccn_ss(time, ccn_col)` (% SS), `ccn_conc(time, ccn_col)` (#/cc),
`ccn_conc_isok(time, ccn_col)` (#/cc), `ccn_flag`,
`ccn_P_mbar`, `ccn_T_C`, `ccn_isokP_mbar`, `ccn_isokT_C`.

### CPC (isokinetic)
`cpc3010`, `cpc3025` (#/cc), `cpc_flag`, `cpc_isokP_mbar`, `cpc_isokT_C`.

### CPC-CVI (cloud-residual aerosol)
`cpc_cvi_n`, `cpc_cvi_n_uk` (#/cc), `cpc_cvi_e_factor`, `cpc_cvi_d_factor`, `cvi_mode`.

### CDP-2000 cloud droplet probe
`Nconc_cdp_total` (#/L total).
`nbin_cdp(time, dp_cdp)` (#/L per bin — ground-truth bin convention; the source file's
`#/L/µm` label is misleading for the wider 14–50 µm bins).
`n_dp_cdp(time, dp_cdp)` (#/L/µm, derived as nbin/dDp for properly-normalized spectra).

### FIMS aerosol PSD
`n_dp_fims(time, dp_fims_bin)` (cm⁻³, dN/dlnDp; bins 1–30 cover 10–400 nm — see
README PDF for the diameter table).
`P_amb_fims` (atm), `T_amb_fims` (°C).

### Merged FCDP + 2DS + HVPS hydrometeor PSD
`n_dp_merged(time, dp_merged)` (L⁻¹ µm⁻¹, dN/dDp; units confirmed by cross-comparison
to CDP — file labels are absent / ambiguous).
`Ntotal_merged` (L⁻¹).

## Derived ACI variables (from `derive_aci.py`)

| Variable              | Units      | Definition                                                                  |
|-----------------------|------------|-----------------------------------------------------------------------------|
| `Nd_cdp`              | cm⁻³       | Σ `nbin_cdp` / 1000 (full CDP range 2–50 µm)                                |
| `LWC_cdp`             | g m⁻³      | (π/6)·ρ_w·10⁻⁹ · Σ `nbin_cdp` · Dp³                                         |
| `MVD_cdp`             | µm         | Median volume diameter from cumulative volume spectrum                      |
| `Nd_fcdp`             | cm⁻³       | (Σ `n_dp_merged` · dDp) / 1000, partial-bin overlap with 2–50 µm window     |
| `LWC_fcdp`            | g m⁻³      | (π/6)·ρ_w·10⁻⁹ · Σ contrib · Dp³  over 2–50 µm                              |
| `MVD_fcdp`            | µm         | Same as MVD_cdp but over the merged-spectrum 2–50 µm window                 |
| `in_cloud_flag`       | (flag)     | 1=cloud, 0=clear, -1=unknown — CDP: `Nd>10 AND LWC>0.01`                    |
| `in_cloud_flag_fcdp`  | (flag)     | Same as above but using `Nd_fcdp` and `LWC_fcdp`                            |
| `cvi_active_flag`     | (flag)     | 1 where `cvi_mode>0` (CVI cloud-residual sampling active)                   |
| `N_cn_stp`            | cm⁻³ STP   | `cpc3010 * (P_stp/P_amb) * (T_amb/T_stp)`, masked to `in_cloud_flag==0`     |
| `N_ccn_ss{A,B}_stp`   | cm⁻³ STP   | `ccn_conc_isok[:, {0,1}] * STP-correction`, masked to clear-air             |
| `act_frac_ss{A,B}`    | 1          | `N_ccn_ssX_stp / N_cn_stp`                                                   |

The CPC iso-kinetic inlet and the CCN counter's DMT constant-pressure inlet are
both **independent of the CVI**, so `cvi_active_flag` is NOT in the mask for
these variables. The only mask is `in_cloud_flag == 0` (to avoid cloud-droplet
shattering on the iso inlet). `cvi_active_flag` should only be used when
interpreting `cpc_cvi_n` (which represents cloud-droplet-residual concentration
only when CVI is active).
| `w_updraft`           | m s⁻¹      | Alias of `w_w`                                                              |

In-cloud thresholds and SI constants:

- Default `Nd > 10 cm⁻³` AND `LWC > 0.01 g m⁻³` for in-cloud (tunable via `--nd-threshold` / `--lwc-threshold`).
- STP = 101325 Pa, 273.15 K.
- ρ_w = 1.0 g cm⁻³.

## Global attributes

`flight_id`, `date`, `leg`, `iop`, `takeoff_utc_s`, `location`, `platform`, `campaign`,
`processing_version`, `derive_aci_version`, `derive_aci_nd_threshold_cm-3`,
`derive_aci_lwc_threshold_g_m-3`, `created_utc`, `missing_instruments`, `source_files`,
`pi_contacts`.

## Quirks discovered while building

1. **CDP bin units mislabelled.** Header says `#/L/µm` but values are actually `#/L per bin`. Verified by summing all bins and matching the reported `Conc(#/L)` total to floating-point precision. The 1 µm-wide bins are numerically identical under both interpretations; the wider 2 µm bins (>=14 µm) revealed the convention.
2. **Merged file units inferred.** The README doesn't state units. Cross-comparison with CDP confirms `Ntotal_merged` is in `L⁻¹`, not `cm⁻³`. Documented in the variable `note` attribute.
3. **Filename leg conventions differ.** FIMS uses `_L1` / `_L2`; AIMMS / CCN / CPC / CPC-CVI / CDP use full `YYYYMMDDhhmmss`; merged uses `YYYYMMDD{a,b}`. Reconciled in `flight_index.csv`.
4. **Some FIMS files are not UTF-8** (degree symbol as Latin-1 byte 0xb0). Handled by the robust opener in `io_readers._open_icartt_robust`.
5. **Some files have duplicated ICARTT headers** (e.g. `CPC_G1_20160830183214`). The robust opener keeps everything from the last `<n>,1001` marker onward.
6. **Anomalous single-second outliers** exist (e.g. CDP on 2016-04-25 at 18:13:49 reports 8.87e10 #/L — clearly a glitch). We do not silently scrub raw data; users should mask via flags or by min/max thresholds in their own analysis.
