#!/usr/bin/env python3
"""Reproducible extreme-rainfall analysis for Dire Dawa (1981-2025).

The script reads the original CSVs and writes all new outputs to a separate
folder. It does not overwrite source data or the original analysis results.

Methods retained from the supplied notebook:
- GEV fitted to annual maxima using L-moments (lmoments3).
- POT threshold based on the 95th percentile of daily area-average rainfall.
- Runs declustering: exceedances separated by no more than 3 days belong to
  the same event; each event is represented by its peak.
- GPD fitted to event excesses with location fixed at zero.

Important improvement:
- POT return levels are calculated from the Poisson annual-maximum
  exceedance-probability relationship, so the return period has the same
  annual exceedance probability interpretation as the GEV return period.
- POT bootstrap intervals resample both event count (Poisson) and event
  excesses. They remain conditional on the selected threshold and run length.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from lmoments3 import distr, lmom_ratios

RETURN_PERIODS = np.array([2, 5, 10, 25, 50, 100], dtype=float)
POT_PERCENTILE = 95.0
RUN_LENGTH_DAYS = 3
N_BOOTSTRAP = 1000
RANDOM_SEED = 42


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("data"),
                        help="Folder containing the original CSV and GeoJSON files")
    parser.add_argument("--output-dir", type=Path, default=Path("revised_analysis"),
                        help="Separate folder for revised outputs")
    parser.add_argument("--bootstrap", type=int, default=N_BOOTSTRAP,
                        help="Number of bootstrap replicates (default: 1000)")
    return parser.parse_args()


def load_and_validate(input_dir: Path):
    daily_path = input_dir / "Dire_Dawa_CHIRPS_Daily_1981_2025.csv"
    annual_path = input_dir / "Dire_Dawa_Annual_Maximum_Rainfall_1981_2025.csv"
    boundary_path = input_dir / "00_Dire_Dawa_Boundary.geojson"

    for path in (daily_path, annual_path, boundary_path):
        if not path.exists():
            raise FileNotFoundError(f"Required input file not found: {path}")

    daily = pd.read_csv(daily_path, parse_dates=["date"])
    annual = pd.read_csv(annual_path)

    required_daily = {"date", "rainfall_mm", "year"}
    required_annual = {"year", "annual_max_rainfall_mm"}
    if not required_daily.issubset(daily.columns):
        raise ValueError(f"Daily CSV must contain columns: {sorted(required_daily)}")
    if not required_annual.issubset(annual.columns):
        raise ValueError(f"Annual CSV must contain columns: {sorted(required_annual)}")

    daily["rainfall_mm"] = pd.to_numeric(daily["rainfall_mm"], errors="coerce")
    daily["year"] = pd.to_numeric(daily["year"], errors="coerce")
    if daily[["date", "rainfall_mm", "year"]].isna().any().any():
        raise ValueError("Daily data contain missing or invalid date/rainfall/year values.")
    if daily["date"].duplicated().any():
        raise ValueError("Daily data contain duplicate dates.")
    if (daily["rainfall_mm"] < 0).any():
        raise ValueError("Daily data contain negative rainfall values.")

    daily = daily.sort_values("date").reset_index(drop=True)
    expected_dates = pd.date_range("1981-01-01", "2025-12-31", freq="D")
    missing_dates = expected_dates.difference(pd.DatetimeIndex(daily["date"]))
    extra_dates = pd.DatetimeIndex(daily["date"]).difference(expected_dates)
    if len(missing_dates) or len(extra_dates):
        raise ValueError(
            f"Daily date coverage is not complete: {len(missing_dates)} missing and "
            f"{len(extra_dates)} dates outside 1981-2025."
        )
    if not (daily["year"].astype(int).to_numpy() == daily["date"].dt.year.to_numpy()).all():
        raise ValueError("The year column does not consistently match the date column.")

    annual = annual.sort_values("year").reset_index(drop=True)
    annual["year"] = pd.to_numeric(annual["year"], errors="coerce").astype(int)
    annual["annual_max_rainfall_mm"] = pd.to_numeric(
        annual["annual_max_rainfall_mm"], errors="coerce"
    )
    recalculated = daily.groupby(daily["date"].dt.year)["rainfall_mm"].max()
    supplied = annual.set_index("year")["annual_max_rainfall_mm"]
    common_years = supplied.index.intersection(recalculated.index)
    if len(common_years) != 45 or not np.allclose(
        supplied.loc[common_years].to_numpy(),
        recalculated.loc[common_years].to_numpy(),
        rtol=0, atol=1e-8,
    ):
        raise ValueError("Supplied annual maxima do not match maxima recalculated from daily data.")

    with boundary_path.open(encoding="utf-8") as f:
        boundary = json.load(f)
    if boundary.get("type") not in {"Feature", "FeatureCollection"}:
        raise ValueError("Boundary GeoJSON is not a Feature or FeatureCollection.")

    validation = {
        "daily_records": int(len(daily)),
        "period_start": str(daily["date"].min().date()),
        "period_end": str(daily["date"].max().date()),
        "missing_dates": int(len(missing_dates)),
        "duplicate_dates": int(daily["date"].duplicated().sum()),
        "negative_rainfall_values": int((daily["rainfall_mm"] < 0).sum()),
        "missing_daily_values": int(daily[["date", "rainfall_mm", "year"]].isna().sum().sum()),
        "annual_maximum_records": int(len(annual)),
        "annual_maxima_match_daily_recalculation": True,
        "maximum_annual_maximum_difference_mm": float(
            np.max(np.abs(supplied.loc[common_years].to_numpy() -
                          recalculated.loc[common_years].to_numpy()))
        ),
        "boundary_geojson_type": boundary.get("type"),
    }
    return daily, annual, validation


def decluster_events(values, dates, threshold, run_length_days=3):
    """Return peak excesses for clusters of threshold exceedances."""
    exceedance_indices = np.flatnonzero(values > threshold)
    if len(exceedance_indices) == 0:
        return pd.DataFrame(columns=["date", "peak_rainfall_mm", "excess_mm"])

    clusters = []
    current = [int(exceedance_indices[0])]
    for idx in exceedance_indices[1:]:
        gap = (dates[idx] - dates[current[-1]]) / np.timedelta64(1, "D")
        if gap <= run_length_days:
            current.append(int(idx))
        else:
            clusters.append(current)
            current = [int(idx)]
    clusters.append(current)

    records = []
    for cluster in clusters:
        peak_idx = cluster[int(np.argmax(values[cluster]))]
        records.append({
            "date": pd.Timestamp(dates[peak_idx]),
            "peak_rainfall_mm": float(values[peak_idx]),
            "excess_mm": float(values[peak_idx] - threshold),
        })
    return pd.DataFrame(records)


def pot_return_level(T, threshold, shape, scale, event_rate):
    """POT level for annual exceedance probability 1/T under a Poisson model."""
    if T <= 1 or event_rate <= 0 or scale <= 0:
        raise ValueError("T must exceed 1 and event rate/scale must be positive.")
    annual_hazard = -np.log1p(-1.0 / T)
    tail_probability = annual_hazard / event_rate
    if tail_probability >= 1:
        # The requested annual exceedance probability is at/above the
        # exceedance rate of the threshold; the corresponding level is <= u.
        return float(threshold)
    if abs(shape) < 1e-8:
        return float(threshold - scale * np.log(tail_probability))
    support_term = tail_probability ** (-shape)
    return float(threshold + (scale / shape) * (support_term - 1.0))


def save_plot(path: Path):
    plt.tight_layout()
    plt.savefig(path, dpi=250, bbox_inches="tight")
    plt.close()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    daily, annual, validation = load_and_validate(args.input_dir)
    out = args.output_dir

    # Observation duration uses elapsed calendar days, not record count / 365.
    duration_years = ((daily.date.max() - daily.date.min()).days + 1) / 365.2425
    values = daily["rainfall_mm"].to_numpy(dtype=float)
    dates = daily["date"].to_numpy()

    # GEV: retain the original notebook's L-moment estimator.
    amax = annual["annual_max_rainfall_mm"].to_numpy(dtype=float)
    if len(amax) < 20 or not np.isfinite(amax).all():
        raise ValueError("Annual maxima are insufficient or contain invalid values.")
    sample_lmoments = lmom_ratios(amax, nmom=4)
    gev_params = distr.gev.lmom_fit(data=amax)
    gev_levels = np.asarray(distr.gev.ppf(1 - 1 / RETURN_PERIODS, **gev_params), dtype=float)
    if not np.isfinite(gev_levels).all():
        raise RuntimeError("GEV return-level calculation produced non-finite values.")

    gev_summary = pd.DataFrame([{
        "Distribution": "GEV_L_moments",
        "c_shape": float(gev_params["c"]),
        "loc_location": float(gev_params["loc"]),
        "scale": float(gev_params["scale"]),
        "L1": float(sample_lmoments[0]),
        "L2": float(sample_lmoments[1]),
        "L3": float(sample_lmoments[2]),
        "L4": float(sample_lmoments[3]),
    }])
    gev_summary.to_csv(out / "03b_GEV_Parameters_Summary_REVISED.csv", index=False)
    gev_results = pd.DataFrame({
        "Return_Period_years": RETURN_PERIODS.astype(int),
        "GEV_Return_Level_mm": gev_levels,
    })
    gev_results.to_csv(out / "03_GEV_Return_Levels_REVISED.csv", index=False)

    # POT: retain 95th percentile and 3-day runs rule from the notebook.
    threshold = float(np.percentile(values, POT_PERCENTILE))
    events = decluster_events(values, dates, threshold, RUN_LENGTH_DAYS)
    if len(events) < 20:
        raise ValueError("Too few independent POT events; review threshold/declustering.")
    excesses = events["excess_mm"].to_numpy(dtype=float)
    shape, loc, scale = stats.genpareto.fit(excesses, floc=0)
    event_rate = len(events) / duration_years
    pot_levels = np.array([
        pot_return_level(T, threshold, shape, scale, event_rate)
        for T in RETURN_PERIODS
    ])
    gpd_summary = pd.DataFrame([{
        "Distribution": "GPD_MLE",
        "c_shape": float(shape),
        "loc_location": float(loc),
        "scale": float(scale),
        "Threshold_u": threshold,
        "Threshold_percentile": POT_PERCENTILE,
        "Run_length_days": RUN_LENGTH_DAYS,
        "Independent_events": int(len(events)),
        "Observation_duration_years": float(duration_years),
        "Event_Rate_per_year": float(event_rate),
        "KS_statistic_descriptive_only": float(stats.kstest(
            excesses, "genpareto", args=(shape, 0, scale)
        ).statistic),
        "KS_p_value_descriptive_only": float(stats.kstest(
            excesses, "genpareto", args=(shape, 0, scale)
        ).pvalue),
    }])
    gpd_summary.to_csv(out / "08b_GPD_Parameters_Summary_REVISED.csv", index=False)
    events.to_csv(out / "POT_Independent_Events_REVISED.csv", index=False)
    pot_results = pd.DataFrame({
        "Return_Period_years": RETURN_PERIODS.astype(int),
        "POT_Return_Level_mm": pot_levels,
    })
    pot_results.to_csv(out / "08_POT_Return_Levels_REVISED.csv", index=False)

    # Bootstrap. GEV resamples annual maxima. POT also resamples event count
    # from a Poisson distribution and then resamples event excesses. Threshold
    # and run length are held fixed, so POT CIs are conditional on these choices.
    rng = np.random.default_rng(RANDOM_SEED)
    gev_boot = []
    for _ in range(args.bootstrap):
        sample = rng.choice(amax, size=len(amax), replace=True)
        try:
            p = distr.gev.lmom_fit(data=sample)
            levels = np.asarray(distr.gev.ppf(1 - 1 / RETURN_PERIODS, **p), dtype=float)
            if np.isfinite(levels).all():
                gev_boot.append(levels)
        except Exception:
            continue
    if len(gev_boot) < max(100, int(args.bootstrap * 0.5)):
        raise RuntimeError("Too few successful GEV bootstrap replicates.")
    gev_boot = np.asarray(gev_boot)
    gev_results["GEV_Lower_95"] = np.percentile(gev_boot, 2.5, axis=0)
    gev_results["GEV_Upper_95"] = np.percentile(gev_boot, 97.5, axis=0)
    gev_results.to_csv(out / "06_GEV_Return_Levels_With_CI_REVISED.csv", index=False)

    pot_boot = []
    for _ in range(args.bootstrap):
        n_events = int(rng.poisson(len(excesses)))
        if n_events < 20:
            continue
        sample = rng.choice(excesses, size=n_events, replace=True)
        try:
            sh, _, sc = stats.genpareto.fit(sample, floc=0)
            boot_rate = n_events / duration_years
            levels = np.array([
                pot_return_level(T, threshold, sh, sc, boot_rate)
                for T in RETURN_PERIODS
            ])
            if np.isfinite(levels).all():
                pot_boot.append(levels)
        except Exception:
            continue
    if len(pot_boot) < max(100, int(args.bootstrap * 0.5)):
        raise RuntimeError("Too few successful POT bootstrap replicates.")
    pot_boot = np.asarray(pot_boot)
    pot_results["POT_Lower_95"] = np.percentile(pot_boot, 2.5, axis=0)
    pot_results["POT_Upper_95"] = np.percentile(pot_boot, 97.5, axis=0)
    pot_results.to_csv(out / "12_POT_Return_Levels_With_CI_REVISED.csv", index=False)

    comparison = pd.DataFrame({
        "Return_Period_years": RETURN_PERIODS.astype(int),
        "GEV_Return_Level_mm": gev_levels,
        "POT_Return_Level_mm": pot_levels,
    })
    comparison["POT_minus_GEV_mm"] = comparison.POT_Return_Level_mm - comparison.GEV_Return_Level_mm
    comparison["POT_minus_GEV_percent_of_GEV"] = (
        comparison.POT_minus_GEV_mm / comparison.GEV_Return_Level_mm * 100
    )
    comparison.to_csv(out / "13_GEV_POT_Comparison_REVISED.csv", index=False)

    # Threshold sensitivity using the same declustering and Poisson return-level definition.
    sensitivity_rows = []
    for percentile in [90, 92.5, 95, 97, 97.5, 98]:
        u = float(np.percentile(values, percentile))
        ev = decluster_events(values, dates, u, RUN_LENGTH_DAYS)
        if len(ev) < 20:
            continue
        exc = ev.excess_mm.to_numpy(dtype=float)
        sh, _, sc = stats.genpareto.fit(exc, floc=0)
        rate = len(ev) / duration_years
        row = {
            "Threshold_percentile": percentile,
            "Threshold_mm": u,
            "Number_of_independent_events": len(ev),
            "GPD_shape": sh,
            "GPD_scale": sc,
            "Event_rate_per_year": rate,
        }
        for T in [10, 25, 50, 100]:
            row[f"RL_{T}_years_mm"] = pot_return_level(T, u, sh, sc, rate)
        sensitivity_rows.append(row)
    pd.DataFrame(sensitivity_rows).to_csv(
        out / "11_POT_Threshold_Sensitivity_REVISED.csv", index=False
    )

    # Save validated daily and annual data copies in the new output directory.
    daily.to_csv(out / "Dire_Dawa_CHIRPS_Daily_1981_2025_VALIDATED.csv", index=False)
    annual.to_csv(out / "Dire_Dawa_Annual_Maximum_Rainfall_1981_2025_VALIDATED.csv", index=False)

    # Figures
    plt.figure(figsize=(13, 4))
    plt.plot(daily.date, daily.rainfall_mm, linewidth=0.45)
    plt.xlabel("Date"); plt.ylabel("Daily area-average rainfall (mm)")
    plt.title("Dire Dawa daily area-average CHIRPS rainfall, 1981–2025")
    plt.grid(alpha=0.25); save_plot(out / "01_Daily_Rainfall_REVISED.png")

    plt.figure(figsize=(11, 4))
    plt.bar(annual.year, annual.annual_max_rainfall_mm, width=0.8)
    plt.xlabel("Year"); plt.ylabel("Annual maximum daily rainfall (mm)")
    plt.title("Annual maximum daily area-average rainfall")
    plt.grid(axis="y", alpha=0.25); save_plot(out / "02_Annual_Maximum_Rainfall_REVISED.png")

    x = np.linspace(max(0, amax.min() * 0.8), amax.max() * 1.25, 300)
    plt.figure(figsize=(7, 5))
    plt.hist(amax, bins=10, density=True, alpha=0.5, label="Observed annual maxima")
    plt.plot(x, distr.gev.pdf(x, **gev_params), label="L-moment GEV fit")
    plt.xlabel("Annual maximum rainfall (mm)"); plt.ylabel("Density")
    plt.title("GEV distribution fit"); plt.legend(); plt.grid(alpha=0.25)
    save_plot(out / "05_GEV_Distribution_Fit_REVISED.png")

    probs = (np.arange(1, len(amax) + 1) - 0.5) / len(amax)
    tq = distr.gev.ppf(probs, **gev_params)
    plt.figure(figsize=(6, 6)); plt.scatter(tq, np.sort(amax))
    loq = min(tq.min(), amax.min()); hiq = max(tq.max(), amax.max())
    plt.plot([loq, hiq], [loq, hiq], "k--")
    plt.xlabel("Theoretical GEV quantiles (mm)"); plt.ylabel("Observed annual maxima (mm)")
    plt.title("GEV Q-Q diagnostic"); plt.grid(alpha=0.25)
    save_plot(out / "04_GEV_QQ_Plot_REVISED.png")

    plt.figure(figsize=(8, 5))
    plt.errorbar(RETURN_PERIODS, gev_levels,
                 yerr=[gev_levels - gev_results.GEV_Lower_95,
                       gev_results.GEV_Upper_95 - gev_levels], fmt="o-", capsize=3)
    plt.xscale("log"); plt.xlabel("Return period (years)")
    plt.ylabel("Return level (mm)"); plt.title("GEV return levels with 95% bootstrap intervals")
    plt.grid(alpha=0.25); save_plot(out / "07_GEV_Return_Level_Uncertainty_REVISED.png")

    plt.figure(figsize=(6, 6))
    pp = (np.arange(1, len(excesses) + 1) - 0.5) / len(excesses)
    theo = stats.genpareto.ppf(pp, c=shape, loc=0, scale=scale)
    plt.scatter(theo, np.sort(excesses)); qlo=min(theo.min(), excesses.min()); qhi=max(theo.max(), excesses.max())
    plt.plot([qlo, qhi], [qlo, qhi], "k--")
    plt.xlabel("Theoretical GPD excess (mm)"); plt.ylabel("Observed event excess (mm)")
    plt.title("GPD Q-Q diagnostic"); plt.grid(alpha=0.25)
    save_plot(out / "09_GPD_QQ_Plot_REVISED.png")

    mrl_thresholds = np.percentile(values, [85, 87.5, 90, 92.5, 95, 97, 98])
    mrl = [np.mean(values[values > u] - u) for u in mrl_thresholds]
    plt.figure(figsize=(8, 5)); plt.plot(mrl_thresholds, mrl, marker="o")
    plt.xlabel("Threshold (mm)"); plt.ylabel("Mean excess (mm)")
    plt.title("Mean residual life diagnostic (daily observations)"); plt.grid(alpha=0.25)
    save_plot(out / "10_Mean_Residual_Life_REVISED.png")

    plt.figure(figsize=(8, 5))
    plt.plot(RETURN_PERIODS, gev_levels, marker="o", label="GEV annual maxima")
    plt.plot(RETURN_PERIODS, pot_levels, marker="s", label="POT-GPD Poisson")
    plt.xscale("log"); plt.xlabel("Return period (years)"); plt.ylabel("Return level (mm)")
    plt.title("GEV and POT return levels on a common annual-exceedance basis")
    plt.legend(); plt.grid(alpha=0.25); save_plot(out / "14_GEV_POT_Comparison_REVISED.png")

    report = {
        "validation": validation,
        "daily_rainfall_mean_mm": float(daily.rainfall_mm.mean()),
        "daily_rainfall_max_mm": float(daily.rainfall_mm.max()),
        "daily_rainfall_max_date": str(daily.loc[daily.rainfall_mm.idxmax(), "date"].date()),
        "mean_annual_maximum_mm": float(annual.annual_max_rainfall_mm.mean()),
        "observation_duration_years": float(duration_years),
        "gev_parameters": {k: float(v) for k, v in gev_params.items()},
        "pot_threshold_mm": threshold,
        "pot_threshold_percentile": POT_PERCENTILE,
        "pot_run_length_days": RUN_LENGTH_DAYS,
        "pot_independent_events": int(len(events)),
        "pot_event_rate_per_year": float(event_rate),
        "gpd_parameters": {"shape": float(shape), "location": float(loc), "scale": float(scale)},
        "bootstrap_replicates_requested": int(args.bootstrap),
        "gev_bootstrap_successful": int(len(gev_boot)),
        "pot_bootstrap_successful": int(len(pot_boot)),
        "pot_confidence_interval_note": (
            "Bootstrap resamples event count and event excesses; intervals remain conditional "
            "on the selected 95th-percentile threshold and 3-day declustering rule."
        ),
        "pot_return_period_definition": (
            "Poisson annual-maximum exceedance probability equals 1/T; this is aligned with "
            "the GEV annual exceedance probability definition."
        ),
        "spatial_gev_warning": (
            "Do not interpret the notebook's spatial GEV proxy map as a fitted return-level "
            "surface. It uses a fixed shape and a scale assumed to be 20% of mean annual maxima, "
            "rather than pixel-wise fitted GEV parameters."
        ),
    }
    (out / "VALIDATION_AND_METHODS_REPORT.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (out / "README_REVISED_ANALYSIS.txt").write_text(
        "Dire Dawa extreme rainfall: revised reproducible analysis\n"
        "========================================================\n\n"
        "Run from a terminal after installing requirements:\n"
        "  python extreme_rainfall_analysis.py --input-dir data --output-dir revised_analysis\n\n"
        "The original source CSVs are read only. New outputs are written to revised_analysis.\n"
        "GEV fitting retains the notebook's L-moment method. POT retains the 95th percentile\n"
        "and 3-day runs declustering. POT return levels now use a Poisson annual exceedance\n"
        "probability definition so GEV and POT return periods are comparable. POT bootstrap\n"
        "intervals account for event-count and excess-sampling variability, but remain\n"
        "conditional on the selected threshold and declustering rule.\n\n"
        "The analysis represents daily area-average CHIRPS rainfall over the Dire Dawa\n"
        "administrative boundary. It does not represent point-gauge or pixel-level extremes.\n"
        "See VALIDATION_AND_METHODS_REPORT.json for validation and method notes.\n",
        encoding="utf-8",
    )

    print("Analysis completed.")
    print(f"Daily records: {len(daily):,}; annual maxima: {len(annual):,}")
    print(f"POT threshold: {threshold:.4f} mm; independent events: {len(events)}")
    print(f"Outputs saved to: {out.resolve()}")


if __name__ == "__main__":
    main()
