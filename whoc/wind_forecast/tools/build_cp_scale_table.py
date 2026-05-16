"""Build the CP stddev scale-factor table consumed by MLForecast.predict_distr.

The TACTiS-2 model's raw predictive standard deviations are honest in shape but
under-confident: at the 90% nominal level the raw bands cover only ~56% of the
truth. Conformal Prediction (CQR) bands cover >=90% by construction. The ratio
of the CP band width to the raw band width is therefore a multiplicative
correction factor for the raw stddev.

This script aggregates that factor from a `calibrated_intervals.parquet`
(produced by wind-forecasting's apply_cp_to_phase0i_b.py) down to a small
per-(lead_step, component) table — 4 leads x 2 components = 8 numbers. Turbine-
to-turbine variation in the factor has a coefficient of variation of only ~0.2,
so aggregating over turbines captures the two axes that genuinely matter
(forecast horizon and wind component) without overfitting one validation slice.

Run once, offline:
    python build_cp_scale_table.py \\
        --intervals <cp_results_dir>/calibrated_intervals.parquet \\
        --output    ../cp_scale_factors/quantile_head.json \\
        --alpha 0.1
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("build-cp-scale")

# component prefixes — must match MLForecast.data_module.target_prefixes
COMPONENTS = ["ws_horz", "ws_vert"]
_TARGET_RE = re.compile(r"^(ws_horz|ws_vert)_wt\d+$")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--intervals", type=Path, required=True,
                    help="calibrated_intervals.parquet from the CP pipeline")
    ap.add_argument("--output", type=Path, required=True,
                    help="destination JSON (e.g. ../cp_scale_factors/quantile_head.json)")
    ap.add_argument("--alpha", type=float, default=0.1,
                    help="confidence level to build the table at (default 0.1 = 90% nominal)")
    args = ap.parse_args()

    log.info("Loading %s", args.intervals)
    ci = pl.read_parquet(args.intervals).filter(pl.col("alpha") == args.alpha)
    if ci.height == 0:
        raise SystemExit(f"No rows with alpha == {args.alpha} in {args.intervals}")
    log.info("  rows at alpha=%s: %d", args.alpha, ci.height)

    # per-row CP scale factor: (cp band width) / (raw band width), guarded
    ci = ci.with_columns([
        (pl.col("upper_cp") - pl.col("lower_cp")).alias("cp_w"),
        (pl.col("upper_raw") - pl.col("lower_raw")).alias("raw_w"),
    ])
    ci = ci.with_columns(
        pl.when(pl.col("raw_w") > 1e-9)
          .then(pl.col("cp_w") / pl.col("raw_w"))
          .otherwise(None)  # degenerate (zero-range) targets -> dropped from the mean
          .alias("scale")
    )
    # component from target_col (e.g. "ws_horz_wt042" -> "ws_horz")
    ci = ci.with_columns(
        pl.col("target_col").str.extract(r"^(ws_horz|ws_vert)_wt\d+$", 1).alias("component")
    )
    n_bad = ci.filter(pl.col("component").is_null()).height
    if n_bad:
        log.warning("  %d rows had a target_col not matching ws_(horz|vert)_wtNNN — dropped", n_bad)
    ci = ci.drop_nulls(["scale", "component"])

    leads = sorted(ci["lead_step"].unique().to_list())
    log.info("  lead steps present: %s", leads)

    scale_factors = {c: [] for c in COMPONENTS}
    log.info("Per-(lead, component) aggregate (mean scale, CoV across turbines+time):")
    for comp in COMPONENTS:
        for lead in leads:
            sub = ci.filter((pl.col("component") == comp) & (pl.col("lead_step") == lead))["scale"]
            arr = sub.to_numpy()
            mean = float(np.mean(arr))
            cov = float(np.std(arr) / mean) if mean != 0 else float("nan")
            scale_factors[comp].append(round(mean, 6))
            log.info("  %s lead %d : mean=%.4f  CoV=%.3f  (n=%d)", comp, lead, mean, cov, arr.size)

    out = {
        "description": (
            "CP stddev scale factors — multiply raw predictive stddev by "
            "scale_factors[component][lead_step] to get a CP-calibrated stddev."
        ),
        "source": str(args.intervals),
        "alpha": args.alpha,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "components": COMPONENTS,
        "lead_steps": leads,
        "scale_factors": scale_factors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    log.info("Wrote %s", args.output)
    log.info("scale_factors = %s", json.dumps(scale_factors))


if __name__ == "__main__":
    main()
