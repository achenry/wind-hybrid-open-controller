"""FAITHFUL verification: predict_distr vs predict_sample on identical real input.

Doubts being tested empirically:
  1. Does predict_distr produce a non-empty df with loc_*/sd_* columns?  (structural)
  2. Are the sd_* values HEALTHY (not collapsed to ~0)?                  (value sanity)
  3. Does predict_distr's sd_* MATCH predict_sample's empirical sample
     std on the SAME input?  predict_sample is the known-good oracle —
     job 16929835 produced healthy spread (F^-1 width 2.09) through it.

Faithful to the real pipeline (run_forecaster_validation.py:1062-1162):
  - data: the cached test split, DENORMALIZED (raw m/s), 15s, all-turbine
  - measurements_timedelta = 15s  (SBATCH used --simulation_timestep 15)
  - input format: wide, turbine-suffixed columns (ws_horz_wtNNN ...)

If predict_sample stays healthy but predict_distr collapses -> real predict_distr bug.
If both collapse -> input still wrong.  If both healthy and ~equal -> predict_distr verified.
"""
import logging
import sys
import traceback

import numpy as np
import pandas as pd
import polars as pl
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("faithful")

sys.path.insert(0, "/fs/dss/home/taed7566/Forecasting/pytorch-transformer-ts")
sys.path.insert(0, "/user/taed7566/Forecasting/wind-hybrid-open-controller")
sys.path.insert(0, "/user/taed7566/Forecasting/wind-forecasting")

MODEL_CONFIG = "/user/taed7566/Forecasting/wind-forecasting/config/training/training_inputs_storm_awaken_unsmoothed_pred60_tactis_phase0i_g.yaml"
CHECKPOINT = "/dss/work/taed7566/Forecasting_Outputs/wind-forecasting/logs/train_phase0i_g_quantile_pinball_tactis/20260512_101511_0_0/manual_save_epoch99.ckpt"
# Note: the cached test split is loaded via the DataModule (build_test_dm), not a direct path.

STAGES = []


def stage(name, fn):
    log.info("=" * 70)
    log.info("STAGE: %s", name)
    log.info("=" * 70)
    try:
        r = fn()
        STAGES.append((name, "PASS", ""))
        return r
    except Exception as e:
        STAGES.append((name, "FAIL", f"{type(e).__name__}: {e}"))
        log.error("STAGE FAILED: %s\n%s", name, traceback.format_exc())
        raise


def main():
    from whoc.wind_forecast.ml_forecast import MLForecast
    from wind_forecasting.preprocessing.data_module import DataModule

    with open(MODEL_CONFIG) as f:
        mcnf = yaml.safe_load(f)

    measurements_timedelta = pd.Timedelta(seconds=15)  # SBATCH --simulation_timestep 15
    ptd = pd.Timedelta(seconds=mcnf["dataset"]["prediction_length"])
    ctd = pd.Timedelta(seconds=mcnf["dataset"]["context_length"])

    # --- Stage 1: build the test_data DataModule exactly like run_forecaster_validation.py:1062
    def build_test_dm():
        dm = DataModule(
            normalized_data_path=mcnf["dataset"]["data_path"],
            normalization_consts_path=mcnf["dataset"]["normalization_consts_path"],
            use_normalization=False,                       # DENORMALIZED — raw m/s
            n_splits=1,
            continuity_groups=None,
            train_split=(1.0 - mcnf["dataset"]["val_split"] - mcnf["dataset"]["test_split"]),
            val_split=mcnf["dataset"]["val_split"],
            test_split=mcnf["dataset"]["test_split"],
            prediction_length=mcnf["dataset"]["prediction_length"],
            context_length=mcnf["dataset"]["context_length"],
            target_prefixes=["ws_horz", "ws_vert"],
            feat_dynamic_real_prefixes=["nd_cos", "nd_sin"],
            freq=f"{int(measurements_timedelta.total_seconds())}s",   # "15s"
            target_suffixes=mcnf["dataset"]["target_turbine_ids"],
            per_turbine_target=False,                      # all-turbine wide format
            as_lazyframe=True,
            dtype=pl.Float32,
        )
        dm.generate_splits(save=True, reload=False, splits=["test"])
        log.info("target_cols[:4]=%s ... (%d total)", dm.target_cols[:4], len(dm.target_cols))
        log.info("feat_dynamic_real_cols[:2]=%s ... (%d total)",
                 dm.feat_dynamic_real_cols[:2], len(dm.feat_dynamic_real_cols))
        return dm

    test_dm = stage("Build test DataModule + generate_splits(test)", build_test_dm)

    # --- Stage 2: load + rename the test split into the wide format predict_* expect
    def build_test_data():
        test_df = test_dm.datasets["test"]
        if hasattr(test_df, "collect"):
            test_df = test_df.collect()
        rename_map = {
            **{f"target_{i}": c for i, c in enumerate(test_dm.target_cols)},
            **{f"feat_dynamic_real_{i}": c
               for i, c in enumerate(test_dm.feat_dynamic_real_cols)},
        }
        test_df = (test_df.rename(rename_map)
                          .with_columns(continuity_group=pl.col("item_id")
                                        .str.extract(r"SPLIT(\d+)").cast(int)))
        log.info("test_data rows=%d, n continuity groups=%d",
                 test_df.height, test_df["continuity_group"].n_unique())
        return test_df

    test_data = stage("Load + rename cached test split", build_test_data)

    # --- Stage 3: construct MLForecast
    def build_forecaster():
        return MLForecast(
            measurements_timedelta=measurements_timedelta,
            controller_timedelta=max(pd.Timedelta(5, unit="s"), measurements_timedelta),
            prediction_timedelta=ptd,
            context_timedelta=ctd,
            fmodel=None,
            true_wind_field=None,
            tid2idx_mapping={str(s): i for i, s in enumerate(test_dm.target_suffixes)},
            turbine_signature=r"\d+",
            use_tuned_params=True,
            kwargs=dict(model_key="tactis", model_checkpoint=CHECKPOINT,
                        optuna_storage=None, study_name=None,
                        model_config=mcnf, resample=False),
            target_turbine_indices=None,
        )

    forecaster = stage("Construct MLForecast", build_forecaster)
    stage("forecaster.reset()", lambda: forecaster.reset())

    # --- Stage 4: pick a continuity group + build a context window
    def pick_context():
        n_ctx = forecaster.n_context
        sizes = (test_data.group_by("continuity_group").agg(pl.len().alias("n"))
                 .sort("n", descending=True))
        cg = sizes["continuity_group"][0]
        log.info("largest continuity_group=%s has %d rows (n_context=%d)",
                 cg, sizes["n"][0], n_ctx)
        sub = (test_data.filter(pl.col("continuity_group") == cg)
                        .sort("time")
                        .head(n_ctx + 20)
                        .drop("continuity_group"))
        # drop non-measurement columns predict_* don't expect
        for extra in ("item_id", "feat_static_cat", "prediction_timedelta"):
            if extra in sub.columns:
                sub = sub.drop(extra)
        current_time = sub["time"][-1]
        log.info("context rows=%d, current_time=%s, cols sample=%s",
                 sub.height, current_time, sub.columns[:5])
        return sub, current_time

    ctx, current_time = stage("Pick continuity group + context window", pick_context)

    # --- Stage 5: predict_sample (ORACLE) on this exact input
    def run_sample():
        pred = forecaster.predict_sample(ctx, current_time, n_samples=200)
        log.info("predict_sample -> rows=%d, cols=%d, n_samples=%d, n_times=%d",
                 pred.height, len(pred.columns),
                 pred["sample"].n_unique(), pred["time"].n_unique())
        return pred

    samp = stage("predict_sample (oracle)", run_sample)

    # --- Stage 6: predict_distr on the SAME input
    def run_distr():
        pred = forecaster.predict_distr(ctx, current_time)
        assert pred is not None and pred.height > 0, "predict_distr returned empty"
        log.info("predict_distr -> rows=%d, cols=%d, n_times=%d",
                 pred.height, len(pred.columns), pred["time"].n_unique())
        return pred

    distr = stage("predict_distr (under test)", run_distr)

    # --- Stage 7: COMPARE oracle vs predict_distr
    def compare():
        # turbine-component columns present in the sample output
        wt_cols = [c for c in samp.columns
                   if c.startswith("ws_horz_") or c.startswith("ws_vert_")]
        # empirical std from predict_sample, per (time, turbine-component)
        emp = (samp.group_by("time")
                   .agg([pl.col(c).std().alias(f"emp_sd_{c}") for c in wt_cols])
                   .sort("time"))
        # align times via JOIN (not set+is_in — polars is_in fails silently on
        # datetime-precision mismatch). Cast both to ns first so the join keys match.
        d = distr.with_columns(pl.col("time").cast(pl.Datetime("ns"))).sort("time")
        e = emp.with_columns(pl.col("time").cast(pl.Datetime("ns"))).sort("time")
        joined = d.join(e, on="time", how="inner").sort("time")
        log.info("common prediction timestamps (via join): %d", joined.height)
        assert joined.height > 0, "no overlapping timestamps between sample and distr"

        # collect all sd_* from distr and emp_sd_* from oracle, matched by column
        ratios, distr_sds, oracle_sds = [], [], []
        for c in wt_cols:
            sd_col = f"sd_{c}"
            emp_col = f"emp_sd_{c}"
            if sd_col not in joined.columns or emp_col not in joined.columns:
                continue
            dv = joined[sd_col].to_numpy().astype(float)
            ev = joined[emp_col].to_numpy().astype(float)
            distr_sds.append(dv)
            oracle_sds.append(ev)
            mask = ev > 1e-9
            ratios.append(dv[mask] / ev[mask])

        distr_sds = np.concatenate(distr_sds)
        oracle_sds = np.concatenate(oracle_sds)
        ratios = np.concatenate(ratios) if ratios else np.array([np.nan])

        log.info("-" * 60)
        log.info("predict_distr  sd_*  : median=%.5f  mean=%.5f  p10=%.5f  p90=%.5f  max=%.5f",
                 np.median(distr_sds), distr_sds.mean(),
                 np.quantile(distr_sds, .1), np.quantile(distr_sds, .9), distr_sds.max())
        log.info("predict_sample emp_sd: median=%.5f  mean=%.5f  p10=%.5f  p90=%.5f  max=%.5f",
                 np.median(oracle_sds), oracle_sds.mean(),
                 np.quantile(oracle_sds, .1), np.quantile(oracle_sds, .9), oracle_sds.max())
        log.info("ratio distr/oracle   : median=%.4f  mean=%.4f  (1.0 == perfect match)",
                 np.median(ratios), np.mean(ratios))
        log.info("-" * 60)

        # VERDICTS
        oracle_healthy = np.median(oracle_sds) > 0.1   # m/s — known-good path should be wide
        distr_healthy = np.median(distr_sds) > 0.1
        distr_matches_oracle = 0.5 < np.median(ratios) < 2.0

        log.info("oracle (predict_sample) healthy (median sd > 0.1 m/s): %s", oracle_healthy)
        log.info("predict_distr healthy (median sd > 0.1 m/s):           %s", distr_healthy)
        log.info("predict_distr matches oracle (0.5 < ratio < 2.0):      %s", distr_matches_oracle)

        return dict(
            distr_sd_median=float(np.median(distr_sds)),
            oracle_sd_median=float(np.median(oracle_sds)),
            ratio_median=float(np.median(ratios)),
            oracle_healthy=bool(oracle_healthy),
            distr_healthy=bool(distr_healthy),
            distr_matches_oracle=bool(distr_matches_oracle),
        )

    result = stage("COMPARE predict_distr vs predict_sample oracle", compare)
    return result


if __name__ == "__main__":
    ok = True
    try:
        result = main()
    except Exception:
        ok = False
        result = None

    print("\n" + "=" * 70)
    print("FAITHFUL VERIFICATION REPORT — predict_distr vs predict_sample oracle")
    print("=" * 70)
    for name, status, detail in STAGES:
        print(f"  [{status}] {name}" + (f"  — {detail}" if detail else ""))
    print("=" * 70)
    if ok and result:
        print(f"predict_distr  sd median : {result['distr_sd_median']:.5f} m/s")
        print(f"predict_sample sd median : {result['oracle_sd_median']:.5f} m/s  (oracle)")
        print(f"ratio (distr/oracle)     : {result['ratio_median']:.4f}")
        print("-" * 70)
        if result["oracle_healthy"] and result["distr_healthy"] and result["distr_matches_oracle"]:
            print("RESULT: predict_distr VERIFIED — healthy spread, matches the oracle.")
            sys.exit(0)
        elif result["oracle_healthy"] and not result["distr_healthy"]:
            print("RESULT: REAL predict_distr BUG — oracle is healthy but predict_distr collapsed.")
            sys.exit(2)
        elif not result["oracle_healthy"]:
            print("RESULT: INCONCLUSIVE — even the oracle collapsed; input still not faithful.")
            sys.exit(3)
        else:
            print("RESULT: predict_distr runs + healthy but does NOT match oracle — investigate scaling.")
            sys.exit(4)
    else:
        print("RESULT: harness failed before comparison — see failed stage above.")
        sys.exit(1)
