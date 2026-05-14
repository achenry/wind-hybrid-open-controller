"""Verify MLForecast.predict_distr works end-to-end with the Phase 0i-G checkpoint.

Phase 0i-G validation (job 16929835) exercised predict_sample. The controller
simulation path (simulate_case_studies) uses predict_distr instead, which is a
separate code path that emits loc_* / sd_* columns. This script constructs the
real MLForecast object the same way run_forecaster_validation.py does, feeds it
a real slice of context data, calls predict_distr, and audits the output.

Run on a GPU node:
    python verify_predict_distr_phase0i_g.py
"""
import logging
import sys
import traceback

import numpy as np
import pandas as pd
import polars as pl
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("verify")

sys.path.insert(0, "/fs/dss/home/taed7566/Forecasting/pytorch-transformer-ts")
sys.path.insert(0, "/user/taed7566/Forecasting/wind-hybrid-open-controller")
sys.path.insert(0, "/user/taed7566/Forecasting/wind-forecasting")

MODEL_CONFIG = "/user/taed7566/Forecasting/wind-forecasting/config/training/training_inputs_storm_awaken_unsmoothed_pred60_tactis_phase0i_g.yaml"
DATA_CONFIG = "/user/taed7566/Forecasting/wind-forecasting/config/preprocessing/preprocessing_inputs_awaken_STORM.yaml"
CHECKPOINT = "/dss/work/taed7566/Forecasting_Outputs/wind-forecasting/logs/train_phase0i_g_quantile_pinball_tactis/20260512_101511_0_0/manual_save_epoch99.ckpt"

STAGES = []


def stage(name, fn):
    log.info("=" * 70)
    log.info("STAGE: %s", name)
    log.info("=" * 70)
    try:
        result = fn()
        STAGES.append((name, "PASS", ""))
        return result
    except Exception as e:
        tb = traceback.format_exc()
        STAGES.append((name, "FAIL", f"{type(e).__name__}: {e}"))
        log.error("STAGE FAILED: %s\n%s", name, tb)
        raise


def main():
    # fmodel is a stored dataclass field only — never referenced by MLForecast.__init__
    # or predict_distr — so we pass None and skip the FLORIS dependency entirely.
    from whoc.wind_forecast.ml_forecast import MLForecast

    with open(MODEL_CONFIG) as f:
        model_config = yaml.safe_load(f)
    with open(DATA_CONFIG) as f:
        data_config = yaml.safe_load(f)

    # --- replicate run_forecaster_validation.py turbine wiring ---
    sig = data_config.get("turbine_signature")
    if isinstance(sig, list):
        turbine_signature = sig[0] if len(sig) == 1 else r"\d+"
    else:
        turbine_signature = sig or r"\d+"

    tmap = data_config.get("turbine_mapping")
    if isinstance(tmap, list) and tmap:
        keys = list(tmap[0].keys()) if len(data_config.get("turbine_signature", [1])) == 1 else list(tmap[0].values())
        tid2idx_mapping = {str(k): i for i, k in enumerate(keys)}
    else:
        # fallback: derive from the data parquet's ws_horz_* columns
        cols = pl.scan_parquet(model_config["dataset"]["data_path"]).collect_schema().names()
        wt_ids = [c.replace("ws_horz_", "") for c in cols if c.startswith("ws_horz_")]
        tid2idx_mapping = {str(k): i for i, k in enumerate(wt_ids)}
    log.info("turbine_signature=%r, n turbines in tid2idx=%d", turbine_signature, len(tid2idx_mapping))

    measurements_timedelta = pd.Timedelta(seconds=15)
    controller_timedelta = max(pd.Timedelta(5, unit="s"), measurements_timedelta)
    ptd = pd.Timedelta(seconds=model_config["dataset"]["prediction_length"])
    ctd = pd.Timedelta(seconds=model_config["dataset"]["context_length"])

    def build_forecaster():
        return MLForecast(
            measurements_timedelta=measurements_timedelta,
            controller_timedelta=controller_timedelta,
            prediction_timedelta=ptd,
            context_timedelta=ctd,
            fmodel=None,
            true_wind_field=None,
            tid2idx_mapping=tid2idx_mapping,
            turbine_signature=turbine_signature,
            use_tuned_params=True,
            kwargs=dict(
                model_key="tactis",
                model_checkpoint=CHECKPOINT,
                optuna_storage=None,
                study_name=None,
                model_config=model_config,
                resample=False,
            ),
            target_turbine_indices=None,
        )

    forecaster = stage("Construct MLForecast (loads Phase 0i-G checkpoint)", build_forecaster)

    stage("forecaster.reset()", lambda: forecaster.reset())

    # --- build a real context window of historic measurements ---
    def load_context():
        # LAZY scan — never materialize the full multi-GB parquet.
        # IMPORTANT: awaken_processed_unsmoothed_normalized.parquet is at 1-SECOND
        # resolution, but the model was trained on 15-SECOND data (data_module.freq).
        # predict_distr / _generate_test_data expect input already at the model freq.
        # So we resample 1s -> 15s by mean (matches the training preprocessing)
        # before handing it over — exactly what the real validation pipeline feeds in.
        n_ctx = forecaster.n_context  # 80 steps at 15s
        model_freq = str(forecaster.data_module.freq)  # "15s"
        lf = pl.scan_parquet(model_config["dataset"]["data_path"])
        first_cg = lf.select("continuity_group").head(1).collect()["continuity_group"][0]
        # grab enough 1s rows to build n_ctx+margin 15s rows
        n_raw = (n_ctx + 25) * 15
        raw = (lf.filter(pl.col("continuity_group") == first_cg)
                 .sort("time")
                 .head(n_raw)
                 .collect()
                 .drop("continuity_group"))
        log.info("raw 1s rows pulled: %d (continuity_group=%s)", raw.height, first_cg)
        # resample 1s -> model freq by mean
        sub = (raw.sort("time")
                  .group_by_dynamic("time", every=model_freq)
                  .agg(pl.all().mean()))
        if sub.height < n_ctx + 1:
            raise RuntimeError(
                f"after 1s->{model_freq} resample only {sub.height} rows, need >= {n_ctx + 1}"
            )
        # Mirror make_predictions' contract: predict_distr receives rows with
        # time <= current_time, so current_time = the LAST resampled timestamp.
        current_time = sub["time"][-1]
        step = (sub["time"][1] - sub["time"][0])
        log.info("resampled to %s: rows=%d, step=%s, n_context=%d, current_time=%s",
                 model_freq, sub.height, step, n_ctx, current_time)
        return sub, current_time

    hist, current_time = stage("Load real context window", load_context)

    # --- the actual test: call predict_distr ---
    def run_predict_distr():
        pred = forecaster.predict_distr(hist, current_time)
        return pred

    pred = stage("CALL predict_distr", run_predict_distr)

    # --- audit the output ---
    def audit():
        assert pred is not None and pred.height > 0, "empty prediction"
        cols = pred.columns
        loc_cols = [c for c in cols if c.startswith("loc_")]
        sd_cols = [c for c in cols if c.startswith("sd_")]
        log.info("output rows=%d, cols=%d", pred.height, len(cols))
        log.info("  loc_* columns: %d (e.g. %s)", len(loc_cols), loc_cols[:3])
        log.info("  sd_*  columns: %d (e.g. %s)", len(sd_cols), sd_cols[:3])
        assert len(sd_cols) > 0, "NO sd_* columns produced — controller would have no stddev input"
        assert len(loc_cols) > 0, "NO loc_* columns produced"
        # NaN / negativity / magnitude checks on sd_*
        sd_arr = pred.select(sd_cols).to_numpy()
        loc_arr = pred.select(loc_cols).to_numpy()
        n_nan_sd = int(np.isnan(sd_arr).sum())
        n_neg_sd = int((sd_arr < 0).sum())
        n_nan_loc = int(np.isnan(loc_arr).sum())
        log.info("  sd_* : NaN=%d, negative=%d, min=%.4f, median=%.4f, max=%.4f",
                 n_nan_sd, n_neg_sd, np.nanmin(sd_arr), np.nanmedian(sd_arr), np.nanmax(sd_arr))
        log.info("  loc_*: NaN=%d, min=%.4f, median=%.4f, max=%.4f",
                 n_nan_loc, np.nanmin(loc_arr), np.nanmedian(loc_arr), np.nanmax(loc_arr))
        assert n_nan_sd == 0, f"{n_nan_sd} NaN values in sd_* columns"
        assert n_neg_sd == 0, f"{n_neg_sd} negative values in sd_* columns"
        assert n_nan_loc == 0, f"{n_nan_loc} NaN values in loc_* columns"
        # the prediction horizon should be prediction_length steps
        log.info("  prediction timestamps: %d (expect ~%d)",
                 pred["time"].n_unique(), int(ptd / measurements_timedelta))
        return dict(rows=pred.height, loc_cols=len(loc_cols), sd_cols=len(sd_cols),
                    sd_median=float(np.nanmedian(sd_arr)))

    summary = stage("Audit predict_distr output", audit)
    return summary


if __name__ == "__main__":
    ok = True
    try:
        summary = main()
    except Exception:
        ok = False
        summary = None

    print("\n" + "=" * 70)
    print("VERIFICATION REPORT — predict_distr on Phase 0i-G checkpoint")
    print("=" * 70)
    for name, status, detail in STAGES:
        mark = "PASS" if status == "PASS" else "FAIL"
        print(f"  [{mark}] {name}" + (f"  — {detail}" if detail else ""))
    print("=" * 70)
    if ok and summary:
        print(f"RESULT: predict_distr WORKS with Phase 0i-G.")
        print(f"  output rows={summary['rows']}, loc_* cols={summary['loc_cols']}, "
              f"sd_* cols={summary['sd_cols']}, sd median={summary['sd_median']:.4f}")
        sys.exit(0)
    else:
        print("RESULT: predict_distr FAILED — see failed stage above.")
        sys.exit(1)
