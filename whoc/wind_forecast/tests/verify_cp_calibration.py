"""End-to-end verification of the CP stddev calibration in MLForecast.predict_distr.

Builds TWO forecasters from the SAME quantile-head TACTiS-2 checkpoint:
  - f_off : cp_calibrate_stddev not set (default OFF) — current behaviour
  - f_on  : cp_calibrate_stddev=True — multiplies raw sd_* by the per-(lead,
            component) CP scale factor from cp_scale_factors/quantile_head.json

Both run predict_distr on the SAME context window with the SAME torch seed, so
the only difference between their sd_* outputs is the calibration multiply.

Assertions:
  A  flag OFF is unchanged    — f_off sd_* median ~0.73 m/s (matches the known
                                predict_sample oracle from verify_predict_distr_faithful.py)
  B  flag ON = raw x scale    — for every (lead, component): f_on.sd / f_off.sd
                                ~= scale_factors[component][lead]  (within 1e-3)
  C  magnitude is sane        — f_on sd_* median ~3-4 m/s; horz < vert
  D  lead structure preserved — f_on per-lead mean sd rises from lead 0 to lead 2
                                (guards against a lead/component axis transpose)

Run on a GPU node.
"""
import json
import logging
import sys
import traceback

import numpy as np
import pandas as pd
import polars as pl
import torch
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("verify-cp")

sys.path.insert(0, "/fs/dss/home/taed7566/Forecasting/pytorch-transformer-ts")
sys.path.insert(0, "/user/taed7566/Forecasting/wind-hybrid-open-controller")
sys.path.insert(0, "/user/taed7566/Forecasting/wind-forecasting")

MODEL_CONFIG = "/user/taed7566/Forecasting/wind-forecasting/config/training/training_inputs_storm_awaken_unsmoothed_pred60_tactis_phase0i_g.yaml"
CHECKPOINT = "/dss/work/taed7566/Forecasting_Outputs/wind-forecasting/logs/train_phase0i_g_quantile_pinball_tactis/20260512_101511_0_0/manual_save_epoch99.ckpt"
TEST_PARQUET = "/dss/work/taed7566/Forecasting_Outputs/wind-forecasting/DATA/preprocessed_awaken_data/awaken_processed_unsmoothed_normalized_train_ready_15s_all_turbine_ctx80_pred4_test_denormalize.parquet"
SCALE_TABLE = "/user/taed7566/Forecasting/wind-hybrid-open-controller/whoc/wind_forecast/cp_scale_factors/quantile_head.json"
SEED = 42

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
    scale_factors = json.load(open(SCALE_TABLE))["scale_factors"]
    log.info("scale_factors = %s", scale_factors)

    measurements_timedelta = pd.Timedelta(seconds=15)
    ptd = pd.Timedelta(seconds=mcnf["dataset"]["prediction_length"])
    ctd = pd.Timedelta(seconds=mcnf["dataset"]["context_length"])

    # --- test DataModule, faithful to run_forecaster_validation.py:1062 ---
    def build_test_dm():
        dm = DataModule(
            normalized_data_path=mcnf["dataset"]["data_path"],
            normalization_consts_path=mcnf["dataset"]["normalization_consts_path"],
            use_normalization=False,
            n_splits=1,
            continuity_groups=None,
            train_split=(1.0 - mcnf["dataset"]["val_split"] - mcnf["dataset"]["test_split"]),
            val_split=mcnf["dataset"]["val_split"],
            test_split=mcnf["dataset"]["test_split"],
            prediction_length=mcnf["dataset"]["prediction_length"],
            context_length=mcnf["dataset"]["context_length"],
            target_prefixes=["ws_horz", "ws_vert"],
            feat_dynamic_real_prefixes=["nd_cos", "nd_sin"],
            freq=f"{int(measurements_timedelta.total_seconds())}s",
            target_suffixes=mcnf["dataset"]["target_turbine_ids"],
            per_turbine_target=False,
            as_lazyframe=True,
            dtype=pl.Float32,
        )
        dm.generate_splits(save=True, reload=False, splits=["test"])
        return dm

    test_dm = stage("Build test DataModule + generate_splits(test)", build_test_dm)

    def build_test_data():
        test_df = test_dm.datasets["test"]
        if hasattr(test_df, "collect"):
            test_df = test_df.collect()
        rename_map = {
            **{f"target_{i}": c for i, c in enumerate(test_dm.target_cols)},
            **{f"feat_dynamic_real_{i}": c
               for i, c in enumerate(test_dm.feat_dynamic_real_cols)},
        }
        return (test_df.rename(rename_map)
                       .with_columns(continuity_group=pl.col("item_id")
                                     .str.extract(r"SPLIT(\d+)").cast(int)))

    test_data = stage("Load + rename cached test split", build_test_data)

    # --- two forecasters from the same checkpoint: calibration OFF and ON ---
    def make_forecaster(cp_on):
        kwargs = dict(model_key="tactis", model_checkpoint=CHECKPOINT,
                      optuna_storage=None, study_name=None,
                      model_config=mcnf, resample=False)
        if cp_on:
            kwargs["cp_calibrate_stddev"] = True
            kwargs["cp_scale_factors_path"] = SCALE_TABLE
        return MLForecast(
            measurements_timedelta=measurements_timedelta,
            controller_timedelta=max(pd.Timedelta(5, unit="s"), measurements_timedelta),
            prediction_timedelta=ptd, context_timedelta=ctd,
            fmodel=None, true_wind_field=None,
            tid2idx_mapping={str(s): i for i, s in enumerate(test_dm.target_suffixes)},
            turbine_signature=r"\d+", use_tuned_params=True,
            kwargs=kwargs, target_turbine_indices=None,
        )

    f_off = stage("Construct MLForecast (calibration OFF)", lambda: make_forecaster(False))
    f_on = stage("Construct MLForecast (calibration ON)", lambda: make_forecaster(True))
    stage("reset() both forecasters", lambda: (f_off.reset(), f_on.reset()))

    # sanity: f_on actually loaded the table, f_off did not
    assert f_on.cp_calibrate_stddev and f_on.cp_scale_factors is not None, "f_on did not load scale factors"
    assert not f_off.cp_calibrate_stddev, "f_off should have calibration disabled"

    def pick_context():
        n_ctx = f_off.n_context
        sizes = (test_data.group_by("continuity_group").agg(pl.len().alias("n"))
                 .sort("n", descending=True))
        cg = sizes["continuity_group"][0]
        sub = (test_data.filter(pl.col("continuity_group") == cg)
                        .sort("time").head(n_ctx + 20).drop("continuity_group"))
        for extra in ("item_id", "feat_static_cat", "prediction_timedelta"):
            if extra in sub.columns:
                sub = sub.drop(extra)
        log.info("context cg=%s rows=%d current_time=%s", cg, sub.height, sub["time"][-1])
        return sub, sub["time"][-1]

    ctx, current_time = stage("Pick continuity group + context window", pick_context)

    # --- run predict_distr on both, same seed so samples match ---
    def run_off():
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        return f_off.predict_distr(ctx, current_time)

    def run_on():
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        return f_on.predict_distr(ctx, current_time)

    pred_off = stage("predict_distr — calibration OFF", run_off)
    pred_on = stage("predict_distr — calibration ON", run_on)

    # --- compare ---
    def compare():
        sd_cols = [c for c in pred_off.columns if c.startswith("sd_")]
        assert sd_cols, "no sd_* columns in pred_off"
        assert set(sd_cols) == set(c for c in pred_on.columns if c.startswith("sd_")), \
            "sd_* column sets differ between OFF and ON"
        n_leads = pred_off.height
        log.info("sd_* columns: %d, prediction rows (leads): %d", len(sd_cols), n_leads)

        off = pred_off.sort("time")
        on = pred_on.sort("time")
        off_arr = off.select(sd_cols).to_numpy()   # [n_leads, n_sd_cols]
        on_arr = on.select(sd_cols).to_numpy()

        # Assertion A — flag OFF unchanged: median ~0.73 m/s (the known oracle)
        off_med = float(np.nanmedian(off_arr))
        log.info("[A] f_off sd_* median = %.4f m/s (expect ~0.73, the predict_sample oracle)", off_med)
        a_ok = 0.5 < off_med < 1.1

        # Assertion B — flag ON = raw x per-(lead,component) scale factor
        max_rel_err = 0.0
        for j, col in enumerate(sd_cols):
            comp = "ws_horz" if "ws_horz" in col else "ws_vert"
            factors = np.array(scale_factors[comp][:n_leads])
            expected = off_arr[:, j] * factors
            got = on_arr[:, j]
            denom = np.where(np.abs(expected) > 1e-9, np.abs(expected), 1.0)
            rel = np.abs(got - expected) / denom
            max_rel_err = max(max_rel_err, float(np.nanmax(rel)))
        log.info("[B] max relative error of f_on vs f_off x scale = %.2e (expect < 1e-3)", max_rel_err)
        b_ok = max_rel_err < 1e-3

        # Assertion C — magnitude sane: f_on median ~3-4 m/s; horz < vert
        on_med = float(np.nanmedian(on_arr))
        horz_cols = [j for j, c in enumerate(sd_cols) if "ws_horz" in c]
        vert_cols = [j for j, c in enumerate(sd_cols) if "ws_vert" in c]
        horz_med = float(np.nanmedian(on_arr[:, horz_cols]))
        vert_med = float(np.nanmedian(on_arr[:, vert_cols]))
        log.info("[C] f_on sd_* median = %.4f m/s  (horz=%.4f, vert=%.4f)", on_med, horz_med, vert_med)
        c_ok = (2.0 < on_med < 5.0) and (horz_med < vert_med)

        # Assertion D — lead structure preserved: per-lead mean rises lead0->lead2
        per_lead_mean = np.nanmean(on_arr, axis=1)  # [n_leads]
        log.info("[D] f_on per-lead mean sd = %s", np.round(per_lead_mean, 4).tolist())
        d_ok = per_lead_mean[0] < per_lead_mean[min(2, n_leads - 1)]

        log.info("-" * 60)
        log.info("[A] flag OFF unchanged (median ~0.73)        : %s", a_ok)
        log.info("[B] flag ON == raw x scale (<1e-3 rel err)   : %s", b_ok)
        log.info("[C] magnitude sane (~3-4 m/s, horz<vert)     : %s", c_ok)
        log.info("[D] lead structure preserved (rises 0->2)    : %s", d_ok)
        return dict(a=a_ok, b=b_ok, c=c_ok, d=d_ok,
                    off_med=off_med, on_med=on_med, horz_med=horz_med,
                    vert_med=vert_med, max_rel_err=max_rel_err)

    return stage("COMPARE calibration OFF vs ON", compare)


if __name__ == "__main__":
    ok = True
    try:
        result = main()
    except Exception:
        ok = False
        result = None

    print("\n" + "=" * 70)
    print("CP CALIBRATION VERIFICATION REPORT")
    print("=" * 70)
    for name, status, detail in STAGES:
        print(f"  [{status}] {name}" + (f"  — {detail}" if detail else ""))
    print("=" * 70)
    if ok and result:
        print(f"f_off sd median : {result['off_med']:.4f} m/s")
        print(f"f_on  sd median : {result['on_med']:.4f} m/s  (horz {result['horz_med']:.3f}, vert {result['vert_med']:.3f})")
        print(f"max rel err (ON vs raw x scale): {result['max_rel_err']:.2e}")
        print("-" * 70)
        if all([result["a"], result["b"], result["c"], result["d"]]):
            print("RESULT: CP calibration VERIFIED — A/B/C/D all pass.")
            sys.exit(0)
        print("RESULT: CP calibration FAILED — see A/B/C/D above.")
        sys.exit(2)
    print("RESULT: harness failed before comparison — see failed stage above.")
    sys.exit(1)
