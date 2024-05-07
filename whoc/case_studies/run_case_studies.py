from mpi4py import MPI
from mpi4py.futures import MPICommExecutor
# from dask_mpi import initialize
# from dask.distributed import Client
# from dask.distributed import wait as dask_wait
from concurrent.futures import ProcessPoolExecutor

import os
import sys

from whoc.interfaces.controlled_floris_interface import ControlledFlorisModel
from whoc.controllers.mpc_wake_steering_controller import MPC
from whoc.controllers.greedy_wake_steering_controller import GreedyController
from whoc.controllers.lookup_based_wake_steering_controller import LookupBasedWakeSteeringController
from whoc.case_studies.initialize_case_studies import initialize_simulations, case_families, case_studies, STORAGE_DIR
from whoc.case_studies.simulate_case_studies import simulate_controller
from whoc.case_studies.process_case_studies import process_simulations, plot_simulations

if __name__ == "__main__":
    REGENERATE_WIND_FIELD = False
    RUN_SIMULATIONS = True
    POST_PROCESS = True

    case_families = ["baseline_controllers", "solver_type",
                        "wind_preview_type", "warm_start", 
                        "horizon_length", "breakdown_robustness",
                        "scalability", "cost_func_tuning", 
                        "stochastic_preview_type"]

    DEBUG = sys.argv[1].lower() == "debug"
    # if sys.argv[2].lower() == "dask":
    #     MULTI = "dask"
    #     initialize()
    #     client = Client()
    if sys.argv[2].lower() == "mpi":
        MULTI = "mpi"
    else:
        MULTI = "cf"

    PARALLEL = sys.argv[3].lower() == "parallel"
    if len(sys.argv) > 4:
        CASE_FAMILY_IDX = [int(i) for i in sys.argv[4:]]
    else:
        CASE_FAMILY_IDX = list(range(len(case_families)))

    if DEBUG:
        N_SEEDS = 1
    else:
        N_SEEDS = 6

    for case_family in case_families:
        case_studies[case_family]["wind_case_idx"] = {"group": 2, "vals": [i for i in range(N_SEEDS)]}

    # MISHA QUESTION how to make AMR-Wind wait for control solution?
    # run_simulations(["baseline_controllers"], REGENERATE_WIND_FIELD)
    # mp.set_start_method('fork')
    os.environ["PYOPTSPARSE_REQUIRE_MPI"] = "true"
    # run_simulations(["perfect_preview_type"], REGENERATE_WIND_FIELD)
    print([case_families[i] for i in CASE_FAMILY_IDX])
    if RUN_SIMULATIONS:
        case_lists, case_name_lists, input_dicts, wind_field_config, wind_mag_ts, wind_dir_ts = initialize_simulations([case_families[i] for i in CASE_FAMILY_IDX], regenerate_wind_field=REGENERATE_WIND_FIELD, n_seeds=N_SEEDS, run_parallel=PARALLEL, multi=MULTI)
        
        # run simulations
        print(f"about to submit calls to simulate_controller")
        run_parallel=PARALLEL
        multi = MULTI
        if run_parallel:
            # if platform == "linux":
            
            # if multi == "dask":
            #     futures = [client.submit(simulate_controller, 
            #                                         controller_class=globals()[case_lists[c]["controller_class"]], input_dict=d, 
            #                                         wind_case_idx=case_lists[c]["wind_case_idx"], wind_mag_ts=wind_mag_ts[case_lists[c]["wind_case_idx"]], wind_dir_ts=wind_dir_ts[case_lists[c]["wind_case_idx"]], 
            #                                         case_name=case_lists[c]["case_names"],
            #                                         lut_path=case_lists[c]["lut_path"], generate_lut=case_lists[c]["generate_lut"], seed=case_lists[c]["seed"], wind_field_config=wind_field_config, verbose=False)
            #                 for c, d in enumerate(input_dicts)]
            #     # dask_wait(futures)
            #     results = client.gather(futures)
            #     # results = [fut.result() for fut in futures]
            # # elif platform == "darwin":
            # else:
            if multi == "mpi":
                comm_size = MPI.COMM_WORLD.Get_size()
                # comm_rank = MPI.COMM_WORLD.Get_rank()
                # node_name = MPI.Get_processor_name()
                executor = MPICommExecutor(MPI.COMM_WORLD, root=0)
            elif multi == "cf":
                executor = ProcessPoolExecutor()
            with executor as run_simulations_exec:
                if multi == "mpi":
                    run_simulations_exec.max_workers = comm_size
                print(f"run_simulations line 618 with {run_simulations_exec._max_workers} workers")
                # for MPIPool executor, (waiting as if shutdown() were called with wait set to True)
                futures = [run_simulations_exec.submit(simulate_controller, 
                                                controller_class=globals()[case_lists[c]["controller_class"]], input_dict=d, 
                                                wind_case_idx=case_lists[c]["wind_case_idx"], wind_mag_ts=wind_mag_ts[case_lists[c]["wind_case_idx"]], wind_dir_ts=wind_dir_ts[case_lists[c]["wind_case_idx"]], 
                                                case_name=case_lists[c]["case_names"], case_family="_".join(case_name_lists[c].split("_")[:-1]),
                                                lut_path=case_lists[c]["lut_path"], generate_lut=case_lists[c]["generate_lut"], seed=case_lists[c]["seed"], wind_field_config=wind_field_config, verbose=False)
                        for c, d in enumerate(input_dicts)]
                
                results = [fut.result() for fut in futures]

            print("run_simulations line 626")

        else:
            results = []
            for c, d in enumerate(input_dicts):
                results.append(simulate_controller(controller_class=globals()[case_lists[c]["controller_class"]], input_dict=d, 
                                                wind_case_idx=case_lists[c]["wind_case_idx"], wind_mag_ts=wind_mag_ts[case_lists[c]["wind_case_idx"]], wind_dir_ts=wind_dir_ts[case_lists[c]["wind_case_idx"]], 
                                                case_name=case_lists[c]["case_names"], case_family="_".join(case_name_lists[c].split("_")[:-1]),
                                                lut_path=case_lists[c]["lut_path"], generate_lut=case_lists[c]["generate_lut"], seed=case_lists[c]["seed"],
                                                wind_field_config=wind_field_config, verbose=False))
        
        # save_simulations(case_lists, case_name_lists, results)
                
    if POST_PROCESS:
        results_dirs = [os.path.join(STORAGE_DIR, case_families[i]) for i in CASE_FAMILY_IDX]
        
        # compute stats over all seeds
        process_simulations(results_dirs)
        
        plot_simulations(results_dirs[0:2])