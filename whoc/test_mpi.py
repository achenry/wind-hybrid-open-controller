from mpi4py import MPI
from mpi4py.futures import MPICommExecutor

def f(worker_idx):
    for i in range(100):
        print(f"Working on worker {worker_idx}, iteration {i}.")

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    max_workers = comm.Get_size() 
    comm_size = comm.Get_size()
    executor = MPICommExecutor(comm, root=0, max_workers=max_workers)
    
    with executor as run_simulations_exec:
        futures = [run_simulations_exec.submit(worker_idx) for worker_idx in range(10)]
        
        _ = [fut.result() for fut in futures]