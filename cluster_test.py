from numba import jit, prange
import numpy as np
import numba
"""
Parameters to vary:
Dynamic Modell and its Model Parameters i.e threshhold or voter model

"""

def main() -> None:

    A = np.zeros(100)
    print(thread_test(A))
    print(f"Used Threading Layer: {numba.threading_layer()}")
    print(f"Number of used threads: {numba.config.NUMBA_NUM_THREADS}")
    print(f"Number of avaiable Threads: {numba.config.NUMBA_DEFAULT_NUM_THREADS}")

    return

@jit(parallel=True)
def thread_test(A):

    for i in prange(A.shape[0]):
        A[i] = 1
    return A


if __name__ == '__main__':
    main()

