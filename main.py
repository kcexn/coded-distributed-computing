from sys import stderr,exit

import numpy as np
from scipy import linalg

from coded_distributed_computing import encode_matrix


if __name__ == "__main__":
    #K = 49, NUM_OF_DIVISION = 8 is a good edge case to consider where the 8th matrix is empty with shape 0x30
    K = 60
    N = 30
    SCALE_FACTOR = 2**10
    NUM_OF_MATRICES = 1
    NUM_OF_PROCESSORS = 1
    MAX_NUM_OF_DIVISIONS = K//NUM_OF_MATRICES + 1
    NUM_OF_DIVISIONS = 2

    try:
        assert NUM_OF_DIVISIONS < MAX_NUM_OF_DIVISIONS
    except AssertionError as error:
        print(f'The maximum number of matrix divisions is {MAX_NUM_OF_DIVISIONS}. NUM_OF_DIVISIONS is too high.', file=stderr)
        exit(1)

    H = SCALE_FACTOR*np.random.random((K,N))-(SCALE_FACTOR/2)*np.ones((K,N))
    x = SCALE_FACTOR*np.random.random((N,1))
    
    reference_solution = np.matmul(H,x)
    #  if K%2 == 0 else H[(K//NUM_OF_DIVISIONS+1)*i:(K//NUM_OF_DIVISIONS+1)*(i+1),:]

    A_matrices = [
        np.asmatrix(
            H[K//NUM_OF_DIVISIONS*i:K//NUM_OF_DIVISIONS*(i+1),:] if K%NUM_OF_DIVISIONS == 0 else H[(K//NUM_OF_DIVISIONS+1)*i:(K//NUM_OF_DIVISIONS+1)*(i+1),:]
        ) for i in range(NUM_OF_DIVISIONS)
    ]

    print(A_matrices[7])
