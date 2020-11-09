from sys import stderr,exit
from typing import List

import numpy as np
from scipy import linalg

from coded_distributed_computing import encode_matrix

if __name__ == "__main__":
    K = 49
    N = 30
    SCALE_FACTOR = 2**10
    NUM_OF_MATRICES = 1
    NUM_OF_PROCESSORS = 9
    MAX_NUM_OF_DIVISIONS = K//NUM_OF_MATRICES + 1
    NUM_OF_DIVISIONS = 8

    try:
        assert NUM_OF_DIVISIONS < MAX_NUM_OF_DIVISIONS
    except AssertionError as error:
        print(f'The maximum number of matrix divisions is {MAX_NUM_OF_DIVISIONS}. NUM_OF_DIVISIONS is too high.', file=stderr)
        exit(1)

    H = SCALE_FACTOR*np.random.random((K,N))-(SCALE_FACTOR/2)*np.ones((K,N))
    x = SCALE_FACTOR*np.random.random((N,1))
    
    reference_solution = np.matmul(H,x)

    A_matrices = np.array_split(H, NUM_OF_DIVISIONS)
    
    if K % NUM_OF_DIVISIONS != 0:
        large_generator = SCALE_FACTOR*np.random.random((K//NUM_OF_DIVISIONS + 1, NUM_OF_PROCESSORS - (K//NUM_OF_DIVISIONS + 1)))
        large_generator = np.concatenate((np.identity(K//NUM_OF_DIVISIONS + 1), large_generator), axis = 1)
    generator = SCALE_FACTOR*np.random.random((K//NUM_OF_DIVISIONS, NUM_OF_PROCESSORS - (K//NUM_OF_DIVISIONS)))
    generator = np.concatenate((np.identity(K//NUM_OF_DIVISIONS), generator), axis = 1)
    
    coded_A_matrices = []
    for i,matrix in enumerate(A_matrices):
        if i < K%NUM_OF_DIVISIONS:
            coded_A_matrices.append(
                np.array(np.asmatrix(np.matmul(np.asmatrix(matrix).T, large_generator)).T)
            )
        else:
            coded_A_matrices.append(
                np.array(np.asmatrix(np.matmul(np.asmatrix(matrix).T, generator)).T)
            )

    # This stacking here is wrong. Instead of concatenating along the 3rd dimension like I've done here.
    # I should concatenate along the second dimension. Then to do the required shuffling, I apply a big
    # row permutation matrix. 
    # big_coded_matrix = np.stack(coded_A_matrices, axis = 2)
    # print(big_coded_matrix.shape)
    # transposed_big_coded_matrix = np.transpose(big_coded_matrix)
    # print(transposed_big_coded_matrix.shape)

    big_coded_matrix = np.concatenate(coded_A_matrices, axis = 0)
    # print(big_coded_matrix.shape)
    # create the row permutation matrix
    permutation = np.zeros((72,72))
    for idx,permute in enumerate(np.random.permutation(np.arange(72))):
        permutation[idx][permute] = 1
    
    big_coded_B_matrices = np.matmul(permutation, big_coded_matrix)
    B_matrices = np.array_split(big_coded_B_matrices, NUM_OF_DIVISIONS)
    print(len(B_matrices))



    # print(np.random.permutation(np.arange(9)))
    # identity = np.identity(9)
    # for i in np.random.permutation(np.arange(9)):
        
    # print(transposed_big_coded_matrix.shape)
    # original_matrix = np.transpose(transposed_big_coded_matrix, axes = (2,1,0))
    # print(original_matrix.shape)






    test_solution_vec = [ np.matmul(A_matrix, x) for A_matrix in A_matrices ]
    # print(len(A_matrices))
    # print(len(test_solution_vec))
    test_solution = np.concatenate(test_solution_vec)
    # print(test_solution.shape)
    # print(np.allclose(test_solution, reference_solution))
