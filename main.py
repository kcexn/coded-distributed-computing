from sys import stderr,exit
from typing import List

import numpy as np
from scipy import linalg

from coded_distributed_computing import encode_matrix

class NoSolution(Exception):
    pass

def gauss_decoder(generator: np.matrix, coded_message: np.array) -> np.array:
    # to make this matrix 1 at the top and linearly dependent rows at the bottom
    transposed_generator = np.transpose(generator)

    # capture erasures that happened in the message part of the matrix
    # for decoding
    erasure_count = 0
    for i in range(transposed_generator.shape[1]):
        if coded_message[i,0] == 0:
            transposed_generator[i][i] = 0
            erasure_count += 1

    # There were no erasures in the message part of the received
    # word so truncate the message and return the word early.
    if erasure_count == 0:
        return coded_message[0:transposed_generator.shape[1]][:]
    
    # capture erasures that happened in the redundancy part of the matrix for decoding
    erasure_set = set(i for i in range(transposed_generator.shape[1], transposed_generator.shape[0]))
    # print(erasure_set)
    for j in range(transposed_generator.shape[1], transposed_generator.shape[0]):
        if coded_message[j][0] == 0:
            transposed_generator[j] = np.array([0 for i in range(transposed_generator.shape[1])])
            erasure_set = erasure_set - set([j])
    
    if len(erasure_set) < erasure_count:
        raise NoSolution
        
    # Perform the necessary pivots ONLY if the identity matrix
    # at the top is disturbed
    pivoted_rows = set()
    for j in range(transposed_generator.shape[1]):
        if transposed_generator[j][j] == 0:
            # Swap row with 0 array. This way is slightly faster than needing to do multiple
            # deep copies I think.
            redundant_row = erasure_set.pop()
            transposed_generator[j] = transposed_generator[redundant_row]
            transposed_generator[redundant_row] = np.array([0 for k in range(transposed_generator.shape[1])])

            # Match with corresponding swaps in the coded_message array
            coded_message[j][0], coded_message[redundant_row][0] = coded_message[redundant_row][0], 0
            pivoted_rows.add(j)
    
    # Perform row reduction for each row that was pivoted.
    for pivoted_row in pivoted_rows:
        # subtract off all other rows in the matrix (elements in the message)
        for j in range(transposed_generator.shape[1]):
            if j != pivoted_row:
                coded_message[pivoted_row][0] -= coded_message[j][0]*(transposed_generator[pivoted_row][j])
        # scale the row appropriately in the matrix (element in the message)
        coded_message[pivoted_row][0] = coded_message[pivoted_row][0]/(transposed_generator[pivoted_row][pivoted_row])
    
    return coded_message[0:transposed_generator.shape[1]][:]


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
    
    # I'm assuming that random never produces a 0.
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

    big_coded_matrix = np.concatenate(coded_A_matrices, axis = 0)
    # print(big_coded_matrix.shape)
    # create the row permutation matrix
    permutation = np.zeros((72,72))
    for idx,permute in enumerate(np.random.permutation(np.arange(72))):
        permutation[idx][permute] = 1
    
    big_coded_B_matrices = np.matmul(permutation, big_coded_matrix)
    B_matrices = np.array_split(big_coded_B_matrices, NUM_OF_DIVISIONS)
    # print(len(B_matrices))

    encoded_solution = [np.matmul(B_matrix, x) for B_matrix in B_matrices]
    big_encoded_solution = np.concatenate(encoded_solution, axis = 0)
    unpermuted_big_encoded_solution = np.matmul(np.transpose(permutation), big_encoded_solution)
    unpermuted_encoded_solution = np.array_split(unpermuted_big_encoded_solution, NUM_OF_DIVISIONS)

    error_encoded_word = unpermuted_encoded_solution[2][:]
    for i in range(5,8):
        error_encoded_word[:][i] = 0


    # print(reference_solution[:])
    # print(unpermuted_encoded_solution[0][:])
    



    test_solution_vec = [ np.matmul(A_matrix, x) for A_matrix in A_matrices ]
    # print(len(A_matrices))
    # print(len(test_solution_vec))
    test_solution = np.concatenate(test_solution_vec)
    # print(test_solution.shape)
    # print(np.allclose(test_solution, reference_solution))
