from sys import stderr,exit
from typing import List
import random
from math import ceil

import numpy as np
from numpy.random import default_rng
from scipy import linalg

from coded_distributed_computing import encode_matrix

class NoSolution(Exception):
    pass

def gauss_decoder(generator: np.matrix, coded_message: np.array) -> np.array:
    # to make this matrix 1 at the top and linearly dependent rows at the bottom
    transposed_generator = np.transpose(np.copy(generator))

    # capture erasures that happened in the message part of the matrix
    # for decoding
    erasure_count = 0
    for j in range(transposed_generator.shape[1]):
        if coded_message[j][0] == 0:
            transposed_generator[j][j] = 0
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
    
    # print(erasure_count)
    # print(erasure_set)

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
    # Need to do the rows before I do the columns.
    for j in range(transposed_generator.shape[1]):
        for pivoted_row in pivoted_rows:
            if j != pivoted_row:
                coded_message[pivoted_row][0] -= coded_message[j][0]*(transposed_generator[pivoted_row][j]/transposed_generator[j][j])
                if j in pivoted_rows:
                    for k in reversed(range(j,transposed_generator.shape[1])):
                        transposed_generator[pivoted_row][k] -= transposed_generator[j][k]*(transposed_generator[pivoted_row][j]/transposed_generator[j][j])
                # transposed_generator[expivoted_row] = transposed_generator[pivoted_row]*(transposed_generator[pivoted_row][j]/transposed_generator[j][j])

    for pivoted_row in pivoted_rows:
        # scale the row appropriately in the matrix (element in the message)
        coded_message[pivoted_row][0] = coded_message[pivoted_row][0]/transposed_generator[pivoted_row][pivoted_row]

    return coded_message[0:transposed_generator.shape[1]][:]


if __name__ == "__main__":
    K = 18
    N = 30
    SCALE_FACTOR = 2**10
    NUM_OF_MATRICES = 1
    NUM_OF_PROCESSORS = 6
    MAX_NUM_OF_DIVISIONS = K//NUM_OF_MATRICES + 1
    NUM_OF_DIVISIONS = 2
    # In reverse order so that I can pop them off like a stack.
    # Actually this can probably be better implemented as a deque
    SEQUENCE_CONSTRAINT_DIVISORS = [3,3]
    NUM_OF_ERASURES = 3

    rng = default_rng()

    try:
        assert len(SEQUENCE_CONSTRAINT_DIVISORS) == NUM_OF_DIVISIONS
    except AssertionError as error:
        print(f'The number of constraint divisors is not equal to the number of divisions. NUM_OF_DIVISIONS: {NUM_OF_DIVISIONS}.')
        exit(1)

    try:
        assert NUM_OF_DIVISIONS < MAX_NUM_OF_DIVISIONS
    except AssertionError as error:
        print(f'The maximum number of matrix divisions is {MAX_NUM_OF_DIVISIONS}. NUM_OF_DIVISIONS is too high.', file=stderr)
        exit(1)

    H = SCALE_FACTOR*np.random.random((K,N))-(SCALE_FACTOR/2)*np.ones((K,N))
    x = SCALE_FACTOR*np.random.random((N,1))
    
    reference_solution = np.matmul(H,x)

    A_matrices = np.array_split(H, NUM_OF_DIVISIONS)

    # Create the list of generator matrices
    # based on the sequential properties that are desired.
    generator_matrices = []
    for A_matrix in A_matrices:
        sequence_constraint = SEQUENCE_CONSTRAINT_DIVISORS.pop()
        total_num_of_rows = ceil(A_matrix.shape[0] * NUM_OF_PROCESSORS / sequence_constraint)
        total_num_of_redundant_rows = total_num_of_rows - A_matrix.shape[0]
        encoding = SCALE_FACTOR*np.random.random((A_matrix.shape[0],  total_num_of_redundant_rows))
        generator = np.concatenate(((np.identity(A_matrix.shape[0])), encoding), axis = 1)
        generator_matrices.append(generator)
        SEQUENCE_CONSTRAINT_DIVISORS.insert(0, sequence_constraint)
    
    coded_A_matrices = [np.transpose(np.matmul(np.transpose(A_matrix), generator)) for A_matrix,generator in zip(A_matrices, generator_matrices)]

    big_coded_A_matrix = np.concatenate(coded_A_matrices, axis = 0)


    # create the row permutation matrix
    permutation = np.zeros((big_coded_A_matrix.shape[0],big_coded_A_matrix.shape[0]))
    permutation_list = np.array_split(permutation, NUM_OF_PROCESSORS)
    row_counter = 0
    for i,A_matrix in enumerate(coded_A_matrices):
        sequence_constraint = SEQUENCE_CONSTRAINT_DIVISORS.pop()
        # Evaluate how to divide up the rows.
        smaller_number_of_rows = A_matrix.shape[0]//NUM_OF_PROCESSORS
        num_of_small_rows = NUM_OF_PROCESSORS - A_matrix.shape[0]%NUM_OF_PROCESSORS
        larger_number_of_rows = ceil(A_matrix.shape[0]/NUM_OF_PROCESSORS)
        num_of_large_rows = A_matrix.shape[0]%NUM_OF_PROCESSORS

        # permutation_list j represents the j*(big_coded_A_matrix.shape[0]//9)th element if
        # divisible otherwise there's the funny division again.
        for j in range(A_matrix.shape[0]):
            if j < num_of_small_rows*smaller_number_of_rows:
                for row in permutation_list[j//smaller_number_of_rows]:
                    if np.sum(row) == 0:
                        row[row_counter + j] = 1
                        break
                else:
                    print('Infeasible configuration for evenly sized matrices?')
            elif j - num_of_small_rows*smaller_number_of_rows < num_of_large_rows*larger_number_of_rows:
                for row in permutation_list[j//larger_number_of_rows]:
                    if np.sum(row) == 0:
                        row[row_counter + j] = 1
                        break
                else:
                    print('Infeasible configuration for evenly sized matrices?')
        row_counter = row_counter + A_matrix.shape[0]
        permutation_list = permutation_list[::-1]
        SEQUENCE_CONSTRAINT_DIVISORS.insert(0, sequence_constraint)

    if big_coded_A_matrix.shape[0]%2 == 1:
        permutation_list = permutation_list[::-1]
    permutation = np.concatenate(permutation_list, axis = 0)


    # # In general this permutation can't be random and instead has to be carefully selected.
    # for idx,permute in enumerate(np.random.permutation(np.arange(72))):
    #     permutation[idx][permute] = 1
    
    big_coded_B_matrix = np.matmul(permutation, big_coded_A_matrix)
    B_matrices = np.array_split(big_coded_B_matrix, NUM_OF_PROCESSORS)
    # print(len(B_matrices))

    encoded_solution = [np.matmul(B_matrix, x) for B_matrix in B_matrices]
    # Introduce erasures
    # print(encoded_solution[random.randint(0,NUM_OF_DIVISIONS-1)])
    for i in np.random.choice(NUM_OF_PROCESSORS, NUM_OF_ERASURES, replace = False):
        for val in encoded_solution[i]:
            val[0] = 0
    # for val in encoded_solution[random.randint(0,NUM_OF_DIVISIONS-1)]:
    #     val[0] = 0
    # for solution in encoded_solution:
    #     print(solution)

    big_encoded_solution = np.concatenate(encoded_solution, axis = 0)
    unpermuted_big_encoded_solution = np.matmul(np.transpose(permutation), big_encoded_solution)
    unpermuted_encoded_solution = np.array_split(unpermuted_big_encoded_solution, NUM_OF_DIVISIONS)
    for solution in unpermuted_encoded_solution:
        print(solution)
    
    words = [gauss_decoder(generator, code_word) for (generator, code_word) in zip(generator_matrices, unpermuted_encoded_solution)]
    # for idx,code_word in enumerate(unpermuted_encoded_solution):
    #     if idx < K%NUM_OF_DIVISIONS:
    #         words.append(gauss_decoder(large_generator, code_word))
    #     else:
    #         words.append(gauss_decoder(generator,code_word))
    solution = np.concatenate(words)
    for i in range(K):
        print(f"{i+1}'th value: {solution[i][0]}:{reference_solution[i][0]}")
    print(solution)
    print(reference_solution)
    print(np.allclose(solution, reference_solution))
