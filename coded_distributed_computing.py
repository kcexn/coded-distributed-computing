''' coded_distributed_computing
This module contains functions related to a study of the coded distributed computing model.

'''
import numpy as np

def encode_matrix(A: np.matrix, G: np.matrix) -> np.matrix:
    ''' encode_matrix
    Parameters:
    ---
    A: np.matrix, input matrix to code.
    G: np.matrix, generator matrix to encode A with.
    ---
    Returns:
    ---
    A*G: np.matrix, output encoded matrix.
    ---
    Description:
    ---
    Following van Lint's text "Introduction to Coding Theory", 
    I am constructing linear block codes using a generator matrix G 
    and an input matrix A. 

    Actually typically the codes would be constructed using a 
    generator matrix G and an input vector k which would create an 
    output message, a vector, m.

    Following from my conversation with Jingge last week though. 
    I'm convinced that encoding a matrix to preserve the 
    matrix vector multiplication Ax is exactly the same as encoding
    multiple messages across time simultaneously. i.e. If I were to 
    accumulate n messages (column vectors) of size k and concatenated them 
    I would end up with a matrix of size k x n (rows and columns). Encoding 
    it with the generator matrix G would give me a matrix of size m x n. Where
    each column in the matrix A*G can be considered one message to be delivered 
    over time. The matrix vector multiplication Ax is simply the rows of multiple
    messages concatenated together multiplied with the vector x.

    This is not a super great analogue, because obviously matrices in a matrix vector 
    multiplication are shared with everyone all at once not one column at a time. 
    But I think it's a useful way to reason about the coding properties of 
    the matrix A*G. And I believe opens up the possibilities of 
    matrix encodings to ALL codes that can be represented as linear block codes 
    (which I believe are simply, ALL linear codes).

    '''
    return np.matmul(A,G)



