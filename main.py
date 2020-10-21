from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg

def singular_value_decomposition_error_study(m: int, n: int, k: int, order: np.array = None, H: np.matrix = None) -> None:
    '''
    Parameters:
    ---
    m: int, the number of rows in the matrix H
    n: int, the number of columns in the matrix H
    k: int, the number of iterations of H*v to simulate
    order: np.array, the order of singular values to create the low rank approximation of matrix H
    H: np.matrix, the matrix to approximate.

    ---
    Returns:
    ---
    None
    
    ---
    Example:
    ---
    singular_value_decomposition_error_study(100, 60, 10000)

    ---
    Description:
    ---

    A random matrix H is defined. If H is not supplied
    k random vectors v are defined

    Perform the singular value decomposition of H.

    Create a low rank approximation of H, H_low_rank.

    The low rank approximation is built in the order of largest to smallest singular values
    unless an order is explicitly provided.

    Find the upper bound of the 2-norm error between H and H_low_rank.
    Defined in the text 'Linear Algebra - Liesen', also in my obsidian notes
    under 'Singular Value Decomposition'.

    Evaluate the actual error between the analytical solution Hv and the approximate solution
    H_low_rank * v. Using the following method.

    ||H * v - H_low_rank * v||/||v||

    A comparison is made between the theoretical upper bound of the error imposed by the low rank
    approximation of the singular value decomposition and the expected error from selecting vectors v
    at random.
    ---
    Author: Kevin Exton
    Date: 2020-10-21

    '''
    # Error Checking
    if order is not None and len(order) != n:
        print(f'the number of elements in order should be {n}.')
        return None
    
    if H is not None:
        if H.shape[0] > m or H.shape[1] > n:
            print(f'the dimensions of H need to be {m} x {n}.')
            return None

    # define H and v.

    if H is None:
        H = (10*np.random.random((m,n))-5*np.ones((m,n))) + (10j*np.random.random((m,n))-5j*np.ones((m,n)))

    # and v is a random vector of the appropriate dimensions.
    # Actually if I define v as a massive matrix and then iterate through the columns that
    # would give me K iterations to look at min,max and average error.

    # K = 10000

    v = (10*np.random.random((n,k))-5*np.ones((n,k))) + (10j*np.random.random((n,k))-5j*np.ones((n,k)))

    # Now I need the singular value decomposition of H
    U, s, Vh = linalg.svd(H)

    # And I can create a low rank approximation of H
    # I can loop through all the ranks of a full rank matrix and create a list of error statistics

    # First create the list of error statistics it will be a K list of named tuples
    ErrorStatistic = namedtuple('ErrorStatistic', 'error_bound min_error mean_error max_error')
    error_statistics = []

    # Create a variable to choose the order of the singular values to construct the
    # low rank approximation
    # order = np.random.choice(np.arange(n), n, replace=False)
    if order is None:
        order = np.arange(n)

    for rank in range(1,n+1):
        selection = order[0:rank]
        # print(np.matrix([U[:,i] for i in selection]).T.shape)

        # Then constructing the matrix as U_low_rank \in C^{n,10} times Sigma_low_rank \in R^{10,10} times Vh_low_rank \in C^{10,m}
        H_low_rank = np.matmul(
            np.matmul(
                np.matrix([U[:,i] for i in selection]).T,
                linalg.diagsvd([s[i] for i in order],U.shape[0],Vh.shape[0])[0:rank,0:rank]
            ), np.matrix([Vh[i,:] for i in selection])
        )
        # The error in the 2 norm sense is going to be ||H*v - H_low_rank*v||
        # And the theoretical upper bound (I think) of that error is going to be the largest excluded singular value
        # First we find that singular value

        error_bound = max( s[i] for i in (set(order) - set(selection)) ) if rank < len(s) else 0
        # error_bound = max(set(s) - set(s[i] for i in order)) if rank < len(s) else 0

        # error_bound = s[rank] if rank < len(s) else 0 # It's RANK not RANK+1 because of python's 0 indexing
        # print(f'Error Bound is: {error_bound}')

        # Now evaluate the error directly normalised by the magnitude of v ||H*v - H_low_rank*v||/||v||
        # (Wait is the norm a linear process can I do this?)

        # iterate through the loop K times and find the min, max and average.
        error_vec = [np.linalg.norm(np.matmul(H,v[:,i]) - np.matmul(H_low_rank, v[:,i]), ord=2)/np.linalg.norm(v[:,i], ord=2) for i in range(k)]
        error_statistics.append(ErrorStatistic(error_bound, min(error_vec), sum(error_vec)/k, max(error_vec)))

    # I'm curious about what the distribution looks like. So I'm going to run a single error vec iteration. with a fixed rank estimation
    # This will be repeating one of the iterations in the above loop which adds redundancy, but I don't think it's a big deal here.
    rank = 3
    selection = order[0:rank]
    H_low_rank = np.matmul(
        np.matmul(
            np.matrix([U[:,i] for i in selection]).T,
            linalg.diagsvd([s[i] for i in order],U.shape[0],Vh.shape[0])[0:rank,0:rank]
        ), np.matrix([Vh[i,:] for i in selection])
    )
    error_vec = [np.linalg.norm(np.matmul(H,v[:,i]) - np.matmul(H_low_rank, v[:,i]), ord=2)/np.linalg.norm(v[:,i], ord=2) for i in range(k)]

    # Below I'm going to try and plot this error_statistics madness
    fig, axs = plt.subplots(2,1) # Create a figure containing a single axes.
    axs[0].plot(range(1,n+1),[error_statistics[i].error_bound for i in range(n)])
    axs[0].errorbar(
        range(1,n+1),
        [error_statistics[i].mean_error for i in range(n)],
        [
            [error_statistics[i].mean_error - error_statistics[i].min_error for i in range(n)],
            [error_statistics[i].max_error - error_statistics[i].mean_error for i in range(n)]
        ],
        fmt='or',
        capsize=3)
    # Add axs labels
    axs[0].set_ylabel('Error Magnitude in the 2-norm')
    axs[0].set_xlabel('Rank of SVD Approximation')
    axs[0].set_title('Error Bounds Comparison between upper bound and \'real\' error.')

    # Plot the Histogram
    axs[1].hist(error_vec, bins=1000)

    # Add axs labels
    axs[1].set_ylabel('No. of Vectors')
    axs[1].set_xlabel(f'Error Magnitude in the 2-norm of rank {rank} approximation')
    axs[1].set_title(f'Error Magnitude Distribution in the 2-norm of rank {rank} approximation')

    # Label the figure
    fig.suptitle('A Study of the Error Performance of the Singular Value Decomposition in the 2-norm')
    plt.show()


def two_norm_of_matrix(a_mat: np.matrix) -> bool:
    """
    Parameters:
    ---

    a_mat: np.matrix

    ---
    Returns:
    ---
    boolean

    ---
    Example:
    ---
    x = two_norm_of_matrix(a_mat)

    ---
    Description:
    ---
    From the definition of the SVD as I've outlined in my note [2].
    There is a property of tall matrices that falls out.

    ||Ax||_{2}^{2} = \lambda ||x||_{2}^{2}

    A nice proof is outlined in [1] but just for future reference
    (in case math stackexchange disappears forever) I'll write it out again here.

    Begin Quote:

    From the SVD of A = UDV^{T} we can see that the eigenvalues
    of A^{T}A = VD^{2}V^{T} are just squared ones from A. At the same time
    the columns of V are the eigenvectors of A^{T}A. So, exploiting orthogonality of
    these eigenvectors.

    ||Ax||_{2}^{2} = ||UDVx||_{2}^{2} = ||D(Vx)||_{2}^{2} = ||De_{\lambda}||x||||_{2}^{2} = ||\sqrt{\lambda}||x||||_{2}^{2} = \lambda ||x||_{2}^{2}

    My calculation is slightly different from this definition because I don't square these values.
    This doesn't really make a big difference for the exercise below because
    the 2-norm is positive definite, so negative numbers are not a problem.

    End Quote
    ---
    References:
    ---
    [1] https://math.stackexchange.com/questions/2165249/2-norm-of-a-matrix
    [2] Obsidian Note: Singular Value Decomposition
    ---
    Author: Kevin Exton
    Date: 2020-10-16
    """
    a_mat_2 = np.matmul(np.asmatrix(a_mat).H, np.asmatrix(a_mat))
    eigval, eigvec = np.linalg.eigh(a_mat_2)
    x = np.asmatrix(eigvec[:,0])

    y = np.matmul(a_mat, x)
    norm_y = np.linalg.norm(y)

    norm_x = np.sqrt(eigval[0])*np.linalg.norm(x)

    return np.allclose(norm_y, norm_x, rtol=1e-5)


if __name__ == "__main__":
    # a = np.random.randint(1,10,size=(4,2)) + 1j*np.random.randint(1,10,size=(4,2))
    # print(two_norm_of_matrix(np.asmatrix(a)))

    singular_value_decomposition_error_study(500,300,10000,np.random.choice(np.arange(300),300,replace=False))
