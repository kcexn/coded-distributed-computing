import numpy as np
from scipy import linalg

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

    # --- Below here is related to looking at the problems in my obsidian note:
    # Graceful Degradation of the Performance of Matrix Operations to Meet Timing Constraints

    # To look at the upper error bound of a matrix H multiplied by a vector v
    # we first need to define both H and v.

    # For the time being we can define H as a random matrix with a reasonably large set of
    # dimensions.

    # First parametrize the dimensions
    M = 50
    N = 30

    # Then define H and v.

    H = (10*np.random.random((M,N))-5*np.ones((M,N))) + (10j*np.random.random((M,N))-5j*np.ones((M,N)))

    # and v is a random vector of the appropriate dimensions.

    v = (10*np.random.random((N,1))-5*np.ones((N,1))) + (10j*np.random.random((N,1))-5j*np.ones((N,1)))

    # Now I need the singular value decomposition of H
    U, s, Vh = linalg.svd(H)

    # And I can create a low rank approximation of H
    # First defining the rank of the approximation
    RANK = 15

    # Then constructing the matrix as U_low_rank \in C^{n,10} times Sigma_low_rank \in R^{10,10} times Vh_low_rank \in C^{10,m}
    H_low_rank = np.matmul( np.matmul(U[:,0:RANK],linalg.diagsvd(s,U.shape[0],Vh.shape[0])[0:RANK,0:RANK]), Vh[0:RANK,:])
    
    # The error in the 2 norm sense is going to be ||H*v - H_low_rank*v||
    # And the theoretical upper bound (I think) of that error is going to be the largest excluded singular value
    # First we find that singular value
    error_bound = s[RANK] if RANK < len(s) else 0 # It's RANK not RANK+1 because of python's 0 indexing
    print(error_bound)

    # Now evaluate the error directly normalised by the magnitude of v ||H*v - H_low_rank*v||/||v||
    error = np.linalg.norm(np.matmul(H,v) - np.matmul(H_low_rank, v), ord=2)/np.linalg.norm(v, ord=2)
    print(error)
