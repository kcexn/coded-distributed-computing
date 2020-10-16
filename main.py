import numpy as np

def two_norm_of_matrix(a_mat: np.matrix) -> bool:
    '''
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

    A nice proof is outlined in [1] but just for future reference (in case math stackexchange disappears forever)
    I'll write it out again here.

    Begin Quote:
    
    From the SVD of A = UDV^{T} we can see that the eigenvalues of A^{T}A = VD^{2}V^{T} are just squared
    ones from A. At the same time the columns of V are the eigenvectors of A^{T}A. So, exploiting orthogonality of
    these eigenvectors.

    ||Ax||_{2}^{2} = ||UDVx||_{2}^{2} = ||D(Vx)||_{2}^{2} = ||De_{\lambda}||x||||_{2}^{2} = ||\sqrt{\lambda}||x||||_{2}^{2} = \lambda ||x||_{2}^{2}

    My calculation is slightly different from this definition because I don't square these values.
    This doesn't really make a big difference for the exercise below because the 2-norm is positive definite, so negative
    numbers are not a problem.

    End Quote
    ---
    References:
    ---
    [1] https://math.stackexchange.com/questions/2165249/2-norm-of-a-matrix
    [2] Obsidian Note: Singular Value Decomposition
    ---
    Author: Kevin Exton
    Date: 2020-10-16
    '''
    a_mat_2 = np.matmul(np.asmatrix(a_mat).H, np.asmatrix(a_mat))
    eigval, eigvec = np.linalg.eigh(a_mat_2)
    x = np.asmatrix(eigvec[:,0])

    y = np.matmul(a_mat, x)
    norm_y = np.linalg.norm(y)

    norm_x = np.sqrt(eigval[0])*np.linalg.norm(x)

    return np.allclose(norm_y, norm_x, rtol=1e-5)


if __name__ == "__main__":
    a = np.random.randint(1,10,size=(4,2)) + 1j*np.random.randint(1,10,size=(4,2))
    print(two_norm_of_matrix(np.asmatrix(a)))
