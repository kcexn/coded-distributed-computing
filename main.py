import numpy as np

from coded_distributed_computing import encode_matrix


if __name__ == "__main__":
    A = np.matrix('1 0; 0 0')
    G = np.matrix('1 0 ; 0 1; 1 1')
    AG = encode_matrix(G,A)
    print(AG)
    