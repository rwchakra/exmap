import numpy as np
import matplotlib.pyplot as plt

from scipy.sparse import csgraph
from scipy.sparse.linalg import lobpcg
from sklearn.manifold._spectral_embedding import _set_diag


def eigenDecomposition(A, plot=True, botK=12, seed=0):
    """
    :param A: Affinity matrix
    :param plot: plots the sorted eigen values for visual inspection
    :param botK: number of eigenvalues to return

    follow same procedure:
    https://github.com/scikit-learn/scikit-learn/blob/364c77e047ca08a95862becf40a04fe9d4cd2c98/sklearn/manifold/_spectral_embedding.py#L367
    (super fast for large sparse matrices)
    """
    random_state = np.random.RandomState(seed)
    eigen_tol = "auto"
    tol = None if eigen_tol == "auto" else eigen_tol

    L, dd = csgraph.laplacian(A, normed=True, return_diag=True)
    L = _set_diag(L, 1, True)

    X = random_state.standard_normal(size=(L.shape[0], botK + 1))
    X[:, 0] = dd.ravel()
    X = X.astype(L.dtype)

    eigenvalues, eigenvectors = lobpcg(L, X, tol=tol, largest=False, maxiter=2000,)
    permutation = eigenvalues.argsort()
    eigenvalues = eigenvalues[permutation]
    eigenvectors = eigenvectors[:, permutation]

    if plot:
        plt.title("Smallest eigenvalues")
        plt.scatter(np.arange(len(eigenvalues)), np.abs(eigenvalues), label=str(plot))
        plt.legend()
        plt.grid()
        plt.show()

    # Identify the optimal number of clusters as the index corresponding
    # to the larger gap between eigen values
    index_largest_gap = np.argsort(np.diff(eigenvalues))[::-1][:botK]
    nb_clusters = index_largest_gap + 1

    return nb_clusters, eigenvalues, eigenvectors
