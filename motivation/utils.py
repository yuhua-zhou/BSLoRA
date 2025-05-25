import numpy as np
import torch.nn.functional as F


def cka_similarity(X, Y):
    def centered_gram(K):
        n = K.shape[0]
        unit = np.ones([n, n])
        return K - unit.dot(K) / n - K.dot(unit) / n + unit.dot(K).dot(unit) / n ** 2

    def frobenius_norm(K):
        return np.sqrt(np.sum(K ** 2))

    Kx = X.T.dot(X)
    Ky = Y.T.dot(Y)

    gKx = centered_gram(Kx)
    gKy = centered_gram(Ky)

    cka = frobenius_norm(gKx.T.dot(gKy)) / (frobenius_norm(gKx) * frobenius_norm(gKy))

    return cka


def cosine_similarity(X, Y):
    X = X.flatten()
    Y = Y.flatten()

    max_len = max(X.shape[0], Y.shape[0])

    X = F.pad(X, (0, max_len - X.shape[0]), "constant", 0)
    Y = F.pad(Y, (0, max_len - Y.shape[0]), "constant", 0)

    norm_X = np.linalg.norm(X)
    norm_Y = np.linalg.norm(Y)

    # 计算矩阵内积
    dot_product = np.dot(X, Y)

    # 计算余弦相似度
    cosine_sim = dot_product / (norm_X * norm_Y.T)

    return cosine_sim


def frobenius_norm(X, Y):
    Z = X - Y
    norm_Z = np.linalg.norm(Z)

    norm_X = np.linalg.norm(X)
    norm_Y = np.linalg.norm(Y)

    # print(norm_X, norm_Y)

    # return norm_Z / (norm_X, norm_Y)
    return norm_Z


def entropy_similarity(X, Y):
    X = X.flatten()
    Y = Y.flatten()

    entropy_X = compute_matrix_entropy(X)
    entropy_Y = compute_matrix_entropy(Y)
    entropy_XY = compute_matrix_entropy(np.concatenate([X, Y]))
    mutual_entropy = entropy_X + entropy_Y - entropy_XY

    # print(entropy_X, entropy_Y, entropy_XY)
    # print(mutual_entropy, np.mean([entropy_X, entropy_Y]))

    RMI = mutual_entropy / np.min([entropy_X, entropy_Y])

    return RMI


# compute the entropy
def compute_matrix_entropy(X):
    # X = X.flatten()
    # var_X = np.std(X)
    # entropy_X = np.log(var_X) + 0.5 * np.log(2 * np.pi) + 0.5
    # return entropy_X

    """
    Approximates the entropy of a vector assuming a Gaussian distribution.

    Parameters:
    vector (numpy array): The input vector

    Returns:
    float: The approximate entropy
    """
    # Reshape the vector if it's a 1D vector to treat it as a row vector
    if len(X.shape) == 1:
        vector = X.reshape(-1, 1)

    # Compute the covariance matrix of the vector
    cov_matrix = np.cov(vector, rowvar=False)  # Compute the covariance matrix
    cov_matrix = np.atleast_2d(cov_matrix)  # Ensure it's a 2D matrix

    # Compute the determinant of the covariance matrix
    det_cov = np.linalg.det(cov_matrix)

    # Get the dimension of the vector
    n = vector.shape[1]

    # Handle the case where the determinant is very small (or negative due to precision)
    if det_cov <= 0:
        det_cov = np.finfo(float).eps  # Use a small epsilon to avoid log(0) or negative determinant

    # Calculate the Gaussian entropy
    entropy = 0.5 * np.log((2 * np.pi * np.e) ** n * det_cov)

    return entropy
