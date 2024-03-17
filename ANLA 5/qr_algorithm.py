import numpy as np
from scipy.linalg import hilbert
import matplotlib.pyplot as pl


def tridiag(A):
    m, n = A.shape
    T = np.copy(A)

    for k in range(n - 2):  # Iterate up to the second-to-last column
        x = T[k+1:, k]  # Subvector below the diagonal
        v = x.copy()
        v[0] = x[0] + np.sign(x[0]) * np.linalg.norm(x)
        v = v / np.linalg.norm(v)

        # Apply Householder transformation to the remaining submatrix
        P = np.eye(m)
        P[k+1:m, k+1:n] -= 2.0 * np.outer(v, v) / np.dot(v, v)
        T = np.dot(P.T, np.dot(T, P))

    return T

def QR_alg(T):
    t = []
    while np.linalg.norm(T - np.diag(np.diag(T))) > 1e-10:
        Q, R = np.linalg.qr(T)
        T = R @ Q
        tm_m1 = abs(T[-1, -2])
        t.append(tm_m1)
        if tm_m1 < 1e-12:
            break
    return (T, t)

def wilkinson_shift(T):

    μ = 0
    # todo
    m, n = T.shape

    delta = T[-1, -1] - T[-2, -2]
    b_m1 = T[-1, -2]

    sign_delta = 1.0 if delta >= 0 else -1.0

    μ = T[-1, -1] - (sign_delta * (b_m1 ** 2)) / (abs(delta) + np.sqrt(delta ** 2 + b_m1 ** 2))
    return μ


def QR_alg_shifted(T):
    t = []
    # todo
    while np.linalg.norm(T - np.diag(np.diag(T))) > 1e-10:
        # Applying the provided wilkinson_shift function
        mu = wilkinson_shift(T)

        # QR decomposition with shifted eigenvalue
        Q, R = np.linalg.qr(T - mu * np.identity(T.shape[0]))

        # Update T for the next iteration
        T = R @ Q + mu * np.identity(T.shape[0])

        # Store the diagonal elements of the current T
        t.append(np.diag(T).tolist())
    return (T, t)


def QR_alg_driver(A, shift):
    all_t = []
    Λ_list = []  # Use a list to accumulate diagonal elements

    while np.linalg.norm(np.triu(tridiag(A), k=1)) > 1e-10:
        if shift:
            T, t = QR_alg_shifted(A)
        else:
            T, t = QR_alg(A)

        # Append the entire list t to all_t
        all_t.extend(t)

        # Accumulate diagonal elements in the list
        Λ_list.append(np.diag(T).tolist())  # Use T instead of A

        # Update A for the next iteration
        A = T

        # Accumulate diagonal elements in the list
        Λ_list.append(np.diag(A).tolist())  # Use A instead of T

    # Concatenate the accumulated diagonal elements after the loop
    #Λ = np.concatenate(Λ_list)
    Λ = np.array(Λ_list[0]).reshape(-1)

    return (Λ, all_t)


if __name__ == "__main__":

    matrices = {
        "hilbert": hilbert(4),
        "diag(1,2,3,4)+ones": np.diag([1, 2, 3, 4]) + np.ones((4, 4)),
        "diag(5,6,7,8)+ones": np.diag([5, 6, 7, 8]) + np.ones((4, 4)),
    }

    fig, ax = pl.subplots(len(matrices.keys()), 2, figsize=(10, 10))

    for i, (mat, A) in enumerate(matrices.items()):
        print(f"A = {mat}")
        Λ,_ = np.linalg.eig(A)
        print(f"Λ = {np.sort(Λ)}\n")
        for j, shift in enumerate([True, False]):
            Λ, conv = QR_alg_driver(A.copy(), shift)
            ax[i, j].semilogy(range(len(conv)), conv, ".-")
            ax[i, j].set_title(f"A = {mat}, shift = {shift}")

    pl.show()