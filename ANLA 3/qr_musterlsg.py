import numpy as np

def givens_qr(A):
    m, n = A.shape
    G = np.eye(m, dtype=complex)
    R = A.copy().astype(complex)

    print("Original matrix A:")
    print(R)

    for i in range(min(m, n)):
        for j in range(i + 1, m):
            # Apply Givens rotation to eliminate R[j, i]
            norm = np.linalg.norm([complex(R[i, i]), complex(R[j, i])])
            c = R[i, i] / norm
            s = R[j, i] / norm

            # Update R matrix using G.T
            temp = c * R[i, :] + s * R[j, :]
            R[j, :] = -np.conj(s) * R[i, :] + np.conj(c) * R[j, :]
            R[i, :] = temp

            # Update only the relevant part of G
            temp = c * G[i, :m] + s * G[j, :m]
            G[j, :m] = -np.conj(s) * G[i, :m] + np.conj(c) * G[j, :m]
            G[i, :m] = temp

    print("\nThe G matrix:")
    print(G)
    print("\nThe final R matrix:")
    print(R)

    return (G, R)

def form_q(G):
    Q = G.conj().T  # Q is the conjugate transpose of G
    print("\nThe final Q matrix:")
    print(Q)
    return Q