import numpy as np


def implicit_qr(A):
    W = None
    R = None
    # TODO
    # logic
    # x = A_(k:m,k)
    # v_k = sign(x1)||x||_2*e_1*x
    # v_k = v_k/||v_k||_2
    # A_(k:m,k:n) = A_(k:m,k:n) -2*v_k(v_k.T *A_(k:m,k:n))
    m, n = A.shape
    W = np.eye(m, n, dtype=complex)
    R = A.copy().astype(complex)

    for k in range(n):
        x = R[k:m, k]
        sign_x = x[0] / np.abs(x[0])
        v_k = x + sign_x * np.linalg.norm(x) * np.eye(len(x), dtype=complex)[:, 0]
        if np.sqrt(np.sum(v_k * np.conj(v_k))) != 0:
            v_k = v_k / np.sqrt(np.sum(v_k * np.conj(v_k)))
            v_k = v_k.astype(complex)  # Ensure v_k is complex
            R[k:m, k:n] = R[k:m, k:n] - 2 * np.outer(v_k, np.conj(v_k)).dot(R[k:m, k:n])
        W[k:m, k] = v_k

    return (W, R)


def form_q(W):
    Q = None
    # TODO
    # logic
    # for k = n down to 1
    # x_(k:m) = x_(k:m) - 2(v_k.T *x(k:m))
    m, n = W.shape
    Q = np.eye(m, dtype=complex)

    for k in range(n - 1, -1, -1):
        v_k = W[k:m, k]
        x_k = Q[k:m, k:]
        Q[k:m, k:] -= 2 * np.outer(v_k, np.conj(v_k)).dot(x_k)

    return Q
