import numpy as np
from scipy.linalg import solve_triangular
from numpy.linalg import qr

def cg(A, b, tol=1e-12):
    m = A.shape[0]
    x = np.zeros(m, dtype=A.dtype)
    r = b - A @ x
    p = r.copy()
    r_b = [np.linalg.norm(r) / np.linalg.norm(b)]  # Initial relative norm

    for k in range(m):
        alpha = np.dot(r, r) / np.dot(p, A @ p)
        x = x + alpha * p
        r_old = r.copy()
        r = r - alpha * A @ p
        beta = np.dot(r, r) / np.dot(r_old, r_old)
        p = r + beta * p

        r_b.append(np.linalg.norm(r) / np.linalg.norm(b))

        if r_b[-1] < tol:
            break

    return x, r_b

def arnoldi_n(A, Q, P):
    # n-th step of arnoldi
    m, n = Q.shape
    q = np.zeros(m, dtype=Q.dtype)
    h = np.zeros(n + 1, dtype=A.dtype)

    # For n = 1,2,3,...
    if n == 0:
        # b = arbitrary, q1 = b/||b||
        q = P / np.linalg.norm(P)
    else:
        # v = Aq_n
        v = np.dot(A, Q[:, n-1])

        # for j = 1 to n
        for j in range(n):
            # h_jn = (q*_j) * v
            h[j] = np.dot(np.conj(Q[:, j]), v)

            # v = v - (h_jn * q_j)
            v = v - h[j] * Q[:, j]

        # h_n+1, n = ||v||
        h[n] = np.linalg.norm(v)

        # q_n+1 = v / h_(n+1, n)
        q = v / h[n]

    return h, q
    #Arnoldi algorithm
    # b = arbitary, q1 = b/||b||
    # for n = 1,2,3,....
        #v = Aq_n
            #for j = 1 to n
                #h_jn = (q*_j)* v
                #v = v -(h_jn* q_j)
            #h_n+1,n = ||v||
            #q_n+1 =v/h_(n+1,n)


def gmres(A, b, P=np.eye(1), tol=1e-12):
    m = A.shape[0]

    if P.shape[0] != A.shape[0]:
        # Default preconditioner P = I
        P = np.eye(m)

    x = np.zeros(m, dtype=b.dtype)
    r_b = [1]
    b = solve_triangular(P, b)
    normb = np.linalg.norm(b)

    Q = np.c_[b / normb]
    H_New = np.zeros((1, 0), dtype=A.dtype)

    for j in range(m):
        h, q = arnoldi_n(A, Q, P)

        if h[Q.shape[1]] == 0:
            break

        H_New = np.r_[H_New, np.zeros((1, H_New.shape[1]))]
        H_New = np.c_[H_New, h]
        Q = np.c_[Q, q]

        QQ, R = qr(H_New, mode="reduced")
        Z = QQ[0, :].conj() * normb
        y = solve_triangular(R[:j + 1, :j + 1], Z[:j + 1])

        x = Q[:, :-1] @ y
        r = Q @ (H_New @ y) - b
        r_b.append(np.linalg.norm(r) / normb)

        if r_b[-1] < tol:
            break

    return x, r_b


def gmres_givens(A, b, P=np.eye(0), tol=1e-12):
    m = A.shape[0]

    if P.shape != A.shape:
        # Default preconditioner P = I
        P = np.eye(m)

    x = np.zeros(m, dtype=b.dtype)
    r_b = [1]

    b = solve_triangular(P, b)
    normb = np.linalg.norm(b)

    Q = np.c_[b / normb]
    H_New = np.zeros((1, 0), dtype=A.dtype)

    for j in range(m):
        h, q = arnoldi_n(A, Q, P)

        if h[Q.shape[1]] == 0:
            break

        H_New = np.r_[H_New, np.zeros((1, H_New.shape[1]))]
        H_New = np.c_[H_New, h]
        Q = np.c_[Q, q]

        QQ, R = qr(H_New, mode="reduced")
        Z = QQ[0, :].conj() * normb
        y = solve_triangular(R[:j + 1, :j + 1], Z[:j + 1])

        x = Q[:, :-1] @ y
        r = Q @ (H_New @ y) - b
        r_b.append(np.linalg.norm(r) / normb)

        if r_b[-1] < tol:
            break

    return x, r_b
