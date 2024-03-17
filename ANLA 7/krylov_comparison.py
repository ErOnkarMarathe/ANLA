import enum
import numpy as np
from scipy.linalg import solve_triangular
import matplotlib.pyplot as pl

#from krylov_musterlsg import cg, gmres, gmres_givens
from krylov import cg, gmres, gmres_givens


def create_A(k, ω):
    np.random.seed(17)
    D = np.diag(sum([[i] * i for i in range(1, k + 1)], []))
    m = D.shape[0]
    M = np.random.rand(m, m) - 0.5
    return D + ω * M

def preconditioner(A, alt=False):
    if alt:
        return np.triu(A)
    else:
        return np.diag(np.diag(A))


def magic(A, b, tol=1e-12):
    P = preconditioner(A)
    #! you should normally not do it like this!
    b_tilde = A.T @ solve_triangular(P.T, solve_triangular(P, b), lower=True)
    A_tilde = A.T @ solve_triangular(P.T, solve_triangular(P, A), lower=True)
    return cg(A_tilde, b_tilde, tol=tol)


def solve_benchmark(A):
    b = np.ones(A.shape[0])
    P = preconditioner(A)

    conv = {}
    sol = {}

    sol["gmres"], conv["gmres"] = gmres(A.copy(), b.copy(), tol=1e-12)
    sol["gmres_G"], conv["gmres_G"] = gmres_givens(A.copy(), b.copy(), tol=1e-12)
    sol["gmres_p"], conv["gmres_p"] = gmres(A.copy(), b.copy(), P=P.copy(), tol=1e-12)
    sol["magic"], conv["magic"] = magic(A.copy(), b.copy(), tol=1e-12)
    sol["cg"], conv["cg"] = cg(A.copy(), b.copy(), tol=1e-12)

    print(f"{'-'*22}")
    print(f"{' '*10} ||r||/||b||")
    for alg, x in sol.items():
        r_b = np.linalg.norm(A @ x - b) / np.linalg.norm(b)
        print(f"{alg:{10}} {r_b:{11}.{3}}")
    print(f"{'-'*22}")

    return conv


def condition(A):
    _, Σ, _ = np.linalg.svd(A)
    return Σ[0] / Σ[-1]


def plot_eigenvalues(A):
    _, ax = pl.subplots(1, 1)
    Λ = np.linalg.eigvals(A)
    ax.scatter(np.real(Λ), np.imag(Λ))
    ax.set_title("Eigenvalues of A")
    ax.set_xlabel("Re")
    ax.set_ylabel("Im")
    # pl.show()


if __name__ == "__main__":
    Ω = [0, 0.1, 1]
    fig, ax = pl.subplots(len(Ω), 1)
    fig.tight_layout()
    for s, ω in enumerate(Ω):
        A = create_A(30, ω)
        conv = solve_benchmark(A)
        for name, r_b in conv.items():
            ax[s].semilogy(r_b, ".-", label=name)
            ax[s].legend()
            ax[s].set_xlabel("iterations")
            ax[s].set_ylabel("||r||/||b||")
            ax[s].set_title(f"solve  (D+ωM)x = b  with ω={ω}")
    pl.show()
