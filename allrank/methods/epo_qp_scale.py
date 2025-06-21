import numpy as np
from numpy.linalg import norm
from cvxopt import matrix, solvers
solvers.options['show_progress'] = False


def epo_qp_get_alpha_scale(r, f, GG):
    assert len(r) == len(f)
    m = len(f)
    rinv = 1.0 / r
    rinv = rinv / norm(rinv)

    # A_l1 = np.ones((1, m))
    # b_l1 = np.ones(1)
    A_lb = [np.eye(m), -np.eye(m)]
    b_lb = [np.zeros(m), -np.ones(m)]
    A_lb, b_lb = np.row_stack(A_lb), np.concatenate(b_lb)

    mu = np.sqrt(1 - f.dot(rinv) ** 2 / norm(f) ** 2)
    if mu > 0.01:
        a = f - f.dot(rinv) * rinv  # anchor_lagrange
        A_eq, b_eq = None, None  # A_l1, b_l1
    else:
        # print(f'f={f}; descending; mu={mu}')
        a = f
        A_strict_des = np.eye(m) - rinv[:, None] @ rinv[None, :]
        A_strict_des = A_strict_des[:-1, :]
        b_strict_des = np.zeros(m - 1)
        A_eq, b_eq = A_strict_des, b_strict_des,  # np.row_stack([A_strict_des, A_l1]), np.concatenate([b_strict_des, b_l1])
    HQc = GG @ GG    # Hessian of quadratic cost
    gQc = -a @ GG      # gradient of quadratic cost

    alpha = solve_qp(HQc, gQc, -A_lb, -b_lb, A_eq, b_eq)
    #if alpha.sum() > 1: # original
    alpha /= alpha.sum() # scale
    return alpha


def solve_qp(P, q, A_ub=None, b_ub=None, A_eq=None, b_eq=None):
    """
        Minimize      0.5 * x @ P @ x + q @ x
        Subject to    A_ub @ x <= b_ub
        and           A_eq @ x = b_eq
    """
    # print(f'P=\n{P}\nq={q}\nA_ub=\n{A_ub}\nb_ub={b_ub}\nA_eq=\n{A_eq}\nb_ub={b_eq}')

    m = len(q)
    P, q = matrix(P), matrix(q)

    if A_ub is not None:
        A_ub, b_ub = matrix(A_ub), matrix(b_ub)

    if A_eq is not None:
        A_eq, b_eq = matrix(A_eq), matrix(b_eq)

    try:
        sol = solvers.qp(P, q, A_ub, b_ub, A_eq, b_eq)
        status = sol['status']
        if status == 'optimal':
            return np.array(sol['x']).ravel()  # , sol['primal objective']
        else:
            print(f'****** QP not optimal: status = {status} ********')
            return np.zeros(m)  # , 0
    except Exception as e:
        print(e)
        return np.zeros(m)  #, None
