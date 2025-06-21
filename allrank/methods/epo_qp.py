import numpy as np
from numpy.linalg import norm
from cvxopt import matrix, solvers
import cvxpy as cp
# solvers.options["show_progress"] = True
solvers.options['maxiters'] = 200


def epo_qp_get_alpha(r, f, GG):
    assert len(r) == len(f)
    m = len(f)
    rinv = 1.0 / r
    rinv = rinv / norm(rinv)
    
    mu = np.sqrt(1 - f.dot(rinv) ** 2 / norm(f) ** 2)
    if mu > 0.01:
        print(f'balancing mode, mu={mu}')
        a = f - f.dot(rinv) * rinv  # anchor_lagrange
    else:
        print(f'descending; mu={mu}')
        a = f
    # print(f'r={r}, rinv={rinv}, f={f}, GG={GG}, {GG.shape}, a={a}')   
    GG_norm = np.linalg.norm(GG, ord=2)
    GG /= GG_norm
    a /= GG_norm
    
    alpha = solve_qp_cvxpy(GG, a)

    if np.abs(alpha).sum() == 0.0 or np.isnan(np.abs(alpha).sum()):
        alpha = r
        # # follow WC
        # rf = r * f

        # # Create a one-hot encoded alpha: 1 at the max position, 0 elsewhere
        # alpha = np.zeros_like(rf)
        # alpha[np.argmax(rf)] = 1  # Set the max position to 1
    alpha = np.divide(alpha, alpha.sum(), dtype=np.float64)
    print(f'alpha = {alpha}')
    return alpha

def solve_qp_cvxpy(GG, a):
    """
    Solves the quadratic programming problem:
        min || GG * alpha - a ||_2^2
        s.t. sum(alpha) = 1, alpha >= 0

    Parameters:
        GG (numpy.ndarray): A matrix used in the quadratic term.
        a (numpy.ndarray): A vector used in the linear term.

    Returns:
        numpy.ndarray: Optimal values of alpha.
    """

    # Number of variables
    K = GG.shape[1]

    # Define optimization variable
    alpha = cp.Variable(K, nonneg=True)

    # Define the objective function
    objective = cp.Minimize(cp.norm(GG @ alpha - a, 2) ** 2)

    # Define constraints
    constraints = [cp.sum(alpha) == 1]

    # Form and solve the problem
    problem = cp.Problem(objective, constraints)
    # problem.solve()
    problem.solve(solver=cp.SCS, verbose=False) # CVXOPT, SCS, XPRESS, MOSEK

    # Return the optimal alpha values
    return alpha.value



# import numpy as np
# from numpy.linalg import norm
# from cvxopt import matrix, solvers
# solvers.options["show_progress"] = False
# solvers.options['maxiters'] = 200


# def epo_qp_get_alpha(r, f, GG):
#     assert len(r) == len(f)
#     m = len(f)
#     rinv = 1.0 / r
#     rinv = rinv / norm(rinv)

#     # A_l1 = np.ones((1, m))
#     # b_l1 = np.ones(1)
#     A_lb = [np.eye(m), -np.eye(m)]
#     b_lb = [np.zeros(m), -np.ones(m)]
#     # b_lb = [np.ones(m), -np.ones(m)]
#     A_lb, b_lb = np.row_stack(A_lb), np.concatenate(b_lb)
#     mu = np.sqrt(1 - f.dot(rinv) ** 2 / norm(f) ** 2)
#     if mu > 0.01:
#         a = f - f.dot(rinv) * rinv  # anchor_lagrange
#         A_eq, b_eq = None, None  # A_l1, b_l1
#     else:
#         # print(f'f={f}; descending; mu={mu}')
#         a = f
#         A_strict_des = np.eye(m) - rinv[:, None] @ rinv[None, :]
#         A_strict_des = A_strict_des[:-1, :]
#         b_strict_des = np.zeros(m - 1)
#         A_eq, b_eq = A_strict_des, b_strict_des,  # np.row_stack([A_strict_des, A_l1]), np.concatenate([b_strict_des, b_l1])
#     HQc = GG @ GG    # Hessian of quadratic cost
#     gQc = -a @ GG      # gradient of quadratic cost

#     alpha = solve_qp(HQc, gQc, -A_lb, -b_lb, A_eq, b_eq)
#     if np.abs(alpha).sum() == 0.0 or np.isnan(np.abs(alpha).sum()):
#         # follow LS
#         # alpha = r
        
#         # follow WC
#         rf = r * f

#         # Create a one-hot encoded alpha: 1 at the max position, 0 elsewhere
#         alpha = np.zeros_like(rf)
#         alpha[np.argmax(rf)] = 1  # Set the max position to 1
        
#     alpha = np.divide(alpha, alpha.sum(), dtype=np.float64)

#     return alpha


# def solve_qp(P, q, A_ub=None, b_ub=None, A_eq=None, b_eq=None):
#     """
#         Minimize      0.5 * x @ P @ x + q @ x
#         Subject to    A_ub @ x <= b_ub
#         and           A_eq @ x = b_eq
#     """
#     # print(f'P=\n{P}\nq={q}\nA_ub=\n{A_ub}\nb_ub={b_ub}\nA_eq=\n{A_eq}\nb_ub={b_eq}')

#     m = len(q)
#     P, q = matrix(P), matrix(q)

#     if A_ub is not None:
#         A_ub, b_ub = matrix(A_ub), matrix(b_ub)

#     if A_eq is not None:
#         A_eq, b_eq = matrix(A_eq), matrix(b_eq)

#     try:
#         sol = solvers.qp(P, q, A_ub, b_ub, A_eq, b_eq)
#         status = sol['status']
#         if status == 'optimal':
#             return np.array(sol['x']).ravel()  # , sol['primal objective']
#         else:
#             print(f'****** QP not optimal: status = {status} ********')
#             return np.zeros(m)  # , 0
#     except Exception as e:
#         print(e)
#         return np.zeros(m)  #, None
