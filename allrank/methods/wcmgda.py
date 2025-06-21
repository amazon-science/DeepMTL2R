import numpy as np
import cvxpy as cvx
from scipy.linalg import sqrtm


def wc_mgda_get_alpha(r, wc, G2, old_alpha, old_objv, old_err=None, wb=None, u=0.01, lb=None, normalize_k=True):
    wc_mgda_get_alpha.counter += 1
    m = len(r)

    assert len(wc) == len(G2) == len(r) == m, "length != m"
    #        assert u < 1, "u greater than 1"

    if wb is None:
        wb = np.zeros(len(wc))
    if lb is None:
        lb = np.zeros(len(wc))

    t = len(wc)

    # set the objective and constraints and solve the problem
    alp = cvx.Variable(t)
    bet = cvx.Variable(t)
    ee = np.ones(t)

    #        A = np.diag(_old_alpha) @ (np.eye(d) - ee @ ee.T / d)
    gamma = cvx.Variable(1)

    K = sqrtm(G2 + 1e-12 * np.eye(t))

    if wc_mgda_get_alpha.counter == 1:
        init_K_norm = np.linalg.norm(K, ord=2)
        wc_mgda_get_alpha.init_K_norm = init_K_norm
    else:
        init_K_norm = wc_mgda_get_alpha.init_K_norm

    K /= init_K_norm

    if normalize_k:
        k = np.linalg.norm(K, ord=2)
    else:
        k = 1

    if u > 0:
        # print(f'u={u}, G2={G2}, K={K}, k={k}, wc={wc}, {wb}={wb}, lb={lb}')
        obj = cvx.Maximize(
            #                 cvx.sum(alp * (wc - wb) )
            cvx.sum(alp @ (wc - wb))
            - u * gamma / k
        )

        constr = [
            ee.T @ alp == 1,
            -K @ alp + bet == 0,
            cvx.SOC(gamma, bet),
            alp >= lb
        ]

        prob = cvx.Problem(obj, constr)
        try:
            prob.solve(solver=cvx.SCS, verbose=False) # CVXOPT, SCS, XPRESS, MOSEK
            alpha = np.array(alp.value).squeeze()
            alpha[alpha < lb] = lb[alpha < lb]
            beta = np.array(bet.value).squeeze()
            gamma = np.array(gamma.value).squeeze()
            rho = constr[0].dual_value
            d = constr[1].dual_value
            kd = K @ d
            err = np.linalg.norm(alpha.T @ kd)
            objv = prob.objective.value
        except:
            print(f'failed solving wc-mgda')
            alpha = old_alpha
            objv = old_objv
            err = old_err

    return alpha, objv, err


wc_mgda_get_alpha.counter = 0

def wc_mgda_u_opt(wc, G2, old_objv, old_u, init_u=0.001, wb=None, lb=None, normalize_k=True):
    if wb is None:
        wb = np.zeros(len(wc))
    if lb is None:
        lb = np.zeros(len(wc))

    t = len(wc)
    K = sqrtm(G2 + 1e-12 * np.eye(t))
    if normalize_k:
        k = np.linalg.norm(K, ord=2)
    else:
        k = 1

    # set the objective and constraints and solve the problem
    ee = np.ones(t)

    u = cvx.Variable(1)
    z = cvx.Variable(t)
    rho = cvx.Variable(1)
    d = cvx.Variable(t)

    obj = cvx.Minimize(
        u
    )
    constr = [
        (wc - wb) + z <= rho * ee + K @ d,
        rho - lb.T @ z <= old_objv,
        cvx.SOC(u / k, d),
        z >= 0,
        u >= init_u
    ]

    prob = cvx.Problem(obj, constr)
    success_flag = 2
    try:
        prob.solve(solver=cvx.SCS, verbose=False)
        z = np.array(z.value).squeeze()
        u = u.value[0]
        d = np.array(d.value).squeeze()
        rho = rho.value[0]
        alpha = constr[0].dual_value
        delta = constr[1].dual_value
        if prob.status != 'optimal':
            success_flag = 0
        elif np.abs(delta) < 1e-6 or u < 1e-6:
            success_flag = 1
        else:
            success_flag = 2

        alpha[alpha < 0] = 0

        err = alpha.T @ K @ d

        alpha = alpha / np.linalg.norm(alpha, ord=1)

        #            objv = prob.objective.value
        objv = rho - lb.T @ z

    except:
        objv = old_objv
        u = old_u
        print('no solution')

    return u, objv, alpha, success_flag   # err,