# Code to generate Fig. 1
import numpy as np
import scipy.linalg as la
from scipy.optimize import fsolve
import control as ctr
import pickle

import matplotlib.pyplot as plt
from matplotlib import ticker, cm
from mpl_toolkits.axes_grid1.inset_locator import (InsetPosition, mark_inset)


plt.rcParams.update(plt.rcParamsDefault)
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['font.size'] = 15


def lyap(input_A, input_Q):
    # This lyap() function solves lyapunov equation as defined in our paper, i.e. AYA^T + Q = Y ;
    # so be careful with the version of scipy
    return la.solve_discrete_lyapunov(input_A, input_Q)


def A_cl(K):
    # close-loop system
    return A + B @ K


def P(K):
    # cost matrix
    return lyap(A_cl(K).T, K.T @ R @ K + Q)


def Y(K):
    # the matrix that induces the metric
    return lyap(A_cl(K), sigma)


def partial(p, q):
    # returns the coordinate tangent vectors under the natural identification
    temp = np.zeros((B.shape[1],B.shape[0]))
    temp[p, q] = 1
    return temp


def diff_Y(K, p, q):
    # differential of Y_K acting on partial_pq
    return lyap(A_cl(K), B @ partial(p, q) @ Y(K) @ A_cl(K).T + A_cl(K) @ Y(K) @ partial(p, q).T @ B.T)


def grad_f(K):
    # grad f at K
    return R @ K + B.T @ P(K) @ A_cl(K)


def S_K_E(K, E):
    return lyap(A_cl(K).T, E.T @ grad_f(K) + grad_f(K).T @ E)


def S_K_tensor(K):
    # outputs a tensor S_K_out where S_K_out[p][q] = S_K_E(K,partial_pq)
    S_K_out = {}
    for p in range(m):
        for q in range(n):
            S_K_out[p, q] = S_K_E(K, partial(p, q))
    return S_K_out


def christoffel(K, connection):
    # outputs the Christoffel symbol of [i, j, k, l, p, q] at K  as christ_out[i, j, k, l, p, q]
    inv_Y = la.inv(Y(K))
    diff_Y_pq = {}
    mult_diff_Y_inv = {}
    for p in range(m):
        for q in range(n):
            diff_Y_pq[p, q] = diff_Y(K, p, q)
            mult_diff_Y_inv[p, q] = diff_Y_pq[p, q] @ inv_Y

    if connection == 'Riemannian':
        christ_out = np.zeros((m, n, m, n, m, n))
        for i in range(m):
            for j in range(n):
                for k in range(m):
                    for l in range(n):
                        for p in range(m):
                            for q in range(n):
                                if i != p and i != k and k != p:
                                    christ_out[i, j, k, l, p, q] = 0
                                elif i != p and i == k:
                                    christ_out[i, j, k, l, p, q] = 0.5 * mult_diff_Y_inv[p, q][l, j]
                                elif i != k and i == p:
                                    christ_out[i, j, k, l, p, q] = 0.5 * mult_diff_Y_inv[k, l][q, j]
                                elif i != k and p == k:
                                    christ_out[i, j, k, l, p, q] = - 0.5 * np.sum([diff_Y_pq[i, s][q, l] * inv_Y[s, j]
                                                                                   for s in range(n)])
                                elif i == k and i == p:
                                    christ_out[i, j, k, l, p, q] = 0.5 * np.sum(
                                        [(diff_Y_pq[i, l][q, s] + diff_Y_pq[i, q][l, s]
                                          - diff_Y_pq[i, s][q, l]) * inv_Y[s, j]
                                         for s in range(n)])
    elif connection == 'Euclidean':
        christ_out = np.zeros((m, n, m, n, m, n))
    return christ_out


def cov_hess_h(K, con):
    # outputs the coefficients of the covariant derivative of f cov_hess_out where
    # cov_hess_out[k,l,p,q] = <Hess_f [partial_kl], partial_pq>

    Y_K = Y(K)
    P_K = P(K)
    proj_grad_f_K = tan_proj(K,grad_f(K))
    Gamma = christoffel(K, con)
    S_K = S_K_tensor(K)

    cov_hess_out = np.zeros((m, n, m, n))
    for k in range(m):
        for l in range(n):
            for p in range(m):
                for q in range(n):
                    if Kappa[k, l] == 1 and Kappa[p, q] == 1:
                        gamma_matrix_ij = np.array([[Gamma[i, j, k, l, p, q] for j in range(n)] for i in range(m)])
                        cov_hess_out[k, l, p, q] = metric(partial(k, l), B.T @ S_K[p, q] @ A_cl(K), Y_K) \
                                                   + metric(partial(p, q), (B.T @ S_K[k, l] @ A_cl(K) +
                                                                                 (R + B.T @ P_K @ B) @ partial(k, l)), Y_K) \
                                                   - metric(proj_grad_f_K, gamma_matrix_ij, Y_K)
    return cov_hess_out


def coordinate2matrix(input_matrix):
    output_matrix = np.zeros((m * n, m * n))
    for p in range(m):
        for q in range(n):
            output_matrix[p * n + q, :] = np.reshape(input_matrix[p, q, :, :], -1)
    return output_matrix


def euc_proj(E):
    # outputs the Euclidean projection of E on a linear subspace kappa
    return E * Kappa


def proj_equations(inputs, *data):
    K, E, indices = data
    E_projected = insert_sparsity(inputs)
    eq_out = euc_proj((E - E_projected) @ Y(K))
    return eq_out.ravel()[indices]


def tan_proj(K, E):
    # outputs the tangential projection of E on to the tangent space of the submanifold at K
    indices = [i for i, x in enumerate(Kappa.ravel() == 1) if x]
    output = fsolve(proj_equations, euc_proj(E).ravel()[indices], args=(K, E, indices))
    return insert_sparsity(output)


def proj_partial(K):
    # returns proj_partial_tensor, where proj_partial_tensor[p][q] is the tangential projection of partial_pq
    proj_partial_tensor = {}
    for p in range(m):
        for q in range(n):
            proj_partial_tensor[p, q] = tan_proj(K, partial(p, q))
    return proj_partial_tensor


def insert_sparsity(parameters):
    E_out = np.zeros(Kappa.size)
    counter = 0
    for i in range(Kappa.size):
        if Kappa.ravel()[i] != 0:
            E_out[i] = parameters[counter]
            counter += 1
    return E_out.reshape(Kappa.shape)


def equations_newton_direction(inputs, *data):
    K, indices, con = data

    newton_dir_out = insert_sparsity(inputs)

    # computing a matrix of hessian consisting
    cov_hessian = cov_hess_h(K, con)
    grad_h_K = tan_proj(K, grad_f(K))
    Y_K = Y(K)

    equation = np.zeros_like(K)
    for i in range(m):
        for j in range(n):
            if Kappa[i, j] == 1:
                equation[i, j] = metric(grad_h_K, partial(i, j), Y_K)
                for p in range(m):
                    for q in range(n):
                        equation[i, j] = equation[i, j] + newton_dir_out[p, q] * cov_hessian[p, q, i, j]
    return equation.ravel()[indices]


def newton_direction(K, con):
    indices = [i for i, x in enumerate(Kappa.ravel() == 1) if x]
    output = fsolve(equations_newton_direction, tan_proj(K, grad_f(K)).ravel()[indices],
                    args=(K, indices, con))
    return insert_sparsity(output)


def heuristic_equations_direction(inputs, *data):
    K, indices, con = data

    dir_out = insert_sparsity(inputs)
    grad_h_K = tan_proj(K, grad_f(K))
    if con =='Riemannian':
        equation = tan_proj(K, (R + B.T @ P(K) @ B) @ dir_out) + grad_h_K
    elif con =='Euclidean':
        equation = euc_proj((R + B.T @ P(K) @ B) @ dir_out) + grad_h_K
    return equation.ravel()[indices]


def heuristic_direction(K, con):
    indices = [i for i, x in enumerate(Kappa.ravel() == 1) if x]
    output = fsolve(heuristic_equations_direction, tan_proj(K, grad_f(K)).ravel()[indices], args=(K, indices, con))
    return insert_sparsity(output)


def lqr_cost(K):
    if np.max(np.abs(la.eigvals(A_cl(K)))) > 1:
        print('The controller is not stabilizing')
        cost = np.infty
    else:
        cost = 0.5 * np.trace(P(K) @ sigma)
    return cost


def rnorm(K, E):
    return np.trace(E.T @ E @ Y(K))


def metric(E, F, Y):
    return np.trace(E.T @ F @ Y)


def stepsize_bound(K, G):
    return np.min(np.abs(la.eigvals(K.T @ R @ K + Q))) * np.min(np.abs(la.eigvals(sigma))) / (2 * np.max(np.abs(la.eigvals(P(K)))) * la.norm(B @ G))


def my_opt_solver(T, K0):



    # ### Learning the optimal structured LQR controller via our algorithm with Riemannian Connection ##############
    # choose the stepsize rule to be 'constant' or 'best'
    stepsize_rule = 'best'

    K_r = [K0]
    nd_r = []
    const_r = []
    eta_r = []
    for _ in range(T):
        if np.max(np.abs(la.eigvals(A_cl(K_r[-1])))) > 1:
            print('Unstable controller in our Alternative algorithm (Riemannian con.) at iteration = ', _)

        # computing newton direction
        nd_r.append(newton_direction(K_r[-1], 'Riemannian'))

        # computing stepsize

        const_r.append(stepsize_bound(K_r[-1], nd_r[-1]))
        if stepsize_rule == 'best':
            eta_r.append(np.min([1, const_r[-1]]))
        elif stepsize_rule == 'constant':
            eta_r.append(const_r[0])

        K_new_r = K_r[-1] + eta_r[-1] * nd_r[-1]
        K_r.append(K_new_r)

    if np.min(la.eigvals(coordinate2matrix(cov_hess_h(K_r[-1], 'Riemannian')))) < -1.0e-10:
        print('The Riemmanian Hessian is not positive definite')
    if la.norm(tan_proj(K_r[-1], grad_f(K_r[-1]))) > 1.0e-5:
        print('The grad h is not zero yet (ours - Riemannian connection)...')

    # ######### Learning the optimal structured LQR controller via our algorithm with Euclidean Connection ############
    stepsize_rule = 'best'
    K_euc = [K0]
    nd_euc = []
    const_euc = []
    eta_euc = []
    for _ in range(T):
        if np.max(np.abs(la.eigvals(A_cl(K_euc[-1])))) > 1:
            print('Unstable controller in our algorithm (Euclidean con.) at iteration = ', _)

        # computing newton direction
        nd_euc.append(newton_direction(K_euc[-1], 'Euclidean'))

        # computing stepsize
        const_euc.append(stepsize_bound(K_euc[-1], nd_euc[-1]))
        if stepsize_rule == 'best':
            eta_euc.append(np.min([1, const_euc[-1]]))
        elif stepsize_rule == 'constant':
            eta_euc.append(const_euc[0])

        K_new_euc = K_euc[-1] + eta_euc[-1] * nd_euc[-1]
        K_euc.append(K_new_euc)

    if np.min(la.eigvals(coordinate2matrix(cov_hess_h(K_euc[-1], 'Euclidean')))) < -1.0e-10:
        print('The Euclidean Hessian is not positive definite')
    if la.norm(tan_proj(K_euc[-1], grad_f(K_euc[-1]))) > 1.0e-5:
        print('The grad h is not zero yet (ours - Euclidean con.)...')

    # ################## Learning the optimal structured LQR controller via Euclidean projected gradient ##############
    stepsize_rule = 'constant'
    K_pg = [K0]
    pg_direction = []
    const_pg = []
    eta_pg = []
    T_pg = T
    for _ in range(T_pg):
        if np.max(np.abs(la.eigvals(A_cl(K_pg[-1])))) > 1:
            print('Unstable controller in projected-gradient (Euclidean) algorithm at iteration = ', _)

        gradient_f = grad_f(K_pg[-1]) @ Y(K_pg[-1])
        pg_direction.append(euc_proj(gradient_f))

        # computing stepsize
        const_pg.append(stepsize_bound(K_pg[-1], pg_direction[-1]))

        if stepsize_rule == 'best':
            eta_pg.append(np.min([0.5 / lqr_cost(K_pg[0]), const_pg[-1]]))
        elif stepsize_rule == 'constant':
            eta_pg.append(const_pg[0])

        K_new_pg = K_pg[-1] - eta_pg[-1] * pg_direction[-1]
        K_pg.append(K_new_pg)

    # ##########################################  plotting the results #############################################
    # convergence plot assuming the best optimal computed numerically
    if (Kappa.ravel() == 1).all():
        K_opt = K_lqr
    else:
        last_iterates = [K_r[-1], K_euc[-1],  K_pg[-1]]
        num_optimal = list(map(lqr_cost, last_iterates))
        K_opt = last_iterates[num_optimal.index(min(num_optimal))]


    try:
        stepsize_one_r = eta_r.index(1)
    except:
        stepsize_one_r = np.nan

    try:
        stepsize_one_euc = eta_euc.index(1)
    except:
        stepsize_one_euc = np.nan

    return K_r, K_euc, K_pg, K_opt, [stepsize_one_r, stepsize_one_euc]


if __name__ == '__main__':
    # problem parameters
    A = np.array([[0.8, 1.], [0, 0.9]])
    B = np.array([[0., 1], [1., 0]])
    n, m = B.shape
    Q = 1 * np.array([[10, 0],
                      [0, 0.5]])
    R = 0.1 * np.eye(m)
    sigma = 1 * np.array([[1, 0],
                          [0, 5]])

    Kappa = np.array([[1, 0],
                      [0, 1]])
    indices = [i for i, x in enumerate(Kappa.ravel() == 1) if x]

    # controllability/observability check
    if np.linalg.matrix_rank(ctr.ctrb(A, B)) < n:
        print('The (A,B) pair is not controllable')

    # optimal unstructured state-feedback discrete-time infinite-horizon LQR controller
    P_lqr = la.solve_discrete_are(A, B, Q, R)
    K_lqr = -la.inv(R + B.T @ P_lqr @ B) @ B.T @ P_lqr @ A


    K_0 = [insert_sparsity([-0.05,-1.3])]
    T = [100]

    for i in range(len(K_0)):
        K_r, K_euc, K_pg, K_opt, stepone = my_opt_solver(T[i], K_0[i])
        T_plot = T[i]

    x = np.arange(-1.6, 1.6, 0.005)
    y = np.arange(-2.5, 2.5, 0.005)
    X_, Y_ = np.meshgrid(x, y)


     ################## main computation ######################
     cost = np.array(list(map(lambda i, j: lqr_cost(K_opt + insert_sparsity([i, j])) if np.max(
         np.abs(la.eigvals(A_cl(K_opt + insert_sparsity([i, j]))))) < 1 else np.infty, np.ravel(X_), np.ravel(Y_))))
     Z_ = cost.reshape(X_.shape)
    
     Euc_hess_pos = np.array(list(map(lambda i, j: 1 if np.max(
         np.abs(la.eigvals(A_cl(K_opt + insert_sparsity([i, j]))))) < 1 and np.min(la.eigvals(coordinate2matrix(cov_hess_h(K_opt + insert_sparsity([i, j]), 'Euclidean'))[np.ix_([0,3],[0,3])])) > 0 else 0, np.ravel(X_), np.ravel(Y_))))
     pEuc_ = Euc_hess_pos.reshape(X_.shape)
    
     Riem_hess_pos = np.array(list(map(lambda i, j: 1 if np.max(
         np.abs(la.eigvals(A_cl(K_opt + insert_sparsity([i, j]))))) < 1 and np.min(la.eigvals(
         coordinate2matrix(cov_hess_h(K_opt + insert_sparsity([i, j]), 'Riemannian'))[
             np.ix_([0, 3], [0, 3])])) > 0 else 0, np.ravel(X_), np.ravel(Y_))))
     pRiem_ = Riem_hess_pos.reshape(X_.shape)


    
     nd_max = 10
     stepsize_one_poss = 0 * pEuc_
     for ind_1 in range(X_.shape[0]):
         for ind_2 in range(X_.shape[1]):
             if pEuc_[ind_1, ind_2] == 1:
                 K_new = K_opt + insert_sparsity([X_[ind_1, ind_2], Y_[ind_1, ind_2]])
                 for kk in range(nd_max):
                     nd = newton_direction(K_new, 'Euclidean')
                     if la.norm(nd) < 1e-15:
                         stepsize_one_poss[ind_1, ind_2] = 1
                         break
                     elif np.max(np.abs(la.eigvals(A_cl(K_new + nd)))) < 1:
                         K_new = K_new + nd
                         stepsize_one_poss[ind_1, ind_2] = 1
                     else:
                         stepsize_one_poss[ind_1, ind_2] = 0
                         break
     is_stepsize_one = stepsize_one_poss.reshape(X_.shape)

    # ############################### level-curve plot lqr cost
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    count0 = ax1.contourf(K_opt.ravel()[indices][0] + X_, K_opt.ravel()[indices][1] + Y_, Z_,
                          levels=lqr_cost(K_opt) * np.array([0, 1e+15]),
                          colors='none', hatches=['\\\\'])

    count1 = ax1.contourf(K_opt.ravel()[indices][0] + X_, K_opt.ravel()[indices][1] + Y_, pRiem_, levels=[0.1, 1.1],
                          cmap=cm.winter, alpha=0.5)

    count2 = ax1.contourf(K_opt.ravel()[indices][0] + X_, K_opt.ravel()[indices][1] + Y_, pEuc_, levels=[0.1, 1.1],
                          cmap=cm.Oranges, alpha=0.7)

    count3 = ax1.contour(K_opt.ravel()[indices][0] + X_, K_opt.ravel()[indices][1] + Y_, Z_,
                         levels=lqr_cost(K_opt) * np.logspace(0.01, 10, 25),
                         locator=ticker.LogLocator(), colors=['purple'], alpha=0.4, linestyles='dashed')

    count5 = ax1.plot(K_opt.ravel()[indices][0], K_opt.ravel()[indices][1], marker='*', color='red', linestyle='None')

    h1, _ = count1.legend_elements()
    h2, _ = count2.legend_elements()
    h3, _ = count0.legend_elements()
    h4, _ = count3.legend_elements()
    mylegend = [r'$\widetilde{S}$',
                r'$h = f|_{\widetilde{S}}$',
                r'$\mathrm{Hess}h \succ 0$',
                r'$\overline{\mathrm{Hess}}h \succ 0$',
                r'$\mathrm{minimum}$',
                r'$\eta = 1~(\overline{\mathrm{Hess}}h)$']

    ax1.legend([h3[0], h4[1], h1[0], h2[0],  count5[0]], mylegend, loc='upper right')



    # ############# Inner Plot
    ax2 = plt.axes([0, 0, 1, 1])
    ip = InsetPosition(ax1, [0.02, 0.1, 0.45, 0.3])
    ax2.set_axes_locator(ip)
    mark_inset(ax1, ax2, loc1=1, loc2=2, fc="none", ec='black')
    ax2.set(ylim=([-1.2, -0.8]), xlim=([-0.3, 0.3]))
    ax2.contourf(K_opt.ravel()[indices][0] + X_, K_opt.ravel()[indices][1] + Y_, Z_,
                          levels=lqr_cost(K_opt) * np.array([0, 1e+15]),
                          colors='none', hatches=['\\\\'])
    ax2.contourf(K_opt.ravel()[indices][0] + X_, K_opt.ravel()[indices][1] + Y_, pRiem_, levels=[0.1, 1.1],
                 cmap=cm.winter, alpha=0.5)
    ax2.contourf(K_opt.ravel()[indices][0] + X_, K_opt.ravel()[indices][1] + Y_, pEuc_, levels=[0.1, 1.1],
                 cmap=cm.Oranges, alpha=0.7)
    ax2.contour(K_opt.ravel()[indices][0] + X_, K_opt.ravel()[indices][1] + Y_, Z_,
                levels=lqr_cost(K_opt) * np.logspace(0.01, 10, 25),
                locator=ticker.LogLocator(), colors=['purple'], alpha=0.4, linestyles='dashed')

    ax2.plot(K_opt.ravel()[indices][0], K_opt.ravel()[indices][1], marker='*', color='red')

    ax1.set_xlabel('$l_1$')
    ax1.set_ylabel('$l_2$')
    ax2.set_xticklabels('')
    ax2.set_yticklabels('')
    ax2.set_xticks([])
    ax2.set_yticks([])
    fig.tight_layout(pad=0.1)
    plt.show()
    fig.savefig('images/motivation.pdf', format='pdf')
