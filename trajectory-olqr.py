import numpy as np
import scipy.linalg as la
from scipy.optimize import fsolve
import control as ctr
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


def partial_tilde(p, q):
    # returns the coordinate tangent vectors under the natural identification
    temp = np.zeros((B.shape[1], C.shape[0]))
    temp[p, q] = 1
    return temp @ C


def diff_Y(K, p, q):
    # differential of Y_K acting on partial_pq
    return lyap(A_cl(K), B @ partial(p, q) @ Y(K) @ A_cl(K).T + A_cl(K) @ Y(K) @ partial(p, q).T @ B.T)


def grad_f(K):
    # grad f at K
    return R @ K + B.T @ P(K) @ A_cl(K)


def S_K_E(K, E):
    return lyap(A_cl(K).T, E.T @ grad_f(K) + grad_f(K).T @ E)


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


def S_K_tensor(K):
    # outputs a tensor S_K_out where S_K_out[p][q] = S_K_E(K,partial_pq)
    S_K_out = {}
    for p in range(m):
        for q in range(d):
            S_K_out[p, q] = S_K_E(K, partial_tilde(p, q))
    return S_K_out


def cov_hess_h(K, con):
    # outputs the coefficients of the covariant derivative of f cov_hess_out where
    # cov_hess_out[k,l,p,q] = <Hess_h [partial_tilde_kl], partial_tilde_pq>
    Y_K = Y(K)
    P_K = P(K)
    grad_h_K= tan_proj(K, grad_f(K))
    Gamma = christoffel(K, con)
    S_K = S_K_tensor(K)

    cov_hess_out = np.zeros((m, d, m, d))
    for k1 in range(m):
        for l1 in range(d):
            for p1 in range(m):
                for q1 in range(d):

                    nabla_UV = np.zeros((B.shape[1],B.shape[0]))
                    for k in range(m):
                        for l in range(n):
                            for p in range(m):
                                for q in range(n):
                                    for i in range(m):
                                        for j in range(n):
                                            nabla_UV = nabla_UV + partial_tilde(k1, l1)[k, l] * partial_tilde(p1, q1)[p, q] * Gamma[
                                                i, j, k, l, p, q] * partial(i, j)

                    cov_hess_out[k1, l1, p1, q1] = metric(partial_tilde(k1, l1), B.T @ S_K[p1, q1] @ A_cl(K),
                                                          Y_K) + metric(partial_tilde(p1, q1), (
                                B.T @ S_K[k1, l1] @ A_cl(K) + (R + B.T @ P_K @ B) @ partial_tilde(k1, l1)), Y_K) - metric(
                        grad_h_K, nabla_UV, Y_K)

    return cov_hess_out


def coordinate2matrix(input_matrix):
    output_matrix = np.zeros((m * d, m * d))
    for p in range(m):
        for q in range(d):
            output_matrix[p * d + q, :] = np.reshape(input_matrix[p, q, :, :], -1)
    return output_matrix


def euc_proj(E):
    # outputs the Euclidean projection of E on a linear subspace kappa
    return la.solve((C @ C.T).T, (E @ C.T).T).T @ C


def tan_proj(K, E):
    # outputs the tangential projection of E on to the tangent space of the submanifold at K
    Y_K = Y(K)
    return la.solve((C @ Y_K @ C.T).T, (E @ Y_K @ C.T).T).T @ C


def equations_newton_direction(inputs, *data):
    L, con = data
    K = L @ C
    L_out = inputs.reshape(L.shape)
    # computing a matrix of hessian consisting
    cov_hessian = cov_hess_h(K, con)
    grad_h_K = tan_proj(K, grad_f(K))
    Y_K = Y(K)

    equation = np.zeros_like(L)
    for p in range(m):
        for q in range(d):
                equation[p, q] = metric(grad_h_K, partial_tilde(p, q), Y_K)
                for k in range(m):
                    for l in range(d):
                        equation[p, q] = equation[p, q] + L_out[k, l] * cov_hessian[k, l, p, q]
    return equation.ravel()


def newton_direction(L, con):
    K = L @ C
    output = fsolve(equations_newton_direction, (tan_proj(K, grad_f(K)) @ la.pinv(C)).ravel(), args=(L, con))
    return output.reshape(L.shape)


def lqr_cost(K):
    if np.max(np.abs(la.eigvals(A_cl(K)))) > 1:
        print('The controller is not stabilizing')
        cost = np.infty
        # cost = float('nan')
    else:
        cost = 0.5 * np.trace(P(K) @ sigma)
    return cost


def rnorm(K, E):
    return np.trace(E.T @ E @ Y(K))


def metric(E, F, Y):
    return np.trace(E.T @ F @ Y)


def stepsize_bound(K, G):
    return np.min(np.abs(la.eigvals(K.T @ R @ K + Q))) * np.min(np.abs(la.eigvals(sigma))) / (2 * np.max(np.abs(la.eigvals(P(K)))) * la.norm(B @ G))


def my_opt_solver(T, L0):

    K0 = L0 @ C

    # ### Learning the optimal structured LQR controller via our algorithm with Riemannian Connection ##############
    # choose the stepsize rule to be 'constant' or 'best'
    stepsize_rule = 'best'

    L_r = [L0]
    K_r = [K0]
    nd_r = []
    const_r = []
    eta_r = []
    for _ in range(T):
        if np.max(np.abs(la.eigvals(A_cl(K_r[-1])))) > 1:
            print('Unstable controller in our Alternative algorithm (Riemannian con.) at iteration = ', _)

        # computing newton direction
        nd_r.append(newton_direction(L_r[-1], 'Riemannian'))

        # computing stepsize

        const_r.append(stepsize_bound(K_r[-1], nd_r[-1]))
        if stepsize_rule == 'best':
            eta_r.append(np.min([1, const_r[-1]]))
        elif stepsize_rule == 'constant':
            eta_r.append(const_r[0])

        L_r.append(L_r[-1] + eta_r[-1] * nd_r[-1])
        K_r.append(L_r[-1] @ C)

    if np.min(la.eigvals(coordinate2matrix(cov_hess_h(K_r[-1], 'Riemannian')))) < -1.0e-10:
        print('The Riemmanian Hessian is not positive definite')
    if la.norm(tan_proj(K_r[-1], grad_f(K_r[-1]))) > 1.0e-5:
        print('The grad h is not zero yet (ours - Riemannian connection)...')

    # ######### Learning the optimal structured LQR controller via our algorithm with Euclidean Connection ############
    stepsize_rule = 'best'
    L_euc = [L0]
    K_euc = [K0]
    nd_euc = []
    const_euc = []
    eta_euc = []
    for _ in range(T):
        if np.max(np.abs(la.eigvals(A_cl(K_euc[-1])))) > 1:
            print('Unstable controller in our algorithm (Euclidean con.) at iteration = ', _)

        # computing newton direction
        nd_euc.append(newton_direction(L_euc[-1], 'Euclidean'))

        # computing stepsize
        const_euc.append(stepsize_bound(K_euc[-1], nd_euc[-1]))
        if stepsize_rule == 'best':
            eta_euc.append(np.min([1, const_euc[-1]]))
        elif stepsize_rule == 'constant':
            eta_euc.append(const_euc[0])

        L_euc.append(L_euc[-1] + eta_euc[-1] * nd_euc[-1])
        K_euc.append(L_euc[-1] @ C)

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
    # eta_pg = 0.1 / lqr_cost(K_pg[0])
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
    if d == n:
        K_opt = K_lqr
    else:
        last_iterates = [K_r[-1], K_euc[-1], K_pg[-1]]
        num_optimal = list(map(lqr_cost, last_iterates))
        K_opt = last_iterates[num_optimal.index(min(num_optimal))]

    T_plot = T
    # ########## the plot for costs
    plt.figure()
    plt.plot(list(range(0, T_plot + 1)),
             list(map(lambda kk: (lqr_cost(kk) - lqr_cost(K_opt)) / (lqr_cost(K_r[0]) - lqr_cost(K_opt)),
                      K_r[0: T_plot + 1])))
    plt.plot(list(range(0, T_plot + 1)),
             list(map(lambda kk: (lqr_cost(kk) - lqr_cost(K_opt)) / (lqr_cost(K_euc[0]) - lqr_cost(K_opt)),
                      K_euc[0: T_plot + 1])))
    plt.plot(list(range(0, T_plot + 1)),
             list(map(lambda kk: (lqr_cost(kk) - lqr_cost(K_opt)) / (lqr_cost(K_pg[0]) - lqr_cost(K_opt)),
                      K_pg[0: T_plot + 1])))

    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    plt.yscale("log")
    plt.xlabel(r'$\mathrm{iteration}$', size=15)
    plt.ylabel(r'$\frac{f(K) - f(K^*)}{f(K_0) - f(K^*)}$', fontsize=20)
    plt.legend(mylegend)
    plt.grid()
    plt.show()
    plt.subplots_adjust(left=0.185)

    # ########## the plot for control iterates
    plt.figure()

    plt.plot(list(range(0, T_plot + 1)),
             list(map(lambda kk: la.norm(kk - K_opt) / la.norm(K_r[0] - K_opt), K_r[0:T_plot + 1])))
    plt.plot(list(range(0, T_plot + 1)),
             list(map(lambda kk: la.norm(kk - K_opt) / la.norm(K_euc[0] - K_opt), K_euc[0:T_plot + 1])))
    plt.plot(list(range(0, T_plot + 1)),
             list(map(lambda kk: la.norm(kk - K_opt) / la.norm(K_pg[0] - K_opt), K_pg[0:T_plot + 1])))

    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    plt.yscale("log")
    plt.xlabel(r'$\mathrm{iteration}$', size=15)
    plt.ylabel(r'$\frac{\|K_t - K^*\|}{\|K_0 - K^*\|}$', fontsize=20)
    plt.legend(mylegend)
    plt.grid()
    plt.show()
    plt.subplots_adjust(left=0.185)

    return K_r, K_euc, K_pg, K_opt, [eta_r.index(1), eta_euc.index(1)]


if __name__ == '__main__':
    # problem parameters
    A = np.array([[0.8, 1.], [0., 0.9]])
    B = np.array([[0., 1], [1., 0]])
    C = np.array([[1., 1.]])
    n, m = B.shape
    d, _ = C.shape
    Q = 1 * np.array([[10, 0],
                      [0, 0.5]])
    R = 0.1 * np.eye(m)
    sigma = 1 * np.array([[1, 0],
                          [0, 5]])

    # controllability/observability check
    if np.linalg.matrix_rank(ctr.ctrb(A, B)) < n:
        print('The (A,B) pair is not controllable')
    if np.linalg.matrix_rank(ctr.ctrb(A.T, C.T)) < n:
        print('The (A,C) pair is not observable')

    # optimal unstructured state-feedback discrete-time infinite-horizon LQR controller
    P_lqr = la.solve_discrete_are(A, B, Q, R)
    K_lqr = -la.inv(R + B.T @ P_lqr @ B) @ B.T @ P_lqr @ A

    mylegend = [r'$\mathrm{QRNPO~(\mathrm{Hess})}$',
                r'$\mathrm{QRNPO~(\overline{\mathrm{Hess}})}$',
                r'$\mathrm{PGD~}$']

    L_0 = [np.array([[-1],[-1.3]]), np.array([[0.1],[-1.3]]), np.array([[-0.4],[-0.4]]), np.array([[-3.15],[-0.45]]), np.array([[-0.1],[-1.73]]), np.array([[-0.33],[0.22]])]
    T = [80, 80, 80, 1200, 500, 1500]
    # #### level-curve plot lqr cost
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    # ############# Inner Plot
    ax2 = plt.axes([0, 0, 1, 1])
    ip = InsetPosition(ax1, [0.15, 0.1, 0.3, 0.3])
    ax2.set_axes_locator(ip)
    mark_inset(ax1, ax2, loc1=1, loc2=2, fc="none", ec='black')
    ax2.set(ylim=([-1., -0.9]), xlim=([-0.7, -0.6]))

    for i in range(len(L_0)):
        K_r, K_euc, K_pg, K_opt, stepone = my_opt_solver(T[i],L_0[i])
        T_plot = T[i]
        ax1.plot(list(K_r[i][0][0] for i in range(T_plot+1)), list(K_r[i][1][0] for i in range(T_plot+1)), color='tab:blue')
        ax1.plot(list(K_euc[i][0][0] for i in range(T_plot+1)), list(K_euc[i][1][0] for i in range(T_plot+1)), color='tab:orange')
        ax1.plot(list(K_pg[i][0][0] for i in range(T_plot+1)), list(K_pg[i][1][0] for i in range(T_plot+1)), color='tab:green')
        ax1.plot(K_r[stepone[0]][0], K_r[stepone[0]][1], marker='s', color='tab:blue', markersize=4)
        ax1.plot(K_euc[stepone[1]][0], K_euc[stepone[1]][1], marker='s', color='tab:orange', markersize=4)

        ax2.plot(list(K_r[i][0][0] for i in range(T_plot + 1)), list(K_r[i][1][0] for i in range(T_plot + 1)),
                 color='tab:blue')
        ax2.plot(list(K_euc[i][0][0] for i in range(T_plot + 1)), list(K_euc[i][1][0] for i in range(T_plot + 1)),
                 color='tab:orange')
        ax2.plot(list(K_pg[i][0][0] for i in range(T_plot + 1)), list(K_pg[i][1][0] for i in range(T_plot + 1)),
                 color='tab:green')
        ax2.plot(K_r[stepone[0]][0], K_r[stepone[0]][1], marker='s', color='tab:blue', markersize=4)
        ax2.plot(K_euc[stepone[1]][0], K_euc[stepone[1]][1], marker='s', color='tab:orange', markersize=4)

    ax1.plot(K_opt[0][0], K_opt[1][0], marker='*', color='red')
    ax2.plot(K_opt[0][0], K_opt[1][0], marker='*', color='red')

    x = np.arange(-2.6, 1, 0.005)
    y = np.arange(-1.4, 1.6, 0.005)
    X_, Y_ = np.meshgrid(x, y)
    zs = np.array(list(map(lambda i, j: lqr_cost(K_opt + np.array([[i], [j]]) @ C) if np.max(
        np.abs(la.eigvals(A_cl(K_opt + np.array([[i], [j]]) @ C)))) < 1 else np.infty, np.ravel(X_), np.ravel(Y_))))
    Z_ = zs.reshape(X_.shape)

    cs = ax1.contour(K_opt[0][0] + X_, K_opt[1][0] + Y_, Z_, levels=lqr_cost(K_opt) * np.logspace(0.001, 3),
                     locator=ticker.LogLocator(), cmap=cm.Purples)
    fig.colorbar(cs, ax=[ax1], ticks=[10 ** n for n in range(-2, 5)], orientation="vertical")
    ax1.set_xlabel('$l_1$')
    ax1.set_ylabel('$l_2$')
    ax1.legend(mylegend, loc='upper left')
    plt.show()
    fig.savefig('images/landscape_olqr.pdf', format='pdf')
