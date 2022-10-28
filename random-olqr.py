# Code to generate Figure 4.b
import numpy as np
import scipy.linalg as la
from scipy.optimize import fsolve
import control as ctr
import matplotlib.pyplot as plt



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
    temp = np.zeros((m, n))
    temp[p, q] = 1
    return temp


def partial_tilde(p, q):
    # returns the coordinate tangent vectors under the natural identification
    temp = np.zeros((m, d))
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

                    nabla_UV = np.zeros((m, n))
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

    equation = np.zeros((m, d))
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
    else:
        cost = 0.5 * np.trace(P(K) @ sigma)
    return cost


def rnorm(K, E):
    return np.trace(E.T @ E @ Y(K))


def metric(E, F, Y):
    return np.trace(E.T @ F @ Y)


def stepsize_bound(K, G):
    return np.min(np.abs(la.eigvals(K.T @ R @ K + Q))) * np.min(np.abs(la.eigvals(sigma))) / (2 * np.max(np.abs(la.eigvals(P(K)))) * la.norm(B @ G))


def random_sys_opt():
    # checking controllability of (A,B) and ensuring open-loop stability of A
    global A
    global B
    global C

    while True:
        A = np.random.rand(n, n)
        B = np.random.rand(n, m)
        C = np.random.rand(d, n)
        if np.linalg.matrix_rank(ctr.ctrb(A, B)) == n and np.linalg.matrix_rank(ctr.ctrb(A.T, C.T)) == n:
            A *= 0.9 / (np.max(np.abs(la.eigvals(A))))
            break
    # Initial structured stabilizing controller as zero due to open-loop stability
    L0 = np.zeros((m, d))

    K0 = L0 @ C

    # optimal unstructured state-feedback discrete-time infinite-horizon LQR controller
    P_lqr = la.solve_discrete_are(A, B, Q, R)
    K_lqr = -la.inv(R + B.T @ P_lqr @ B) @ B.T @ P_lqr @ A

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

        if la.norm(tan_proj(K_r[-1], grad_f(K_r[-1]))) > tol:
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
        else:
            K_r.append(K_r[-1])


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

        if la.norm(tan_proj(K_euc[-1], grad_f(K_euc[-1]))) > tol:
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
        else:
            K_euc.append(K_euc[-1])
            
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

        if _ > np.max([K_r.__len__(), K_euc.__len__()]):
            break


    # ################## Learning the optimal structured LQR controller via Riemannian projected gradient ##############
    stepsize_rule = 'constant'
    K_nat = [K0]
    pg_direction = []
    const_nat = []
    eta_nat = []
    T_nat = T
    for _ in range(T_nat):
        if np.max(np.abs(la.eigvals(A_cl(K_nat[-1])))) > 1:
            print('Unstable controller in projected-gradient (Euclidean) algorithm at iteration = ', _)

        gradient_f = grad_f(K_nat[-1])
        pg_direction.append(tan_proj(K_nat[-1], gradient_f))

        # computing stepsize
        const_nat.append(stepsize_bound(K_nat[-1], pg_direction[-1]))

        if stepsize_rule == 'best':
            eta_nat.append(np.min([0.5 / lqr_cost(K_nat[0]), const_nat[-1]]))
        elif stepsize_rule == 'constant':
            eta_nat.append(const_nat[0])

        K_new_nat = K_nat[-1] - eta_nat[-1] * pg_direction[-1]
        K_nat.append(K_new_nat)

        if _ > np.max([K_r.__len__(), K_euc.__len__()]):
            break

    if d == n:
        K_opt = K_lqr
    else:
        last_iterates = [K_r[-1], K_euc[-1],  K_pg[-1], K_nat[-1]]
        num_optimal = list(map(lqr_cost, last_iterates))
        K_opt = last_iterates[num_optimal.index(min(num_optimal))]

    f_K_r_traj = list(map(lambda kk: (lqr_cost(kk) - lqr_cost(K_opt)) / (lqr_cost(K_r[0]) - lqr_cost(K_opt)), K_r))
    f_K_euc_traj =list(map(lambda kk: (lqr_cost(kk) - lqr_cost(K_opt)) / (lqr_cost(K_euc[0]) - lqr_cost(K_opt)), K_euc))
    f_K_pg_traj =list(map(lambda kk: (lqr_cost(kk) - lqr_cost(K_opt)) / (lqr_cost(K_pg[0]) - lqr_cost(K_opt)), K_pg))
    f_K_nat_traj = list(map(lambda kk: (lqr_cost(kk) - lqr_cost(K_opt)) / (lqr_cost(K_nat[0]) - lqr_cost(K_opt)), K_nat))

    K_r_traj = list(map(lambda kk: la.norm(kk - K_opt) / la.norm(K_r[0] - K_opt), K_r))
    K_euc_traj = list(map(lambda kk: la.norm(kk - K_opt) / la.norm(K_euc[0] - K_opt), K_euc))
    K_pg_traj = list(map(lambda kk: la.norm(kk - K_opt) / la.norm(K_pg[0] - K_opt), K_pg))
    K_nat_traj = list(map(lambda kk: la.norm(kk - K_opt) / la.norm(K_nat[0] - K_opt), K_nat))

    return [[f_K_r_traj, f_K_euc_traj, f_K_pg_traj, f_K_nat_traj], [K_r_traj, K_euc_traj, K_pg_traj, K_nat_traj]]


def plot_traj():
    # ##########################################  plotting the results #############################################
    # convergence plot assuming the best optimal computed numerically
    mylegend = [r'$\mathrm{Alg.1~(\mathrm{Hess})}$',
                r'$\mathrm{Alg.1~(\overline{\mathrm{Hess}})}$',
                r'$\mathrm{PGD~}$',
                r'$\mathrm{NPGD~}$']
    alpha = 0.4
    indices =[]
    conv_iter = []
    # finding the min, max and mean of iterations
    for kk in range(4):
        conv_iter.append(list(map(lambda M: np.nan if M[-1] > tol else next(x for x, val in enumerate(M) if val < tol),
                                  [data_K[iii][kk] for iii in range(data_K.__len__())])))
        indices.append([conv_iter[-1].index(min(conv_iter[-1])), conv_iter[-1].index(max(conv_iter[-1]))])

    # PG is not converged so this is how we find min and max
    conv_iter[2] = [data_K[i][2][-1] for i in range(data_K.__len__())]
    indices[2] = [conv_iter[2].index(np.min(conv_iter[2])), conv_iter[2].index(np.max(conv_iter[2]))]

    conv_iter[3] = [data_K[i][3][-1] for i in range(data_K.__len__())]
    indices[3] = [conv_iter[3].index(np.min(conv_iter[3])), conv_iter[3].index(np.max(conv_iter[3]))]

    # ########## the plot for costs
    fig, axs = plt.subplots(2)

    # shading
    axs[0].fill_between(list(range(T + 1)), data_f[indices[0][0]][0][0:T + 1],
                        data_f[indices[0][1]][0][0:T + 1], color='tab:blue', alpha=alpha)
    axs[0].fill_between(list(range(T + 1)), data_f[indices[1][0]][1][0:T + 1],
                        data_f[indices[1][1]][1][0:T + 1], color='tab:orange', alpha=alpha)
    axs[0].fill_between(list(range(T + 1)), data_f[indices[2][0]][2][0:T + 1],
                        data_f[indices[2][1]][2][0:T + 1], color='tab:green', alpha=alpha)
    axs[0].fill_between(list(range(T + 1)), data_f[indices[3][0]][3][0:T + 1],
                        data_f[indices[3][1]][3][0:T + 1], color='tab:purple', alpha=alpha)
    axs[0].legend(mylegend, loc='lower right')


    # plotting the median among convergence times
    conv_median = []
    for kk in range(2):
        conv_median.append(conv_iter[kk].index(int(np.median(
            [conv_iter[kk][j] for j in [i for i in range(conv_iter[kk].__len__()) if not np.isnan(conv_iter[kk][i])]]))))
    axs[0].plot(data_f[conv_median[0]][0], color='tab:blue')
    axs[0].plot(data_f[conv_median[1]][1], color='tab:orange')
    axs[0].plot(
        np.average(np.concatenate([np.array(data_f[ii][2]).reshape(-1, 1)
                                   for ii in range(data_f.__len__()) if not np.isnan(conv_iter[2][ii])], axis=1), axis=1)
        , color='tab:green')
    axs[0].plot(
        np.average(np.concatenate([np.array(data_f[ii][3]).reshape(-1, 1)
                                   for ii in range(data_f.__len__()) if not np.isnan(conv_iter[3][ii])], axis=1),
                   axis=1)
        , color='tab:purple')

    axs[0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axs[0].set_yscale("log")
    axs[0].set_ylabel(r'$\frac{f(K) - f(K^*)}{f(K_0) - f(K^*)}$', fontsize=25)
    axs[0].set_ylim([1e-12, 10])
    axs[0].set_xticklabels([])
    axs[0].grid()


    # ########## the plot for control iterates
    # shading
    axs[1].fill_between(list(range(T+1)), data_K[indices[0][0]][0][0:T+1],
                     data_K[indices[0][1]][0][0:T+1], color='tab:blue', alpha=alpha)
    axs[1].fill_between(list(range(T+1)), data_K[indices[1][0]][1][0:T+1],
                     data_K[indices[1][1]][1][0:T+1], color='tab:orange', alpha=alpha)
    axs[1].fill_between(list(range(T+1)), data_K[indices[2][0]][2][0:T+1],
                     data_K[indices[2][1]][2][0:T + 1], color='tab:green', alpha=alpha)
    axs[1].fill_between(list(range(T + 1)), data_K[indices[3][0]][3][0:T + 1],
                        data_K[indices[3][1]][3][0:T + 1], color='tab:purple', alpha=alpha)
    axs[1].legend(mylegend, loc='lower right')
    
    # plotting the median among convergence times
    axs[1].plot(data_K[conv_median[0]][0], color='tab:blue')
    axs[1].plot(data_K[conv_median[1]][1], color='tab:orange')
    axs[1].plot(
        np.average(np.concatenate(
            [np.array(data_K[i][2]).reshape(-1, 1) for i in range(data_K.__len__()) if not np.isnan(conv_iter[2][i])],
            axis=1), axis=1), color='tab:green')
    axs[1].plot(
        np.average(np.concatenate(
            [np.array(data_K[i][3]).reshape(-1, 1) for i in range(data_K.__len__()) if not np.isnan(conv_iter[3][i])],
            axis=1), axis=1), color='tab:purple')

    axs[1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axs[1].set_yscale("log")
    axs[1].set_xlabel(r'$\mathrm{iteration}$', size=20)
    axs[1].set_ylabel(r'$\frac{\|K_t - K^*\|}{\|K_0 - K^*\|}$', fontsize=25)
    axs[1].set_ylim([1e-12, 10])
    axs[1].grid()

    fig.tight_layout(pad=0.1)
    fig.subplots_adjust(left=0.185)
    fig.savefig('images/random_olqr_extended.pdf', format='pdf')


if __name__ == '__main__':
    # problem parameters
    n = 6
    m = 3
    d = 2
    Q = np.eye(n, n)
    R = np.eye(m)
    sigma = np.eye(n, n)

    # maximum number of iterations
    T = 50

    # tolerance of convergence for norm of the gradient
    tol = 1.0e-12

    data_f = []
    data_K = []
    for i in range(100):
        print(['iteration =', i])
        [temp_f, temp_K] = random_sys_opt()
        data_f.append(temp_f)
        data_K.append(temp_K)
    
    plot_traj()
