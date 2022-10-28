# Code to generate Figure 4.a
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


def rand_bin_matrix(n1, m1, k1):
    # outputs a random binary matrix of size n1xm1 with k1 number of 1 entries
    arr = np.zeros(n1 * m1)
    arr[:k1] = 1
    np.random.shuffle(arr)
    return arr.reshape((n1, m1))


def random_sys_opt():
    # checking controllability of (A,B) and ensuring open-loop stability of A
    global A
    global B
    global Kappa

    while True:
        A = np.random.rand(n, n)
        B = np.random.rand(n, m)
        if np.linalg.matrix_rank(ctr.ctrb(A, B)) == n:
            A *= 0.9 / (np.max(np.abs(la.eigvals(A))))
            break

    # sparsity constraint as a matrix of 0,1 that indicates zero entries of K
    # Kappa = np.random.randint(2, size=K0.shape)
    Kappa = rand_bin_matrix(m, n, int(n * m / 2))

    # Initial structured stabilizing controller as zero due to open-loop stability
    K0 = np.zeros((m, n))

    # optimal unstructured state-feedback discrete-time infinite-horizon LQR controller
    P_lqr = la.solve_discrete_are(A, B, Q, R)
    K_lqr = -la.inv(R + B.T @ P_lqr @ B) @ B.T @ P_lqr @ A

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

        if la.norm(tan_proj(K_r[-1], grad_f(K_r[-1]))) > tol:
            # computing newton direction
            nd_r.append(newton_direction(K_r[-1], 'Riemannian'))

            # computing stepsize
            const_r.append(stepsize_bound(K_r[-1], nd_r[-1]))
            if stepsize_rule == 'best':
                eta_r.append(np.min([1, const_r[-1]]))
            elif stepsize_rule == 'constant':
                eta_r.append(const_r[0])

            K_new_r = K_r[-1] + eta_r[-1] * nd_r[-1]
            # K_new_r = K_r[-1] + 1 * nd_r[-1]
            K_r.append(K_new_r)
        else:
            K_r.append(K_r[-1])

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

        if la.norm(tan_proj(K_euc[-1], grad_f(K_euc[-1]))) > tol:
            # computing newton direction
            nd_euc.append(newton_direction(K_euc[-1], 'Euclidean'))

            # computing stepsize
            const_euc.append(stepsize_bound(K_euc[-1], nd_euc[-1]))
            if stepsize_rule == 'best':
                eta_euc.append(np.min([1, const_euc[-1]]))
            elif stepsize_rule == 'constant':
                eta_euc.append(const_euc[0])

            K_new_euc = K_euc[-1] + eta_euc[-1] * nd_euc[-1]
            # K_new_euc = K_euc[-1] + 1 * nd_euc[-1]
            K_euc.append(K_new_euc)
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

    # ### Learning the optimal structured LQR controller via Riemannian Projected Gradient #########
    # choose the stepsize rule to be 'constant' or 'best'
    stepsize_rule = 'constant'
    K_nat = [K0]
    pg_direction = []
    const_nat = []
    eta_nat = []
    T_nat = T
    for _ in range(T_nat):
        if np.max(np.abs(la.eigvals(A_cl(K_nat[-1])))) > 1:
            print('Unstable controller in projected-gradient (Euclidean) algorithm at iteration = ', _)

        Riem_gradient_f = grad_f(K_nat[-1])
        pg_direction.append(tan_proj(K_nat[-1], Riem_gradient_f))

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

    # ################# computing the error from optimality ####################################################
    if (Kappa.ravel() == 1).all():
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
    fig.savefig('images/random_slqr_extended.pdf', format='pdf')


if __name__ == '__main__':
    # problem parameters
    n = 6
    m = 3
    Q = np.eye(n, n)
    R = np.eye(m)
    sigma = np.eye(n, n)

    # maximum number of iterations
    T = 50

    # tolerance of convergence for norm of the gradient
    tol = 1.0e-14

    data_f = []
    data_K = []
    for i in range(100):
        print(['iteration =', i])
        [temp_f, temp_K] = random_sys_opt()
        data_f.append(temp_f)
        data_K.append(temp_K)

    plot_traj()
