from cvxopt import matrix, solvers
import numpy as np
from scipy import sparse

def get_u_vec(i, j, n):
    u_vec = np.zeros(n*(n+1)//2)
    pos = (j-1) * n + i - j*(j-1)//2
    u_vec[pos-1] = 1
    return u_vec


def get_T_mat(i, j, n):
    Tij_mat = np.zeros((n, n))
    Tij_mat[i-1, j-1] = Tij_mat[j-1, i-1] = 1
    return np.ravel(Tij_mat)


def create_dup_matrix(num_vertices):
    M_mat = sparse.csr_matrix((num_vertices**2, num_vertices*(num_vertices + 1)//2))
    # tmp_mat = np.arange(num_vertices**2).reshape(num_vertices, num_vertices)
    for j in range(1, num_vertices+1):
        for i in range(j, num_vertices+1):
            u_vec = sparse.csr_matrix(get_u_vec(i, j, num_vertices))
            Tij = sparse.csr_matrix(get_T_mat(i, j, num_vertices))
            # pdb.set_trace()
            M_mat += sparse.kron(u_vec, Tij).reshape(u_vec.shape[1], Tij.shape[1]).T
    return M_mat


def get_a_vec(i, n):
    a_vec = np.zeros(n*(n+1)//2)
    if i == 0:
        a_vec[np.arange(n)] = 1
    else:
        tmp_vec = np.arange(n-1, n-i-1, -1)
        tmp2_vec = np.append([i], tmp_vec)
        tmp3_vec = np.cumsum(tmp2_vec)
        a_vec[tmp3_vec] = 1
        end_pt = tmp3_vec[-1]
        a_vec[np.arange(end_pt, end_pt + n-i)] = 1

    return a_vec

def create_A_mat(n):
    A_mat = np.zeros((n+1, n*(n+1)//2))
    # A_mat[0, 0] = 1
    # A_mat[0, np.cumsum(np.arange(n, 0, -1))] = 1
    for i in range(0, A_mat.shape[0] - 1):
        A_mat[i, :] = get_a_vec(i, n)
    A_mat[n, 0] = 1
    A_mat[n, np.cumsum(np.arange(n, 1, -1))] = 1

    return A_mat


def create_b_mat(n):
    b_mat = np.zeros(n+1)
    b_mat[n] = n
    return b_mat


def create_G_mat(n):
    G_mat = np.zeros((n*(n-1)//2, n*(n+1)//2))
    tmp_vec = np.cumsum(np.arange(n, 1, -1))
    tmp2_vec = np.append([0], tmp_vec)
    tmp3_vec = np.delete(np.arange(n*(n+1)//2), tmp2_vec)
    for i in range(G_mat.shape[0]):
        G_mat[i, tmp3_vec[i]] = 1

    return G_mat

def create_static_matrices_for_L_opt(num_vertices, beta):
    # Static matrices are those independent of Y
    M_mat = create_dup_matrix(num_vertices)
    P_mat = sparse.csr_matrix(2 * beta * np.dot(M_mat.T, M_mat))
    A_mat = create_A_mat(num_vertices)
    b_mat = create_b_mat(num_vertices)
    G_mat = create_G_mat(num_vertices)
    h_mat = np.zeros(G_mat.shape[0])
    return M_mat, P_mat, A_mat, b_mat, G_mat, h_mat


def gl_sig_model(inp_signal, max_iter, alpha, beta):
    """
    inp_signal: num_sig * num_nodes

    Returns Output Signal Y, Graph Laplacian L
    """
    Y = inp_signal.T
    num_vertices = inp_signal.shape[1]
    M_mat, P_mat, A_mat, b_mat, G_mat, h_mat = create_static_matrices_for_L_opt(num_vertices, beta)
    # For convenience, only M_mat is sparse format (csr)
    # M_c = matrix(M_mat)
    # P_c = matrix(P_mat)
    A_c = matrix(A_mat)
    b_c = matrix(b_mat)
    G_c = matrix(G_mat)
    h_c = matrix(h_mat)
    curr_cost = np.linalg.norm(np.ones((num_vertices, num_vertices)), 'fro')
    q_mat = alpha * np.dot(sparse.csr_matrix(np.ravel(np.dot(Y, Y.T))), M_mat)
    for it in range(max_iter):
        # pdb.set_trace()
        # Update L
        prev_cost = curr_cost
        # pdb.set_trace()
        # q_c = matrix(q_mat)
        q_c = q_mat
        sol = solvers.qp(P_c, q_c, G_c, h_c, A_c, b_c)
        l_vech = np.array(sol['x'])
        l_vec = np.dot(M_mat, sparse.csr_matrix(l_vech))
        L = l_vec.reshape(num_vertices, num_vertices).todense()
        # Assert L is correctly learnt.
        # assert L.trace() == num_vertices
        assert np.allclose(L.trace(), num_vertices)
        assert np.all(L - np.diag(np.diag(L)) <= 0)
        assert np.allclose(np.dot(L, np.ones(num_vertices)), np.zeros(num_vertices))
        # print('All constraints satisfied')
        # Update Y
        Y = np.dot(np.linalg.inv(np.eye(num_vertices) + alpha * L), inp_signal.T)

        curr_cost = (np.linalg.norm(inp_signal.T - Y, 'fro')**2 +
                     alpha * np.dot(np.dot(Y.T, L), Y).trace() +
                     beta * np.linalg.norm(L, 'fro')**2)
        q_mat = alpha * np.dot(sparse.csr_matrix(np.ravel(np.dot(Y, Y.T))), M_mat).todense()
        # pdb.set_trace()
        calc_cost = (0.5 * np.dot(np.dot(sparse.csr_matrix(l_vech.T), P_mat), sparse.csr_matrix(l_vech)).todense().squeeze() +
                     np.dot(q_mat, l_vech).squeeze() + np.linalg.norm(inp_signal.T - Y, 'fro')**2)
        # pdb.set_trace()
        assert np.allclose(curr_cost, calc_cost)
        # print(curr_cost)
        if np.abs(curr_cost - prev_cost) < 1e-4:
            # print('Stopped at Iteration', it)
            break
        # print
    return L, Y

def get_adjacent_matrix(data):
    '''
        data: cell * genes
    '''
    # get adjacent matrix
    alpha = 0.0032
    beta = 0.1
    epochs = 1000
    L, Y = gl_sig_model(data.T, epochs, alpha, beta)
    # L = D - W
    W_out = -L_out
    np.fill_diagonal(W_out, 0)
    W_out[W_out < thresh] = 0
    return W_out