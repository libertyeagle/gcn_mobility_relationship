import scipy.sparse as sp
import numpy as np
import random
from sklearn.preprocessing import StandardScaler

def normalize_adj(adj):
    adj = sp.coo_matrix(adj)
    row_sum = np.array(adj.sum(1))
    d_inv_sqrt  = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_adj

def sparse_to_tuple(sp_mat):
    if not sp.isspmatrix_coo(sp_mat):
        sp_mat = sp_mat.tocoo()
    coords = np.vstack((sp_mat.row, sp_mat.col)).transpose()
    values = sp_mat.data
    shape = sp_mat.shape
    return coords, values, shape

def load_graph(graph_file_name):
    graph = sp.load_npz(graph_file_name)
    return graph

def construct_supports(graph, num_users, num_pois):
    rows = graph.row
    cols = graph.col
    values = graph.data
    num_nodes = num_users + num_pois
    # add a dummy node with zero vector (feature)
    uu_graph = sp.lil_matrix((num_nodes + 1, num_nodes + 1), dtype='d')
    ll_graph = sp.lil_matrix((num_nodes + 1, num_nodes + 1), dtype='d')
    ul_graph = sp.lil_matrix((num_nodes + 1, num_nodes + 1), dtype='d')
    for row, col, val in zip(rows, cols, values):
        if row < num_users and col < num_users:
            uu_graph[row, col] = val
        elif row >= num_users and col >= num_users:
            ll_graph[row, col] = val
        else:
            ul_graph[row, col] = val

    for i in range(num_users):
        uu_graph[i, i] = 1.
    for i in range(num_pois):
        ll_graph[num_users + i, num_users + i] = 1.

    uu_adj_normalized = normalize_adj(uu_graph)
    ll_adj_normalized = normalize_adj(ll_graph)
    ul_adj_normalized = normalize_adj(ul_graph)
    
    uu_adj_tuple = sparse_to_tuple(uu_adj_normalized)
    ll_adj_tuple = sparse_to_tuple(ll_adj_normalized)
    ul_adj_tuple = sparse_to_tuple(ul_adj_normalized)

    supports = []
    supports.append(uu_adj_tuple)
    supports.append(ll_adj_tuple)
    supports.append(ul_adj_tuple)
    return supports

def construct_semi_supervised_supports(graph, partial_social_graph, num_users, num_pois):
    supports = construct_supports(graph, num_users, num_pois)
    num_nodes = num_users + num_pois
    semi_sup_graph = sp.lil_matrix((num_nodes + 1, num_nodes + 1), dtype='d')
    rows = partial_social_graph.row
    cols = partial_social_graph.col
    for row, col in zip(rows, cols):
        semi_sup_graph[row, col] = 1.
    for i in range(num_users):
        semi_sup_graph[i, i] = 1.
    semi_sup_graph_normalized = normalize_adj(semi_sup_graph)
    semi_sup_tuple = sparse_to_tuple(semi_sup_graph_normalized)
    supports.append(semi_sup_tuple)
    return supports

def load_features(user_emb_file_name, poi_emb_file_name, method='concat'):
    user_embeddings = np.load(user_emb_file_name)
    poi_embeddings = np.load(poi_emb_file_name)
    if method == 'concat':
        zero_pad_user = np.zeros((user_embeddings.shape[0], 
                                poi_embeddings.shape[1]), dtype=np.float32)
        zero_pad_poi = np.zeros((poi_embeddings.shape[0],
                                user_embeddings.shape[1]), dtype=np.float32)
        user_embeddings = np.concatenate((user_embeddings, zero_pad_user), axis=1)
        poi_embeddings = np.concatenate((zero_pad_poi, poi_embeddings), axis=1)
    node_features = np.vstack((user_embeddings, poi_embeddings))
    # add dummy zero vector
    node_features = np.vstack((node_features, np.zeros(user_embeddings.shape[1])))
    scalar = StandardScaler().fit(node_features)
    node_features = scalar.transform(node_features)
    return node_features

def convert_to_adj_list(graph):
    adj_list = {}
    rows = graph.row
    cols = graph.col
    for row, col in zip(rows, cols):
        adj_list.setdefault(row, [])
        adj_list[row].append(col)
    print('number of isolated nodes:', graph.shape[0] - len(adj_list))
    return adj_list

def sample_context(graph, num_users, num_pois, walk_len, num_walks, num_pos_sample):
    graph = convert_to_adj_list(graph)
    num_nodes = num_users + num_pois
    homo_samples = dict()
    heter_samples = dict()

    for node in range(num_nodes):
        if node not in graph:
            continue
        for v in graph[node]:
            if (node < num_users and v < num_users) or \
                (node >= num_users and v >= num_users):
                homo_samples.setdefault(node, [])
                homo_samples[node].append(v)
            else:
                heter_samples.setdefault(node, [])
                heter_samples[node].append(v)

    homo_samples_walk = dict()
    heter_samples_walk = dict()
    for node in range(num_nodes):
        if node not in graph:
            continue
        for _ in range(num_walks):
            curr_node = node
            for _ in range(walk_len):
                next_node = random.choice(graph[curr_node])
                if curr_node != node:
                    if (node < num_users and curr_node < num_users) or \
                       (node >= num_users and curr_node >= num_users):
                        homo_samples_walk.setdefault(node, [])
                        homo_samples_walk[node].append(curr_node)
                    else:
                        heter_samples_walk.setdefault(node, [])
                        heter_samples_walk[node].append(curr_node)
                curr_node = next_node
        if node % 1000 == 0:
            print("finish random walk for", node, "nodes")

    # fill blanks in positive sample:
    homo_samples_matrix = num_nodes * np.ones((num_nodes + 1, num_pos_sample))
    heter_samples_matrix = num_nodes * np.ones((num_nodes + 1, num_pos_sample))
    for node in range(num_nodes):
        if node in homo_samples:
            num_samples = len(homo_samples_walk[node])
            samples = np.random.choice(homo_samples[node], num_pos_sample // 2, replace=True)
            if num_samples >= num_pos_sample:
                samples = np.concatenate((
                    samples,
                    np.random.choice(homo_samples_walk[node], num_pos_sample // 2, replace=False)
                ))
                homo_samples_matrix[node] = samples
            else:
                samples = np.concatenate((samples, 
                          np.random.choice(homo_samples_walk[node], num_pos_sample // 2, replace=True)))
                homo_samples_matrix[node] = samples

        if node not in heter_samples:
            continue
        num_samples = len(heter_samples_walk[node])
        samples = np.random.choice(heter_samples[node], num_pos_sample // 2, replace=True)
        if num_samples > num_pos_sample:
            samples = np.concatenate((
                samples,
                np.random.choice(heter_samples_walk[node], num_pos_sample // 2, replace=False)
            ))
            heter_samples_matrix[node] = samples
        else:
            samples = np.concatenate((samples, 
                     np.random.choice(heter_samples_walk[node], num_pos_sample // 2, replace=True)))
            heter_samples_matrix[node] = samples
    del homo_samples
    del heter_samples

    return homo_samples_matrix, heter_samples_matrix

def sample_semi_supervised_context(partial_social_graph, num_pos_sample):
    num_users = partial_social_graph.shape[0]
    partial_social_graph = convert_to_adj_list(partial_social_graph)
    samples = np.zeros((num_users, num_pos_sample), dtype=np.int32)
    for u in range(num_users):
        if u not in partial_social_graph:
            samples[u] = u * np.ones(num_pos_sample, dtype=np.int32)
            continue
        context = []
        for v in partial_social_graph[u]:
            context.append(v)
        samples[u] = np.random.choice(context, num_pos_sample, replace=True)
    return samples