from numpy import array, zeros, full, argmin, inf, ndim
from scipy.spatial.distance import cdist
from math import isinf
import torch
import numpy as np


"""
Dynamic time warping (DTW) is used as a similarity measured between temporal sequences. 
Original DTW code found at https://github.com/pierre-rouanet/dtw

"""

# Find the best timing match between a reference and a hypothesis, using DTW
def calculate_dtw(references, hypotheses):
    """
    Calculate the DTW costs between a list of references and hypotheses

    :param references: list of reference sequences to compare against
    :param hypotheses: list of hypothesis sequences to fit onto the reference

    :return: dtw_scores: list of DTW costs
    """
    # Euclidean norm is the cost function, difference of coordinates
    euclidean_norm = lambda x, y: np.sum(np.abs(x - y))

    dtw_scores = []

    # Remove the BOS frame from the hypothesis
    # hypotheses = hypotheses[:, 1:]

    # For each reference in the references list
    for i, ref in enumerate(references):
        # Cut the reference down to the max count value
        _ , ref_max_idx = torch.max(ref[:, -1], 0)
        if ref_max_idx == 0: ref_max_idx += 1
        # Cut down frames by to the max counter value, and chop off counter from joints
        ref_count = ref[:ref_max_idx,:-1].cpu().numpy()

        # Cut the hypothesis down to the max count value
        hyp = hypotheses[i]
        _, hyp_max_idx = torch.max(hyp[:, -1], 0)
        if hyp_max_idx == 0: hyp_max_idx += 1
        # Cut down frames by to the max counter value, and chop off counter from joints
        hyp_count = hyp[:hyp_max_idx,:-1].cpu().numpy()

        # Calculate DTW of the reference and hypothesis, using euclidean norm
        d, cost_matrix, acc_cost_matrix, path = dtw(ref_count, hyp_count, dist=euclidean_norm)

        # Normalise the dtw cost by sequence length
        d = d/acc_cost_matrix.shape[0]

        dtw_scores.append(d)

    # Return dtw scores and the hypothesis with altered timing
    return dtw_scores


# Apply DTW
def dtw(x, y, dist, warp=1, w=inf, s=1.0):
    """
    Computes Dynamic Time Warping (DTW) of two sequences.

    :param array x: N1*M array
    :param array y: N2*M array
    :param func dist: distance used as cost measure
    :param int warp: how many shifts are computed.
    :param int w: window size limiting the maximal distance between indices of matched entries |i,j|.
    :param float s: weight applied on off-diagonal moves of the path. As s gets larger, the warping path is increasingly biased towards the diagonal
    Returns the minimum distance, the cost matrix, the accumulated cost matrix, and the wrap path.
    """
    assert len(x)
    assert len(y)
    assert isinf(w) or (w >= abs(len(x) - len(y)))
    assert s > 0
    r, c = len(x), len(y)
    if not isinf(w):
        D0 = full((r + 1, c + 1), inf)
        for i in range(1, r + 1):
            D0[i, max(1, i - w):min(c + 1, i + w + 1)] = 0
        D0[0, 0] = 0
    else:
        D0 = zeros((r + 1, c + 1))
        D0[0, 1:] = inf
        D0[1:, 0] = inf
    D1 = D0[1:, 1:]  # view
    for i in range(r):
        for j in range(c):
            if (isinf(w) or (max(0, i - w) <= j <= min(c, i + w))):
                D1[i, j] = dist(x[i], y[j])
    C = D1.copy()
    jrange = range(c)
    for i in range(r):
        if not isinf(w):
            jrange = range(max(0, i - w), min(c, i + w + 1))
        for j in jrange:
            min_list = [D0[i, j]]
            for k in range(1, warp + 1):
                i_k = min(i + k, r)
                j_k = min(j + k, c)
                min_list += [D0[i_k, j] * s, D0[i, j_k] * s]
            D1[i, j] += min(min_list)
    if len(x) == 1:
        path = zeros(len(y)), range(len(y))
    elif len(y) == 1:
        path = range(len(x)), zeros(len(x))
    else:
        path = _traceback(D0)
    return D1[-1, -1], C, D1, path

def _traceback(D):
    i, j = array(D.shape) - 2
    p, q = [i], [j]
    while (i > 0) or (j > 0):
        tb = argmin((D[i, j], D[i, j + 1], D[i + 1, j]))
        if tb == 0:
            i -= 1
            j -= 1
        elif tb == 1:
            i -= 1
        else:  # (tb == 2):
            j -= 1
        p.insert(0, i)
        q.insert(0, j)
    return array(p), array(q)


if __name__ == '__main__':
    x = np.random.rand(10, 150)
    # y = np.random.rand(10, 150)
    y = x
    euclidean_norm = lambda x, y: np.sum(np.abs(x - y))
    d, cost_matrix, acc_cost_matrix, path = dtw(x, y, dist=euclidean_norm)

    # Normalise the dtw cost by sequence length
    # d = d/acc_cost_matrix.shape[0]

    print(d, acc_cost_matrix.shape[0])

    # x2 = np.random.randn(5, 10, )
    # w = inf
    # s = 1.0
    # if 1:  # 1-D numeric
    #     from sklearn.metrics.pairwise import manhattan_distances
    #     x = [0, 0, 1, 1, 2, 4, 2, 1, 2, 0]
    #     y = [1, 1, 1, 2, 2, 2, 2, 3, 2, 0]
    #     dist_fun = manhattan_distances
    #     w = 1
    #     # s = 1.2
    # elif 0:  # 2-D numeric
    #     from sklearn.metrics.pairwise import euclidean_distances
    #     x = [[0, 0], [0, 1], [1, 1], [1, 2], [2, 2], [4, 3], [2, 3], [1, 1], [2, 2], [0, 1]]
    #     y = [[1, 0], [1, 1], [1, 1], [2, 1], [4, 3], [4, 3], [2, 3], [3, 1], [1, 2], [1, 0]]
    #     dist_fun = euclidean_distances
    # else:  # 1-D list of strings
    #     from nltk.metrics.distance import edit_distance
    #     # x = ['we', 'shelled', 'clams', 'for', 'the', 'chowder']
    #     # y = ['class', 'too']
    #     x = ['i', 'soon', 'found', 'myself', 'muttering', 'to', 'the', 'walls']
    #     y = ['see', 'drown', 'himself']
    #     # x = 'we talked about the situation'.split()
    #     # y = 'we talked about the situation'.split()
    #     dist_fun = edit_distance
    # dist, cost, acc, path = dtw(x, y, dist_fun, w=w, s=s)

    # # Vizualize
    # from matplotlib import pyplot as plt
    # plt.imshow(cost.T, origin='lower', cmap=plt.cm.Reds, interpolation='nearest')
    # plt.plot(path[0], path[1], '-o')  # relation
    # plt.xticks(range(len(x)), x)
    # plt.yticks(range(len(y)), y)
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.axis('tight')
    # if isinf(w):
    #     plt.title('Minimum distance: {}, slope weight: {}'.format(dist, s))
    # else:
    #     plt.title('Minimum distance: {}, window widht: {}, slope weight: {}'.format(dist, w, s))
    # plt.show()
