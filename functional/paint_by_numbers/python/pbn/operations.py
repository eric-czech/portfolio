__author__ = 'eczech'

from random import random as rnd
import numpy as np
import pandas as pd
from sklearn import mixture
import math

def unravel_old(d):
    n2d = d.shape[0] * d.shape[1]
    colors = np.empty((n2d, 3))
    for i in range(d.shape[0]):
        for j in range(d.shape[1]):
            z = i * d.shape[1] + j
            colors[z] = d[i, j, :] * 255
    return pd.DataFrame(colors, columns=['R', 'G', 'B'])


def unravel(d, color_cols=['l', 'a', 'b']):
    n2d = d.shape[0] * d.shape[1]
    colors = np.empty((n2d, 5))
    for i in range(d.shape[0]):
        for j in range(d.shape[1]):
            z = i * d.shape[1] + j
            colors[z] = [float(i), float(j)] + list(d[i, j, :])
    return pd.DataFrame(colors, columns=['x', 'y'] + color_cols)


def reravel(d, x, y):
    assert x * y == d.shape[0],\
        'Num rows in flat input array ({}) does not match rows * cols in reraveled array ({})'\
        .format(d.shape[0], x * y)
    dc = np.empty((x, y, d.shape[1]))
    for row in d.iterrows():
        i, r = row[0], row[1]
        r_i = int(i / float(dc.shape[1]))
        r_j = i % dc.shape[1]
        dc[r_i, r_j] = r
    return dc


def cluster_color_space(img_df, **kwargs):
    dpgmm = mixture.DPGMM(**kwargs)
    dpgmm.fit(img_df)
    pred = dpgmm.predict(img_df)
    return pd.Series(pred)

def cluster_color_space2(img_df, **kwargs):
    dpgmm = mixture.GMM(**kwargs)
    dpgmm.fit(img_df)
    pred = dpgmm.predict(img_df)
    return pd.Series(pred)


def blur(img_nd):
    for i, v in np.ndenumerate(img_nd):
        patch = _get_patch(i, delta=1)
        votes = {}
        for point in patch:
            x, y = point
            if point == i:
                continue
            if not _is_in_bounds(point, img_nd):
                continue
            vp = img_nd[point]
            if vp not in votes:
                votes[vp] = 0
            votes[vp] += 1
        vsame = votes.get(v, 0)
        if vsame > 1:
            continue
        votes = pd.Series(votes).order(ascending=False)
        top_c, top_v = votes.index[0], votes.iloc[0]
        img_nd[i] = top_c
    return img_nd

# def blur(img_nd):
#     for i, v in np.ndenumerate(img_nd):
#         patch = _get_patch(i, delta=1)
#         votes = {}
#         for point in patch:
#             x, y = point
#             if point == i:
#                 continue
#             if x < 0 or x >= img_nd.shape[0]:
#                 continue
#             if y < 0 or y >= img_nd.shape[1]:
#                 continue
#             vp = img_nd[point]
#             if not vp in votes:
#                 votes[vp] = 0
#             votes[vp] += 1
#         vsame = votes.get(v, 0)
#         if vsame > 1:
#             continue
#         votes = pd.Series(votes).order(ascending=False)
#         top_c, top_v = votes.index[0], votes.iloc[0]
#         if vsame > 0:
#             if top_v > 5:
#                 img_nd[i] = top_c
#         if vsame == 0:
#             img_nd[i] = top_c
#     return img_nd


def image_from_clusters(spatial_clusters, color_clusters, img_nd, use_random_color=False):
    img = np.empty_like(img_nd)
    for cc_id, cs in spatial_clusters.items():
        color = color_clusters.loc[cc_id]
        for c in cs:
            if use_random_color:
                color = rnd(), rnd(), rnd()
            for p in c:
                img[p[0], p[1]] = color
    return img


def collapse_clusters(spatial_clusters, color_matrix, **kwargs):
    res = np.array(color_matrix, copy=True)
    n_collapses = 0
    n_total = 0
    for cc_id, cs in spatial_clusters.items():
        for cluster in cs:
            n_total += 1
            members = set(cluster)
            neighbors = {}
            for point in cluster:
                is_edge, nbs = _get_point_props(point, members, color_matrix)
                neighbors.update(nbs)

            collapsed_id = _get_collapsed_cluster(cc_id, cluster, neighbors, **kwargs)
            # print(collapsed_id)
            if collapsed_id != cc_id:
                n_collapses += 1
                # print('Collapsing cluster with {} members'.format(len(cluster)))
                for point in cluster:
                    res[point] = collapsed_id
    print('{} clusters collapsed of {}'.format(n_collapses, n_total))
    return res

from scipy import ndimage



def cluster_by_proximity(color_matrix, n_iterations=1, **kwargs):
    spatial_clusters = _get_clusters(color_matrix)
    for _ in range(n_iterations):
        color_matrix = collapse_clusters(spatial_clusters, color_matrix, **kwargs)
        spatial_clusters = _get_clusters(color_matrix)
    return spatial_clusters


# def cluster_euclidean_space(img_nd):
#    return _get_clusters(img_nd)


__OFFSETS = [(-1, -1), (-1, 0), (-1, 1), (1, -1), (1, 0), (1, 1), (0, -1), (0, 1)]


def _traverse(node, d):
    res = {node[0]: node[1]}
    stack = [list(res.items())[0]]
    while len(stack) > 0:
        loc, value = stack.pop(-1)
        for offset in __OFFSETS:
            neighbor = _get_neighbor(offset, loc, d)
            if neighbor \
                    and not neighbor[0] in res \
                    and neighbor[1] == value:
                res[neighbor[0]] = neighbor[1]
                stack.append(neighbor)
    return res


def _get_neighbor(offset, loc, d):
    i, j = loc
    ni, nj = i + offset[0], j + offset[1]
    if not _is_in_bounds((ni, nj), d):
        return None
    return (ni, nj), d[ni, nj]


def _get_clusters(d):
    candidates = dict([(i, v) for i, v in np.ndenumerate(d)])
    clusters = {}
    while len(candidates) > 0:
        # print 'candidate len:', len(candidates)
        loc, val = candidates.popitem()
        if val not in clusters:
            clusters[val] = []
        res = _traverse((loc, val), d)
        clusters[val].append(res.keys())
        for k in res:
            if k != loc:
                candidates.pop(k)
    return clusters


LABEL_MARGIN = 2


def get_cluster_props(spatial_clusters, color_clusters, color_matrix):
    res = {}
    for cc_id, cs in spatial_clusters.items():
        if cc_id not in res:
            res[cc_id] = []
        color = color_clusters.loc[cc_id]
        for cluster in cs:
            members = set(cluster)
            neighbors = {}
            edges, labels = [], []
            for point in cluster:
                is_edge, nbs = _get_point_props(point, members, color_matrix)
                if is_edge:
                    edges.append(point)
                if _is_patch_enclosed(point, members, LABEL_MARGIN):
                    labels.append(point)
                neighbors.update(nbs)

            default_label = None
            if len(labels) > 0:
                default_label = _get_patch(labels[0], delta=LABEL_MARGIN)

            res[cc_id].append({
                'color': color,
                'points': cluster,
                'edges': edges,
                'labels': labels,
                'default_label': default_label,
                'neighbors': neighbors
            })
    return res


def _get_collapsed_cluster(current_id, points, neighbors, shrinkage=2, threshold=.6):
    votes = pd.Series(list(neighbors.values())).value_counts()
    cc_votes = math.pow(len(points), 1/float(shrinkage))
    top_neighbor = votes.order(ascending=False)[:1]
    n_id, n_votes = top_neighbor.index.values[0], top_neighbor.iloc[0]
    #print(cc_votes, n_votes, len(points), len(neighbors))
    if n_votes > cc_votes and float(n_votes)/votes.sum() > threshold:
        # print(cc_votes, votes)
        return n_id
    return current_id


def _is_in_bounds(p, matrix):
    x, y = p
    if x < 0 or x >= matrix.shape[0]:
        return False
    if y < 0 or y >= matrix.shape[1]:
        return False
    return True


def _get_point_props(point, members, color_matrix):
    is_edge = False
    neighbors = {}
    for p in _get_patch(point, delta=1):
        if p not in members:
            is_edge = True
            if _is_in_bounds(p, color_matrix):
                neighbors[p] = color_matrix[p]
    return is_edge, neighbors


def _get_patch(point, delta=1):
    x, y = point
    res = []
    for dx in range(-delta, delta + 1):
        for dy in range(-delta, delta + 1):
            res.append((x + dx, y + dy))
    return res


def _is_patch_enclosed(point, members, delta):
    for point in _get_patch(point, delta=delta):
        if point not in members:
            return False
    return True


def render_pbn(fc, img_nd, bkg=[.99, .99, .99], edg=[1., 1., 1.], solution=False, size_limit=25, ):
    res = np.empty_like(img_nd)
    for cc_id in fc:
        for c in fc[cc_id]:
            actual_color = c['color']  # this was dividing by 255 before
            def_label = c['default_label']
            use_actual = solution or len(c['points']) < size_limit
            color = actual_color if use_actual else bkg

            for point in c['points']:
                res[point] = color

            if not use_actual:
                for point in c['edges']:
                    res[point] = actual_color if edg is None else edg

            if def_label:
                for point in def_label:
                    res[point] = actual_color
    return res
