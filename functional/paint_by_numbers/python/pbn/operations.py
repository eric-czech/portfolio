__author__ = 'eczech'

from random import random as rnd
import numpy as np
import pandas as pd
from sklearn import mixture
import math
import itertools
from . import dots, conversions

LABEL_MARGIN = 3


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
            colors[z] = [i, j] + list(d[i, j, :])
    res = pd.DataFrame(colors, columns=['x', 'y'] + color_cols)
    res[['x', 'y']] = res[['x', 'y']].astype(np.int64)
    return res


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


def get_cluster_props(spatial_clusters, color_clusters, color_matrix):
    res = {}
    for cc_id, cs in spatial_clusters.items():
        if cc_id not in res:
            res[cc_id] = []
        color = color_clusters.loc[cc_id]
        for cluster in cs:
            members = set(cluster)
            neighbors = {}
            edges = []
            for point in cluster:
                is_edge, nbs = _get_point_props(point, members, color_matrix)
                if is_edge:
                    edges.append(point)

                #if _is_patch_enclosed(point, members, LABEL_MARGIN):
                #if _is_shape_enclosed(members, _get_label_patch(point)):
                    #labels.append(point)
                neighbors.update(nbs)

            res[cc_id].append({
                'color': color,
                'points': cluster,
                'edges': edges,
                #'labels': labels,
                'neighbors': neighbors
            })
    return res


def _get_patch(point, delta=1):
    return _get_shape_patch(point, range(-delta, delta+1), range(-delta, delta+1))


def _is_shape_enclosed(members, candidates):
    for point in candidates:
        if point not in members:
            return False
    return True


def _is_patch_enclosed(point, members, delta):
    return _is_shape_enclosed(members, _get_patch(point, delta))


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


def _add_to_result(p, neighbors, color, edg_color, img_orig, img_res, is_edge, scale):
    res = np.empty((scale, scale, img_orig.shape[2]))
    res[:, :] = color
    if is_edge:
        s = scale - 1
        for n in _get_patch(p):
            if _is_in_bounds(n, img_orig) and n in neighbors:
                if n[0] > p[0] and n[1] > p[1]:
                    res[s, s] = edg_color
                elif n[0] > p[0] and n[1] < p[1]:
                    res[s, 0] = edg_color
                elif n[0] > p[0] and n[1] == p[1]:
                    res[s, :] = edg_color
                elif n[0] < p[0] and n[1] > p[1]:
                    res[0, s] = edg_color
                elif n[0] < p[0] and n[1] < p[1]:
                    res[0, 0] = edg_color
                elif n[0] < p[0] and n[1] == p[1]:
                    res[0, :] = edg_color
                elif n[0] == p[0] and n[1] > p[1]:
                    res[:, s] = edg_color
                elif n[0] == p[0] and n[1] < p[1]:
                    res[:, 0] = edg_color
    r, c = p[0] * scale, p[1] * scale
    img_res[r:(r+scale), c:(c+scale)] = res


def _draw_cluster_label(p, renderer, color_cluster, color_lbl, color_bkg, img_res, scale):
    r, c = p[0] * scale, p[1] * scale
    dot_matrix = renderer.get_matrix_for_key(tuple(color_cluster), color_lbl, color_bkg)

    # Center location for dot matrix
    ro, co = int(dot_matrix.shape[0]/2), int(dot_matrix.shape[1]/2)
    r, c = r - ro, c - co

    # Add label to image if it still fits when centered
    rd, cd = r + dot_matrix.shape[0], c + dot_matrix.shape[1]
    if r >= 0 and c >= 0 and rd <= img_res.shape[0] and cd <= img_res.shape[1]:
        img_res[r:rd, c:cd, :] = dot_matrix


def distance_between(p1, p2):
    return np.sqrt((p2[1] - p1[1])**2 + (p2[0] - p1[0])**2)


def _get_label_renderer(clusters):
    colors = []
    for cc_id in clusters:
        for c in clusters[cc_id]:
            colors.append(tuple(c['color']))
    colors.sort()
    colors = list(c for c, _ in itertools.groupby(colors))
    renderer = dots.get_label_renderer()
    if len(colors) > renderer.get_max_colors():
        raise ValueError('Currently no more than {} unique colors are supported'.format(renderer.get_max_colors()))
    labels = renderer.get_labels()[:len(colors)]
    color_index = dict(zip(colors, labels))
    renderer.set_keys_for_labels(color_index)
    return renderer, color_index


def _get_closest_to_median(points):
    if points is None or len(points) == 0:
        return None
    median_p = (np.median([p[0] for p in points]), np.median([p[1] for p in points]))

    closest_p, min_dist = points[0], np.inf
    for p in points:
        dist = distance_between(p, median_p)
        if dist < min_dist:
            closest_p = p
            min_dist = dist
    return closest_p


import math


def _get_label_patch(point, scale_factor):
    return _get_shape_patch(point, range(-1, 2), range(-1, 1))


def _get_shape_patch(point, xrange, yrange):
    res = []
    x, y = point
    for i in xrange:
        for j in yrange:
            res.append((x + i, y + j))
    return res


def render_pbn(fc, img_nd, alpha,
               bkg=[.99, .99, .99], edg=[1., 1., 1.], lbl=[.99, .99, .99],
               solution=False, scale_factor=1):

    bkg = np.array(conversions._rgb_to_lab(bkg)) * alpha
    edg = np.array(conversions._rgb_to_lab(edg)) * alpha
    lbl = np.array(conversions._rgb_to_lab(lbl)) * alpha

    img_res = np.empty((img_nd.shape[0] * scale_factor, img_nd.shape[1] * scale_factor, img_nd.shape[2]))

    label_renderer, color_index = _get_label_renderer(fc)

    for cc_id in fc:
        for c in fc[cc_id]:
            actual_color = c['color']
            # color = actual_color if solution else bkg
            # color = (rnd(), rnd(), rnd()) if use_actual else bkg

            members = set(c['points'])
            labels = []
            for point in c['points']:
                if _is_shape_enclosed(members, _get_label_patch(point, scale_factor)):
                    labels.append(point)

            color = actual_color if solution or len(labels) == 0 else bkg
            for point in c['points']:
                _add_to_result(point, c['neighbors'], color, edg, img_nd, img_res, False, scale_factor)

            if not solution and len(labels) > 0:
                for point in c['edges']:
                    _add_to_result(point, c['neighbors'], color, edg, img_nd, img_res, True, scale_factor)

                closest_p = _get_closest_to_median(labels)
                if closest_p is not None:
                    _draw_cluster_label(closest_p, label_renderer, actual_color, lbl, bkg, img_res, scale_factor)

    return img_res, color_index
