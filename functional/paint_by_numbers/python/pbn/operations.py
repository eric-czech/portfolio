__author__ = 'eczech'

from random import random as rnd
import numpy as np
import pandas as pd
from sklearn import mixture


def unravel(d):
    n2d = d.shape[0] * d.shape[1]
    colors = np.empty((n2d, 3))
    for i in range(d.shape[0]):
        for j in range(d.shape[1]):
            z = i * d.shape[1] + j
            colors[z] = d[i, j, :] * 255
    return pd.DataFrame(colors, columns=['R', 'G', 'B'])


def reravel(d, x, y):
    dc = np.empty((x, y, d.shape[1]))
    for row in d.iterrows():
        i, r = row[0], row[1]
        r_i = int(i / float(dc.shape[1]))
        r_j = i % dc.shape[1]
        dc[r_i, r_j] = r
    return dc

def cluster_color_space(img_df, max_clusters=5):
    dpgmm = mixture.DPGMM(n_components=max_clusters)
    dpgmm.fit(img_df)
    pred = dpgmm.predict(img_df)
    return pd.Series(pred)


def image_from_clusters(spatial_clusters, color_clusters, img_nd, use_random_color=False):
    img = np.empty_like(img_nd)
    for cc_id, cs in spatial_clusters.items():
        color = color_clusters.loc[cc_id] / float(255)
        for c in cs:
            if use_random_color:
                color = rnd(), rnd(), rnd()
            for p in c:
                img[p[0], p[1]] = color
    return img

def cluster_euclidean_space(img_nd):
    return _get_clusters(img_nd)

__OFFSETS = [(-1, -1), (-1, 0), (-1, 1), (1, -1), (1, 0), (1, 1), (0, -1), (0, 1)]

def _traverse(node, d):
    res = {node[0]: node[1]}
    stack = [res.items()[0]]
    while len(stack) > 0:
        loc, value = stack.pop(-1)
        for offset in __OFFSETS:
            neighbor = _get_neighbor(offset, loc, d)
            if neighbor\
                and not neighbor[0] in res\
                and neighbor[1] == value:
                res[neighbor[0]] = neighbor[1]
                stack.append(neighbor)
    return res

def _get_neighbor(offset, loc, d):
    i, j = loc
    ni, nj = i + offset[0], j + offset[1]
    if ni < 0 or ni >= d.shape[0]:
        return None
    if nj < 0 or nj >= d.shape[1]:
        return None
    return (ni, nj), d[ni, nj]

def _get_clusters(d):
    candidates = dict([(i, v) for i, v in np.ndenumerate(d)])
    clusters = {}
    while len(candidates) > 0:
        #print 'canidate len:', len(candidates)
        loc, val = candidates.popitem()
        if val not in clusters:
            clusters[val] = []
        res = _traverse((loc, val), d)
        clusters[val].append(res.keys())
        for k in res:
            if k != loc:
                candidates.pop(k)
    return clusters


LABEL_MARGIN = 4

def get_cluster_props(spatial_clusters, color_clusters):
    res = {}
    for cc_id, cs in spatial_clusters.items():
        if cc_id not in res:
            res[cc_id] = []
        color = color_clusters.loc[cc_id]
        for cluster in cs:
            members = set(cluster)
            edges, labels = [], []
            for point in cluster:
                if not _is_patch_enclosed(point, members, 1):
                    edges.append(point)
                if _is_patch_enclosed(point, members, LABEL_MARGIN):
                    labels.append(point)

            default_label = None
            if len(labels) > 0:
                default_label = _get_patch(labels[0], delta=LABEL_MARGIN)

            res[cc_id].append({
                'color': color,
                'points': cluster,
                'edges': edges,
                'labels': labels,
                'default_label': default_label
            })
    return res


def _get_patch(point, delta=1):
    x, y = point
    res = []
    for dx in range(-delta, delta+1):
        for dy in range(-delta, delta+1):
            res.append((x+dx, y+dy))
    return res

def _is_patch_enclosed(point, members, delta):
    for point in _get_patch(point, delta=delta):
        if point not in members:
            return False
    return True


def render_pbn(fc, img_nd, bkg=[.99, .99, .99], edg=[0., 0., 0.], solution=False, size_limit=100):
    res = np.empty_like(img_nd)
    for cc_id in fc:
        for c in fc[cc_id]:
            actual_color = c['color']/float(255)
            def_label = c['default_label']
            use_actual = solution or not def_label or len(c['points']) < size_limit
            color = actual_color if use_actual else bkg

            for point in c['points']:
                res[point] = color

            if not use_actual:
                for point in c['edges']:
                    res[point] = edg

            if def_label:
                for point in def_label:
                    res[point] = actual_color
    return res
