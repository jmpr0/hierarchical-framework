import numpy as np


def equal_partitioner(nb, ps):
    if list(ps) != list(reversed(sorted(ps))):
        print('Warning: ps vector had to be alrealy sorted in ascendent way')
        return []
    buckets_i = []
    buckets_p = []
    for _ in range(nb):
        buckets_i.append([])
        buckets_p.append(0)
    for i, p in enumerate(ps):
        min_bucket_i = buckets_p.index(np.min(buckets_p))
        buckets_i[min_bucket_i].append(i)
        buckets_p[min_bucket_i] += p
    return buckets_i


def online_equal_partitioner(nb, ps, ts):
    if list(ps) != list(reversed(sorted(ps))):
        print(
            'Warning: ps vector had to be alrealy sorted in ascendent way. Moreover, elements of ps and ts have to be aligned.')
        return []
    buckets_i = []
    buckets_t = []
    for i, t in enumerate(ts):
        if i < nb:
            buckets_i.append([i])
            buckets_t.append(t)
        else:
            min_bucket_i = buckets_t.index(np.min(buckets_t))
            buckets_i[min_bucket_i].append(i)
            buckets_t[min_bucket_i] += t
    return buckets_i


def classist_partitioner(nb, nc):
    combos, k = [], 0
    nb_max_complete = nc % nb
    nb_dim_complete = np.ceil(nc / nb)
    nb_max_partial = nb - nc % nb
    nb_dim_partial = np.floor(nc / nb)
    for _ in range(nb):
        combos.append([])
    k = 0
    nb_count_complete = 0
    nb_count_partial = 0
    for j in range(nc):
        combos[k].append(j)
        if k < nb_max_complete and len(combos[k]) == nb_dim_complete \
                or nb_max_complete <= k < nb_max_complete + nb_max_partial and len(combos[k]) == nb_dim_partial:
            if k % 2 != 0:
                combos[k] = list(reversed(combos[k]))
            k += 1
    return combos
