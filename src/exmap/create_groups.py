# import os
import numpy as np

def local_groups_from_clusters(y, clusters_class0, clusters_class1):
    assert len(y[y==0]) == len(clusters_class0)
    assert len(y[y==1]) == len(clusters_class1)

    tot_clusters_class0 = np.max(clusters_class0) + 1

    new_groups = np.zeros_like(y)
    new_groups[y==0] = clusters_class0
    new_groups[y==1] = tot_clusters_class0 + clusters_class1

    return new_groups

def global_groups_from_clusters(y, clusters_classall):
    assert len(y) == len(clusters_classall)

    tot_clusters = np.max(clusters_classall[y==0]) + 1
    new_groups = np.zeros_like(y)
    new_groups[y==0] = clusters_classall[y==0]
    new_groups[y==1] = tot_clusters + clusters_classall[y==1]

    return new_groups
