import numpy as np
from sklearn.cluster import SpectralClustering, KMeans
import matplotlib.pyplot as plt
from exmap.eigen import eigenDecomposition
import umap 


def spectral(args, data, num_clusters, name):
    # create pipeline
    N = len(data)
    k_knn = int(np.log(len(data)))

    pipeline = SpectralClustering(
        n_clusters=num_clusters,
        affinity="nearest_neighbors",
        n_neighbors=k_knn,
        n_jobs=-1,
        random_state=args.seed,
        assign_labels="cluster_qr",
        eigen_solver="lobpcg",
    )
    if args.auto:
        # fit to get affinity matrix
        affinity_matrix = pipeline.fit(data).affinity_matrix_

        _, eigv, _ = eigenDecomposition(
            affinity_matrix, seed=args.seed, plot=name,
        )

        # find largest eigengap among 10 first eigenvalues
        num_clusters = np.diff(eigv)[1:10].argmax() + 2
        # redo with new number of clusters
        pipeline = SpectralClustering(
            n_clusters=num_clusters,
            affinity="precomputed",
            n_neighbors=k_knn,
            n_jobs=-1,
            random_state=args.seed,
            assign_labels="cluster_qr",
            eigen_solver="lobpcg",
        )
        data = affinity_matrix

    if num_clusters == 1:
        cluster_assign = np.zeros(N, dtype=int)
    else:
        cluster_assign = pipeline.fit_predict(data)

    return cluster_assign


def kmeans(args, data, num_clusters, do_umap=False):
    if do_umap:
        n_neigh = int(np.log(len(data)))
        data = umap.UMAP(n_components=2, n_neighbors=n_neigh, n_jobs=1).fit_transform(data)
    pipeline = KMeans(
        n_clusters=num_clusters,
        n_init=10,
        max_iter=300,
        tol=1e-4,
        random_state=args.seed,
    )
    cluster_assign = pipeline.fit_predict(data)

    return cluster_assign


def clustering(args, heatmaps, y, g):
    # divisions for clustering: cluster only points where y=1, y=0, all. Interesting to use predictions?
    types = {
        "y0": {"mask": y == 0, "k_clusters": args.clusters_class_0},
        "y1": {"mask": y == 1, "k_clusters": args.clusters_class_1},
        "all": {"mask": y == y, "k_clusters": args.clusters_class_all},
    }

    results = {}

    for key, value in types.items():
        mask = value["mask"]
        idx = np.arange(len(mask))[mask]
        k_clusters = value["k_clusters"]
        heatmaps_sub = [
            heatmaps[i] / np.abs(heatmaps[i]).max()
            for i in range(len(heatmaps))
            if (i == idx).any()
        ]

        # cluster
        if args.clustering_type == "spectral":
            cluster_assign = spectral(args, heatmaps_sub, k_clusters, name=key)
        elif args.clustering_type == "kmeans":
            cluster_assign = kmeans(args, heatmaps_sub, k_clusters)
        elif args.clustering_type == "umap_kmeans":
            cluster_assign = kmeans(args, heatmaps_sub, k_clusters, do_umap=True)
        else:
            raise ValueError("Clustering type not supported")

        results[key] = cluster_assign

        # print some cluster assignment result
        num_c = len(np.unique(cluster_assign))
        print(f"How much of each spurious type is in each cluster. {key}.".upper())
        tot = 0
        for i in range(num_c):
            counts = np.bincount(g[mask][cluster_assign == i])
            tot += counts.max()
            print(f"cluster {i}:", counts)
        print("accuracy:", tot / len(cluster_assign))

    clust_y0 = results["y0"]
    clust_y1 = results["y1"]
    clust_all = results["all"]
    plt.show()
    return clust_y0, clust_y1, clust_all
