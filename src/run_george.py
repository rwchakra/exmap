import os 
import numpy as np

import exmap

from run_exmap import get_dataloader, get_model, set_seed
import exmap.create_groups as create_groups


def main(args):
    set_seed(args.seed)

    loader = get_dataloader(args)
    model = get_model(args, loader.dataset.n_classes)

    heatmaps, ys, gs, preds = exmap.feature_extraction(args, model, loader)

    print(args.downsize)
    clust_y0, clust_y1, clust_all = exmap.clustering(args, heatmaps, ys, gs)

    local_groups = create_groups.local_groups_from_clusters(ys, clust_y0, clust_y1)
    global_groups = create_groups.global_groups_from_clusters(ys, clust_all)

    np.save(
        os.path.join(args.results_dir, "local_cluster_estimation.npy"), local_groups
    )
    np.save(
        os.path.join(args.results_dir, "global_cluster_estimation.npy"), global_groups
    )
    # np.savez(os.path.join(args.results_dir, f"heatmaps.npz"), *[arr for arr in heatmaps[:args.max_samples]])

    print(np.unique(local_groups, return_counts=True))
    print(np.unique(global_groups, return_counts=True))

    exmap.plot_data(args, heatmaps, ys, preds, gs, local_groups, global_groups)