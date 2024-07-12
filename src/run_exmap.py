import argparse
import torch
import numpy as np
import os
from torch.utils.data import DataLoader

import exmap
import models
import data
import exmap.create_groups as create_groups


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=False,
        default="imagenet_resnet50",
        choices=[
            "imagenet_resnet50",
            "cifar_resnet18",
            "imagenet_resnet50_pretrained",
            "imagenet_resnet18_pretrained",
            "imagenet_resnet34_pretrained",
            "imagenet_resnet101_pretrained",
            "imagenet_resnet152_pretrained",
            "imagenet_densenet121_pretrained",
            "imagenet_densenet121",
            "imagenet_vgg19_pretrained",
            "imagenet_vgg16_pretrained",
            "imagenet_alexnet_pretrained",
        ],
        help="Base model",
    )
    parser.add_argument(
        "--ckpt_path", type=str, required=True, help="Checkpoint path for model weights"
    )
    parser.add_argument("--seed", type=int, default=1, help="Random seed to use")
    parser.add_argument(
        "--base_dir", type=str, required=True, help="Base directory for dataset"
    )
    parser.add_argument(
        "--dataset", 
        type=str, 
        required=True, 
        choices=["WaterbirdsDataset",
                 "CelebADataset", 
                 "FGWaterbirdsDataset", 
                 "CMNISTDataset", 
                 "BothUrbancarsDataset", 
                 "BgUrbancarsDataset", 
                 "CoOccurObjUrbancarsDataset"
        ], help="Dataset type to use.")
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val", "test"],
        help="split to get pseudo group labels for.",
    )
    parser.add_argument(
        "--max_samples", type=int, default=None, help="Max number of samples to use"
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for data loader"
    )
    parser.add_argument(
        "--downsize", type=int, default=None, help="Downsize heatmaps to this size"
    )

    parser.add_argument(
        "--clustering_type", type=str, default="spectral", help="Clustering method to use"
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Automatically find number of clusters using eigengap among (max 10)",
    )
    parser.add_argument(
        "--clusters_class_0",
        type=int,
        default=2,
        help="Number of clusters for class 0 (if 'auto'=False)",
    )
    parser.add_argument(
        "--clusters_class_1",
        type=int,
        default=2,
        help="Number of clusters for class 1 (if 'auto'=False)",
    )
    parser.add_argument(
        "--clusters_class_all",
        type=int,
        default=2,
        help="Number of clusters for all datapoints (if 'auto'=False)",
    )

    parser.add_argument(
        "--results_dir", type=str, default="", help="Location to store results"
    )
    parser.add_argument(
        "--plot_type", type=str, default="tsne", choices=["tsne", "umap"], help="Type of plot to showcase clustering"
    )

    args = parser.parse_args()
    return args


def set_seed(seed):
    """Sets seed"""
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_transform(args):
    augment = False
    if args.dataset in ["WaterbirdsDataset", "CelebADataset", "FGWaterbirdsDataset"]:
        transform = data.BaseTransform(augment)
    elif args.dataset == "CMNISTDataset":
        transform = None
    elif "UrbancarsDataset" in args.dataset:
        transform = data.UrbancarsTransform(augment)
    return transform
        

def get_dataloader(args):
    transform = get_transform(args)
    dataset_class = getattr(data, args.dataset)
    dataset = dataset_class(
        basedir=args.base_dir, split=args.split, transform=transform
    )
    
    if (args.max_samples is not None) and (args.max_samples > len(dataset)):
        args.max_samples = len(dataset)
    
    sampler = data.get_sampler(args.max_samples)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=12,
        pin_memory=True,
    )
    return loader


def get_model(args, n_classes):
    model_cls = getattr(models, args.model)
    model = model_cls(n_classes)
    model.load_state_dict(torch.load(args.ckpt_path))
    model.cuda()
    model.eval()
    return model


def main(args):
    set_seed(args.seed)

    loader = get_dataloader(args)
    model = get_model(args, loader.dataset.n_classes)

    heatmaps, ys, gs, preds = exmap.lrp(args, model, loader)

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


if __name__ == "__main__":
    args = get_args_parser()
    main(args)
