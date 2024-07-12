"""Data utilities required for erm and dfr training."""

import numpy as np
import torch
from torch.utils.data.dataloader import default_collate
from functools import partial

from .sampler import SubsetSampler
from torch.utils.data.sampler import WeightedRandomSampler


def mixup_batch(batch, num_classes, alpha=0.2):
    x, y, g, s = batch
    y = torch.nn.functional.one_hot(y, num_classes)
    batch_size = x.size()[0]
    lam = np.random.beta(alpha, alpha, size=batch_size)
    lam_ = torch.from_numpy(lam).float()
    lam_x = lam_.float().reshape((-1, 1, 1, 1))
    lam_y = lam_.float().reshape((-1, 1))
    index = torch.randperm(batch_size)
    mixed_x = lam_x * x + (1 - lam_x) * x[index, :]
    mixed_y = lam_y * y + (1 - lam_y) * y[index, :]
    return mixed_x, mixed_y, g, s

def mixup_collate(batch, num_classes):
    collated = default_collate(batch)
    return mixup_batch(collated, num_classes)


def get_collate_fn(mixup, num_classes):
    if mixup:
        return partial(mixup_collate, num_classes=num_classes)
    else:
        return default_collate


def remove_minority_groups(trainset, num_remove):
    if num_remove == 0:
        return
    print("Removing minority groups")
    print("Initial groups", np.bincount(trainset.group_array))
    num_groups = np.bincount(trainset.group_array).size
    group_counts = trainset.group_counts
    minority_groups = np.argsort(group_counts.numpy())[:num_remove]
    idx = np.where(np.logical_and.reduce(
        [trainset.group_array != g for g in minority_groups], initial=True))[0]
    trainset.x_array = trainset.x_array[idx]
    trainset.y_array = trainset.y_array[idx]
    trainset.group_array = trainset.group_array[idx]
    trainset.spurious_array = trainset.spurious_array[idx]
    if hasattr(trainset, 'filename_array'):
        trainset.filename_array = trainset.filename_array[idx]
    trainset.metadata_df = trainset.metadata_df.iloc[idx]
    trainset.group_counts = torch.from_numpy(
            np.bincount(trainset.group_array, minlength=num_groups))
    print("Final groups", np.bincount(trainset.group_array))


def balance_groups(ds):
    print("Original groups", ds.group_counts)
    group_counts = ds.group_counts.long().numpy()
    min_group = np.min(group_counts)
    group_idx = [np.where(ds.group_array == g)[0]
        for g in range(ds.n_groups)]
    for idx in group_idx:
        np.random.shuffle(idx)
    group_idx = [idx[:min_group] for idx in group_idx]
    idx = np.concatenate(group_idx, axis=0)
    ds.y_array = ds.y_array[idx]
    ds.group_array = ds.group_array[idx]
    ds.spurious_array = ds.spurious_array[idx]
    ds.filename_array = ds.filename_array[idx]
    ds.metadata_df = ds.metadata_df.iloc[idx]
    ds.group_counts = torch.from_numpy(np.bincount(ds.group_array))
    print("Final groups", ds.group_counts)



def get_sampler_counts(counts, permutation=None):
    weights = sum(counts) / counts
    if permutation is not None:
        try:
            weights = weights[permutation]
        except:
            weights = weights[permutation.long()]
    return WeightedRandomSampler(weights, sum(counts).item(), replacement=True)


def get_sampler_training(data, args):
    sampler = None   
    if hasattr(args, 'max_samples'):
        # this is a hack to get the subset sampler working
        # TODO: implement this as a wrapper to chosen sampler
        # TODO: make it random and not just the first samples (might be problem between clustering and DFR. sampling the same?)
        if args.max_samples is not None:
            sampler = SubsetSampler(list(range(args.max_samples)))
            print("Subset sampler used. Sampler reweighting of groups, classes or spurious is ignored!")
            return sampler

    if args.reweight_groups:
        sampler = get_sampler_counts(data.group_counts, data.group_array)
    elif args.reweight_classes:
        sampler = get_sampler_counts(data.y_counts, data.y_array)
    elif args.reweight_spurious:
        sampler = get_sampler_counts(data.spurious_counts, data.spurious_array)
    return sampler