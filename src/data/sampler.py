import torch


class SubsetSampler(torch.utils.data.Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def get_sampler(max_samples=None):
    sampler = None
    if max_samples is not None:
        # this is a hack to get the subset sampler working
        # TODO: implement this as a wrapper to chosen sampler
        # TODO: make it random and not just the first samples (might be problem between clustering and DFR. sampling the same?)
        sampler = SubsetSampler(list(range(max_samples)))
        return sampler
    return sampler



