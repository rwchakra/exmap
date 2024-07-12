import torchvision.transforms as transforms

IMAGENET_STATS = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def _add_totensor_normalize(transform_lst, normalize_stats):
    transform_lst.append(transforms.ToTensor())
    if normalize_stats:
        transform_lst.append(transforms.Normalize(*normalize_stats))


class BaseTransform(transforms.Compose):
    def __init__(self, train=False, normalize_stats=IMAGENET_STATS):
        target_resolution = (224, 224)
        resize_resolution = (256, 256)
        self.transforms = []
        if train:
            self.transforms = [
                transforms.RandomResizedCrop(
                    target_resolution,
                    scale=(0.7, 1.0),
                    ratio=(0.75, 1.3333333333333333),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.RandomHorizontalFlip(),
            ]
        else:
            self.transforms = [
                transforms.Resize(resize_resolution),
                transforms.CenterCrop(target_resolution),
            ]
        _add_totensor_normalize(self.transforms, normalize_stats)

class UrbancarsTransform(transforms.Compose):
    def __init__(self, train, normalize_stats=IMAGENET_STATS):
        self.transforms = []
        if train:
            self.transforms = [transforms.RandomHorizontalFlip()]
        _add_totensor_normalize(self.transforms, normalize_stats)