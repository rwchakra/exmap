import os
import numpy as np
import pandas as pd
from PIL import Image
import torch

from torch.utils.data import Dataset
from torchvision.datasets import MNIST
from matplotlib.colors import to_rgb

from .urbancars_original import UrbanCars


def _get_split(split):
    try:
        return ["train", "val", "test"].index(split)
    except ValueError:
        raise (f"Unknown split {split}")


def _cast_int(arr):
    if isinstance(arr, np.ndarray):
        return arr.astype(int)
    elif isinstance(arr, torch.Tensor):
        return arr.int()
    else:
        raise NotImplementedError


class SpuriousCorrelationDataset(Dataset):
    def __init__(self, basedir, split="train", transform=None):
        self.basedir = basedir
        self.metadata_df = self._get_metadata(split)

        self.transform = transform
        self.y_array = self.metadata_df["y"].values
        if "spurious" in self.metadata_df:
            self.spurious_array = self.metadata_df["spurious"].values
        else:
            self.spurious_array = self.metadata_df["place"].values
        self._count_attributes()
        if "group" in self.metadata_df:
            self.group_array = self.metadata_df["group"].values
        else:
            self._get_class_spurious_groups()
        self._count_groups()
        self.text = not "img_filename" in self.metadata_df
        if self.text:
            print("NLP dataset")
            self.text_array = list(
                pd.read_csv(os.path.join(basedir, "text.csv"))["text"]
            )
        else:
            self.filename_array = self.metadata_df["img_filename"].values

    def _get_metadata(self, split):
        split_i = _get_split(split)
        metadata_df = pd.read_csv(os.path.join(self.basedir, "metadata.csv"))
        metadata_df = metadata_df[metadata_df["split"] == split_i]
        return metadata_df

    def _count_attributes(self):
        self.n_classes = np.unique(self.y_array).size
        self.n_spurious = np.unique(self.spurious_array).size
        self.y_counts = self._bincount_array_as_tensor(self.y_array)
        self.spurious_counts = self._bincount_array_as_tensor(self.spurious_array)

    def _count_groups(self):
        self.group_counts = self._bincount_array_as_tensor(self.group_array)
        # self.n_groups = np.unique(self.group_array).size
        self.n_groups = len(self.group_counts)

    def _get_class_spurious_groups(self):
        self.group_array = _cast_int(
            self.y_array * self.n_spurious + self.spurious_array
        )

    @staticmethod
    def _bincount_array_as_tensor(arr):
        return torch.from_numpy(np.bincount(arr)).long()

    def __len__(self):
        return len(self.metadata_df)

    def __getitem__(self, idx):
        y = self.y_array[idx]
        g = self.group_array[idx]
        s = self.spurious_array[idx]
        if self.text:
            x = self._text_getitem(idx)
        else:
            x = self._image_getitem(idx)
        return x, y, g, s

    def _image_getitem(self, idx):
        img_path = os.path.join(self.basedir, self.filename_array[idx])
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img

    def _text_getitem(self, idx):
        text = self.text_array[idx]
        if self.transform:
            text = self.transform(text)
        return text
    

class WaterbirdsDataset(SpuriousCorrelationDataset):
    def __init__(self, basedir, split="train", transform=None):
        super().__init__(basedir, split, transform)


class CelebADataset(SpuriousCorrelationDataset):
    def __init__(self, basedir, split="train", transform=None):
        super().__init__(basedir, split, transform)


class FGWaterbirdsDataset(SpuriousCorrelationDataset):
    """Requires masks to be stored together with images using the same filename, but in png format.
        The masks can be found here: https://data.caltech.edu/records/w9d68-gec53"""
    def __init__(self, basedir, split="train", transform=None):
        super().__init__(basedir, split, transform)
        self.split_id = _get_split(split)
        self.mask_filename_array = [os.path.splitext(filename)[0] + ".png" for filename in self.filename_array]

    def _image_getitem(self, idx):
        img_path = os.path.join(self.basedir, self.filename_array[idx])
        img = Image.open(img_path).convert("RGB")
        if self.split_id == 2:
            mask_path = os.path.join(self.basedir, self.mask_filename_array[idx])
            mask = Image.open(mask_path).convert("RGB")
            # multiply image with thresholded mask
            img = Image.fromarray((np.array(img) * (np.array(mask) > 200)).astype("uint8"))

        if self.transform:
            img = self.transform(img)
        return img


class CMNISTDataset(SpuriousCorrelationDataset):
    """Colored MNIST dataset with SpuriousCorrelationDataset API.
    
    Colors with given correlation are used as spurious attributes.
    """
    def __init__(self, basedir, split, transform=None, color_correlation=0.99, val_size=10000, seed=0):
        split_i = _get_split(split)
        self.ds = MNIST(
            root=basedir, train=(split_i != 2),
            download=True, transform=None)
        if split_i == 0:
            self.ds.data = self.ds.data[:-val_size]
            self.ds.targets = self.ds.targets[:-val_size]
        elif split_i == 1:
            self.ds.data = self.ds.data[-val_size:]
            self.ds.targets = self.ds.targets[-val_size:]

        np.random.seed(seed)

        self.y_array = (np.array(self.ds.targets) > 4).astype(int)
        self.n_classes = 2
        self.n_spurious = 2

        self.spurious_array = np.zeros_like(self.y_array)
        self.spurious_array[self.y_array == 0] = np.random.binomial(1, color_correlation, size=(self.y_array == 0).sum())
        self.spurious_array[self.y_array == 1] = np.random.binomial(1, 1 - color_correlation, size=(self.y_array == 1).sum())
        self._get_class_spurious_groups()
        self._count_groups()

        self.color_correlation = color_correlation
        self.text = False
        self.color_replaced = False

        self.ds.data = self._data_prep(self.ds.data) 

    def _data_prep(self, data):
        spurious_tensor = torch.tensor(self.spurious_array)
        data = data / 255.
        new_data = torch.zeros(data.shape[0], 3, data.shape[1], data.shape[2]).float()
        new_data[spurious_tensor==0, 0] = data[spurious_tensor==0]
        new_data[spurious_tensor==1, 1] = data[spurious_tensor==1]
        return new_data

    
    def _color_mask(self, data, color, mask):
        """Replace pixels in data where mask is true with color."""
        color = to_rgb(color) # either tuple in range 0-1 or string. List of strings: https://matplotlib.org/stable/gallery/color/named_colors.html 

        for i, c in enumerate(color):
            if not self.color_replaced: data[:, i, :, :] = c * mask
            else: data[:, i, :, :] += c * mask
        
        self.color_replaced = True
        return data

    def __getitem__(self, idx):
        y = self.y_array[idx]
        g = self.group_array[idx]
        s = self.spurious_array[idx]
        x = self.ds.data[idx]
        return x, y, g, s

    def __len__(self):
        return len(self.y_array)


class BaseUrbancarsDataset(SpuriousCorrelationDataset):
    def __init__(self, basedir, split="train", transform=None, group_label="both"):
        self.basedir = basedir

        self.dataset = UrbanCars(basedir, split, transform=transform, group_label=group_label)
        self.transform = transform

        self.y_array = self.dataset.obj_label.numpy()
        self.spurious_array = self.dataset.domain_label.numpy()
        self.group_array = self.dataset.group_array.numpy()
        self._count_attributes()
        self._count_groups()

        self.filename_array = self.dataset.img_fpath_list
        self.text = False

    def __len__(self):
        return len(self.dataset)

    def _image_getitem(self, idx):
        img_path = self.filename_array[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img

class BothUrbancarsDataset(BaseUrbancarsDataset):
    def __init__(self, basedir, split="train", transform=None):
        super().__init__(basedir, split, transform, group_label="both")

class BgUrbancarsDataset(BaseUrbancarsDataset):
    def __init__(self, basedir, split="train", transform=None):
        super().__init__(basedir, split, transform, group_label="bg")

class CoOccurObjUrbancarsDataset(BaseUrbancarsDataset):
    def __init__(self, basedir, split="train", transform=None):
        super().__init__(basedir, split, transform, group_label="co_occur_obj")


