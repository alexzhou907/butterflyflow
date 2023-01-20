import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset


# image downsampling function from emerging conv, only works in batches
def downsample(x, resolution):
    assert x.dtype == np.float32
    assert x.shape[1] % resolution == 0
    assert x.shape[2] % resolution == 0
    if x.shape[1] == x.shape[2] == resolution:
        return x
    s = x.shape
    x = np.reshape(x, [s[0], resolution, s[1] // resolution,
                       resolution, s[2] // resolution, s[3]])
    x = np.mean(x, (2, 4))
    return x


class GalaxyDataset(Dataset):
    """Defines the base dataset class.

    This class supports loading data from a full-of-image folder, a lmdb
    database, or an image list. Images will be pre-processed based on the given
    `transform` function before fed into the data loader.

    NOTE: The loaded data will be returned as a directory, where there must be
    a key `image`.
    """
    def __init__(self,
                 root_dir,
                 split,
                 resolution,
                 transform=None,
                 transform_kwargs=None,
                 **_unused_kwargs):
        """Initializes the dataset.

        Args:
            root_dir: Root directory containing the dataset.
            split: [train, val]
            resolution: The resolution of the returned image.
            transform: The transform function for pre-processing.
                (default: `datasets.transforms.normalize_image()`)
            transform_kwargs: The additional arguments for the `transform`
                function. (default: None)
        """
        self.root_dir = root_dir
        self.resolution = resolution
        self.transform = transform
        self.transform_kwargs = transform_kwargs or dict()

        with open(self.root_dir, 'rb') as fp:
          X_train, X_val, X_test = pickle.load(fp)
    
        # combine val and test set for val_data
        X_valid = np.vstack([X_val, X_test])

        if split == 'train':
          self.data = torch.from_numpy(X_train)
        else:
          self.data = torch.from_numpy(X_valid)
        self.num_samples = len(self.data)

    def __len__(self):
        assert self.num_samples == 5000 
        return self.num_samples

    def __getitem__(self, idx):
        image = self.data[idx]

        if self.transform is not None:
          image = self.transform(image, **self.transform_kwargs)
        else:
          # test data
          image = image / 255.
        
        # TODO: account for downsampling
        # TODO: also, do we need additional preprocess/postprocessing?

        return image, 0.  # fake label