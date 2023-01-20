
from .image import load_galaxy,load_cifar10, load_mimic,iterate_minibatches, get_batch, binarize_data, binarize_image
from .image import preprocess, postprocess, permute, get_permute_all, get_permute_matrix, logit_transform, sigmoid_transform

from .patient import MIMIC

def load_datasets(dataset, data_path, **kwargs):
    
    if dataset == 'cifar10':
        return load_cifar10(data_path)
        
    elif dataset == 'galaxy':
        return load_galaxy('./galaxy.pkl', 32)
    elif dataset == 'mimic':
        return load_mimic(data_path, **kwargs)
        
    else:
        raise ValueError('unknown data set %s' % dataset)
