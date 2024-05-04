import torch.utils.data
from data.dataset import ImageFilelist
from data.dataset import get_transform

def create_dataloader(dataset, dataset_opt):
    return torch.utils.data.DataLoader(dataset,
                                       batch_size=dataset_opt['batch_size'],
                                       shuffle=dataset_opt['use_shuffle'],
                                       num_workers=dataset_opt['n_workers'],
                                       drop_last=True,
                                       pin_memory=True)
def create_val_dataset(dataset_opt):
    dataset = ade20k(dataset_opt)
    return dataset


def create_dataset(dataset_opt, subset):
    dataset = ImageFilelist(dataset_opt, transform=get_transform(dataset_opt, subset))
    return dataset
