import torch
import numpy as np
from torchvision import transforms

def get_dataset(args, dataset_name, phase):
    if dataset_name == 'cub':
        from torchmeta.datasets.helpers import cub as dataset_helper
        image_size = 84
        padding_len = 8
    elif dataset_name == 'miniimagenet':
        from torchmeta.datasets.helpers import miniimagenet as dataset_helper
        image_size = 84
        padding_len = 8
    elif dataset_name == 'omniglot':
        from torchmeta.datasets.helpers import omniglot as dataset_helper
        image_size = 28
        padding_len = 8
    else:
        raise ValueError('Non-supported Dataset.')

    # augmentations
    # 参考：https://github.com/Sha-Lab/FEAT
    if args.augment and phase == 'train':
        transforms_list = [
            transforms.RandomResizedCrop(image_size),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    else:
        transforms_list = [
            transforms.Resize(image_size+padding_len),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ]

    # pre-processing 
    if args.backbone == 'resnet12':
        transforms_list = transforms.Compose(
                transforms_list + [
                transforms.Normalize(np.array([x / 255.0 for x in [120.39586422,  115.59361427, 104.54012653]]),
                                     np.array([x / 255.0 for x in [70.68188272,   68.27635443,  72.54505529]]))
            ])

    else:
        transforms_list = transforms.Compose(
                transforms_list + [
                transforms.Normalize(np.array([0.485, 0.456, 0.406]),
                                     np.array([0.229, 0.224, 0.225]))
            ])


    # get datasets
    dataset = dataset_helper(
        args.data_folder,
        shots=args.num_shots,
        ways=args.num_ways,
        shuffle=(phase == 'train'),
        test_shots=args.test_shots,
        meta_split=phase,
        download=args.download,
        transform=transforms_list
    )

    return dataset

