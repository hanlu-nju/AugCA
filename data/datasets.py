import os
from typing import Union, List, Optional, Tuple, Callable, Iterable

import numpy as np
from PIL import Image
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, STL10, ImageFolder

from data import transformation
from utils import is_debug


class BaseDataset(Dataset):
    def __init__(self, setname, transform=None, target_transform=None) -> None:
        self.setname = setname
        self.data, self.labels = self.set_data()
        self.wnids = np.unique(self.labels)
        self.num_class = len(self.wnids)
        self.transform = transform
        self.target_transform = target_transform

    @property
    def data_dir(self) -> Union[str, List[str]]:
        raise NotImplementedError

    def set_data(self):
        raise NotImplementedError

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        if self.labels is not None:
            img, target = self.data[index], int(self.labels[index])
        else:
            img, target = self.data[index], None

        if isinstance(img, str):
            img = Image.open(img).convert('RGB')
        elif isinstance(img, np.ndarray):
            img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)


class DatasetBase:
    seed: int = 42

    def __init__(self, data_dir, train_type='unsup', **kwargs) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.train_type = train_type

    def configure_dataset(self):
        pass

    def get_train(self, transform=None):
        raise NotImplementedError

    def get_clf(self, transform=None):
        raise NotImplementedError

    def get_val(self, transform=None):
        raise NotImplementedError

    def get_test(self, transform=None):
        raise NotImplementedError

    @property
    def name(self) -> str:
        raise NotImplementedError

    @property
    def dim(self):
        raise NotImplementedError

    @property
    def num_total_classes(self):
        raise NotImplementedError

    @property
    def num_label_classes(self):
        return self.num_total_classes

    def test_transform(self, crop, mean, std, **kwargs):
        return transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize(mean,
                                                        std)])

    def linear_transform(self, crop, mean, std, **kwargs):
        return transforms.Compose([transforms.RandomResizedCrop(crop),
                                   transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean,
                                                        std)])


class CIFAR10Dataset(DatasetBase):

    def configure_dataset(self):
        self.train = CIFAR10(self.data_dir, train=True, download=True)
        self.test = CIFAR10(self.data_dir, train=False, download=True)

    def get_train(self, transform=None):
        return CIFAR10(self.data_dir, train=True, download=False, transform=transform)

    def get_clf(self, transform=None):
        return self.get_train(transform)

    def get_val(self, transform=None):
        return CIFAR10(self.data_dir, train=False, download=False, transform=transform)

    def get_test(self, transform=None):
        return CIFAR10(self.data_dir, train=False, download=False, transform=transform)

    @property
    def name(self) -> str:
        return 'cifar10'

    @property
    def dim(self):
        return 3, 32, 32

    @property
    def num_total_classes(self):
        return 10


class CIFAR100Dataset(DatasetBase):

    def configure_dataset(self):
        self.train = CIFAR100(self.data_dir, train=True, download=True)
        self.test = CIFAR100(self.data_dir, train=False, download=True)

    def get_clf(self, transform=None):
        return self.get_train(transform)

    def get_train(self, transform=None):
        return CIFAR100(self.data_dir, train=True, download=False, transform=transform)

    def get_val(self, transform=None):
        return CIFAR100(self.data_dir, train=False, download=False, transform=transform)

    def get_test(self, transform=None):
        return CIFAR100(self.data_dir, train=False, download=False, transform=transform)

    @property
    def name(self) -> str:
        return 'cifar100'

    @property
    def dim(self):
        return 3, 32, 32

    @property
    def num_total_classes(self):
        return 100


class STL10Dataset(DatasetBase):

    def configure_dataset(self):
        self.train = STL10(self.data_dir, split='train+unlabeled', download=True)
        STL10(self.data_dir, split='train', download=True)
        self.test = STL10(self.data_dir, split='test', download=True)

    def get_clf(self, transform=None):
        return STL10(self.data_dir, split='train', download=False, transform=transform)

    def get_train(self, transform=None):
        if self.train_type == 'sup':
            return STL10(self.data_dir, split='train', download=False, transform=transform)
        else:
            return STL10(self.data_dir, split='train+unlabeled', download=False, transform=transform)

    def get_val(self, transform=None):
        return STL10(self.data_dir, split='test', download=False, transform=transform)

    def get_test(self, transform=None):
        return STL10(self.data_dir, split='test', download=False, transform=transform)

    @property
    def name(self) -> str:
        return 'stl10'

    @property
    def dim(self):
        return 3, 64, 64

    @property
    def num_total_classes(self):
        return 10

    def test_transform(self, crop, mean, std, **kwargs):
        return transforms.Compose([transforms.Resize(70, interpolation=3),
                                   transforms.CenterCrop(crop),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean,
                                                        std)])


class TinyImageNetDataset(DatasetBase):

    def get_clf(self, transform=None):
        return self.get_train(transform)

    def get_train(self, transform=None):
        return ImageFolder(root=os.path.join(self.data_dir, 'train'), transform=transform)

    def get_val(self, transform=None):
        return ImageFolder(root=os.path.join(self.data_dir, 'val', 'images'), transform=transform)

    def get_test(self, transform=None):
        return ImageFolder(root=os.path.join(self.data_dir, 'val', 'images'), transform=transform)

    @property
    def name(self) -> str:
        return 'tiny-imagenet'

    @property
    def dim(self):
        return 3, 64, 64

    @property
    def num_total_classes(self):
        return 200


class RawDataset(DatasetBase):

    def configure_dataset(self):
        self.base_dataset.configure_dataset()

    @property
    def num_total_classes(self):
        return self.num_class

    def __init__(self, data_dir, base_dataset: DatasetBase, train_type='unsup',
                 **kwargs) -> None:
        super().__init__(data_dir, train_type, **kwargs)
        self.base_dataset = base_dataset
        self.num_class = self.base_dataset.num_total_classes

    def get_train(self, transform: Tuple[Callable] = None):
        return self.base_dataset.get_train(transform)

    def get_val(self, transform: Union[Callable, Tuple[Callable]] = None):
        if isinstance(transform, Callable):
            return self.base_dataset.get_test(transform), \
                   self.base_dataset.get_clf(transform)
        else:
            return self.base_dataset.get_test(transform[0]), \
                   self.base_dataset.get_clf(transform[1])

    def get_test(self, transform: Tuple[Callable] = None):
        return self.get_val(transform)

    @property
    def name(self) -> str:
        return self.base_dataset.name

    @property
    def dim(self):
        return self.base_dataset.dim


class SelfSupervisedDataModule(LightningDataModule):

    def __init__(self, args, dataset: DatasetBase, train_transforms=None, linear_transforms=None,
                 test_transforms=None, num_workers: int = 4,
                 batch_size: int = 32, **kwargs):
        super().__init__()
        self.args = args
        self.dataset = dataset
        self.num_workers = num_workers
        self.batch_size = batch_size

        self.train_transform = train_transforms
        self.linear_transform = linear_transforms
        self.test_transform = test_transforms

    def prepare_data(self, *args, **kwargs):
        self.dataset.configure_dataset()

    def setup(self, stage: Optional[str] = None):
        pass

    def train_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        ds = self.dataset.get_train(self.train_transform)
        if isinstance(ds, Iterable):
            loaders = []
            for d in ds:
                loaders.append(DataLoader(
                    d,
                    batch_size=self.batch_size,
                    shuffle=True,
                    num_workers=self.num_workers,
                    drop_last=False,
                    pin_memory=True
                ))
            return loaders
        else:
            return DataLoader(
                ds,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                drop_last=True,
                pin_memory=True
            )

    def val_dataloader(self, *args, **kwargs):
        return self.get_val_loaders()

    def get_val_loaders(self):
        ds = self.dataset.get_val((self.test_transform, self.linear_transform))
        if isinstance(ds, Iterable):
            loaders = []
            for d in ds:
                loaders.append(DataLoader(
                    d,
                    batch_size=self.batch_size,
                    shuffle=True,
                    num_workers=self.num_workers,
                    drop_last=False,
                    pin_memory=True
                ))
            return loaders
        else:
            return DataLoader(
                ds,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                drop_last=False,
                pin_memory=True
            )

    def test_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return self.get_test_loaders()

    def get_test_loaders(self):
        ds = self.dataset.get_test((self.test_transform, self.linear_transform))
        if isinstance(ds, Iterable):
            loaders = []
            for d in ds:
                loaders.append(DataLoader(
                    d,
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=self.num_workers,
                    drop_last=False,
                    pin_memory=True
                ))
            return loaders
        else:
            return DataLoader(
                ds,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                drop_last=False,
                pin_memory=True
            )


if is_debug():
    debug_param = {'num_workers': 0}
else:
    debug_param = {}

dataset_dict = {
    'cifar10': {'module': CIFAR10Dataset,
                'crop': 32,
                'mean': (0.4914, 0.4822, 0.4465),
                'std': (0.2023, 0.1994, 0.2010),
                'dir_name': 'cifar10',
                'params': {
                    **debug_param
                }},
    'cifar100': {'module': CIFAR100Dataset,
                 'crop': 32,
                 'mean': (0.5071, 0.4867, 0.4408),
                 'std': (0.2675, 0.2565, 0.2761),
                 'dir_name': 'cifar100',
                 'params': {
                     **debug_param
                 }},
    'tiny-imagenet': {'module': TinyImageNetDataset,
                      'crop': 64,
                      'mean': (0.480, 0.448, 0.398),
                      'std': (0.277, 0.269, 0.282),
                      'dir_name': 'tiny-imagenet',
                      'params': {
                          **debug_param
                      }},
    'stl10': {'module': STL10Dataset,
              'crop': 64,
              'mean': (0.43, 0.42, 0.39),
              'std': (0.27, 0.26, 0.27),
              'dir_name': 'stl10',
              'params': {
                  **debug_param
              }},
}


def get_data_module(args):
    dataset_info = dataset_dict[args.dataset]
    data_dir = args.data_dir
    dataset = dataset_info['module'](
        data_dir=data_dir,
        train_type=args.type,
        **dataset_info['params'],
    )
    train_transform = getattr(transformation, f'get_{args.train_transform}_transform')(**dataset_info)
    linear_transform = dataset.test_transform(**dataset_info)
    test_transform = dataset.test_transform(**dataset_info)
    if args.multi_augment > 1 or args.with_anchor:
        augments = [train_transform] * args.multi_augment
        if args.with_anchor:
            augments = [test_transform] + augments
        train_transform = transformation.SequenceTransform(augments)

    dataset = RawDataset(data_dir='', base_dataset=dataset)
    data_module = SelfSupervisedDataModule(args, dataset, batch_size=args.batch_size,
                                           train_transforms=train_transform,
                                           linear_transforms=linear_transform,
                                           test_transforms=test_transform,
                                           **dataset_info['params'])
    return data_module
