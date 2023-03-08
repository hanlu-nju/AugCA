import random
from typing import Callable, Iterable, List

import attr
import torch
from PIL import ImageFilter, Image
from torchvision import transforms

from .randaugment import rand_augment_transform


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class GaussianNoise:
    def __init__(self, std=0.15):
        self.std = std

    def __call__(self, x):
        with torch.no_grad():
            return x + torch.randn_like(x) * self.std

    def __repr__(self):
        return f"GaussianNoise(std={self.std})"


def get_raw_transform(crop, mean, std, **kwargs):
    transforms_list = [transforms.CenterCrop(crop),
                       transforms.ToTensor(),
                       transforms.Normalize(mean=mean, std=std)]
    return transforms.Compose(transforms_list)


def get_randaug_transform(crop, mean, std, n=2, m=10, **kwargs):
    transform_list = [
        rand_augment_transform(f'rand-m{m}-n{n}', dict()),
        transforms.RandomResizedCrop(crop),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
    return transforms.Compose(transform_list)


def get_simclr_transform(crop, mean, std, s=0.5, ratio=0.2, apply_blur: bool = False, use_momentum: bool = False,
                         **kwargs):
    # if use_momentum:
    # memory bank likes 0.08
    # ratio = 0.08
    # else:
    # moco cache likes 0.2
    transform_list = [
        transforms.RandomResizedCrop(crop, scale=(ratio, 1.0), interpolation=Image.BICUBIC),
        transforms.RandomApply(
            [transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)], p=0.8
        ),
        transforms.RandomGrayscale(p=0.1),
        transforms.RandomHorizontalFlip(p=0.5),
        # transforms.RandomRotation()
    ]
    if apply_blur:
        transform_list.append(transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5))
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(mean=mean, std=std))
    train_transform = transforms.Compose(transform_list)
    # transform = DoubleTransform(train_transform, train_transform)
    return train_transform


@attr.s(auto_attribs=True)
class ContrastiveTransform:
    sup_trans: Callable = None
    train_trans_1: Callable = None
    train_trans_2: Callable = None
    val_trans: Callable = None
    test_trans: Callable = None

    def get_supervised_transform(self):
        return self.sup_trans

    def get_unsupervised_transform(self):
        return DoubleTransform(self.train_trans_1, self.train_trans_2)

    def get_val_transform(self):
        return self.val_trans

    def get_test_transform(self):
        return self.test_trans


@attr.s(auto_attribs=True, slots=True)
class StackTransforms:
    ts: Iterable[Callable] = None

    def __call__(self, x):
        x_s = [t(x) for t in self.ts]
        return torch.cat(x_s, dim=0)


@attr.s(auto_attribs=True, slots=True)
class DoubleTransform:
    t1: Callable = None
    t2: Callable = None

    def __call__(self, x):
        return torch.stack((self.t1(x), self.t2(x)))


@attr.s(auto_attribs=True, slots=True)
class RepeatTransform:
    t: Callable = None
    repeat: int = 1

    def __call__(self, x):
        return torch.stack([self.t(x) for _ in range(self.repeat)])


@attr.s(auto_attribs=True, slots=True)
class SequenceTransform:
    ts: List[Callable] = None

    def __call__(self, x):
        return torch.stack([t(x) for t in self.ts])
