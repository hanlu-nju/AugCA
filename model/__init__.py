from .base import ModelBase
from .simclr import SimCLR
from .ACA import AugmentationComponentAnalysis

model_dict = {
    'aca': AugmentationComponentAnalysis,
    'simclr': SimCLR,
}
