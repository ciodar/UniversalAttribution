from torchvision import transforms
import numpy as np
from PIL import Image

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
OPENAI_DEFAULT_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DEFAULT_STD = (0.26862954, 0.26130258, 0.27577711)

try:
    from torchvision.transforms import InterpolationMode, ToTensor

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


def get_transform(config):
    normalize = transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN,
                                     std=IMAGENET_DEFAULT_STD) if config.backbone.startswith(
        'openai/clip-') else transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    transform_type = config.transform
    resize_size = config.resize_size
    
    if transform_type == 'default':
        train_transform = transforms.Compose([
            transforms.Resize(resize_size, interpolation=BICUBIC),
            transforms.CenterCrop(config.resize_size[0]),
            transforms.ToTensor(),
            normalize
        ])

        test_transform = transforms.Compose([
            transforms.Resize(resize_size),
            transforms.CenterCrop(config.resize_size[0]),
            transforms.ToTensor(),
            normalize
        ])

    else:

        raise NotImplementedError

    return (train_transform, test_transform)